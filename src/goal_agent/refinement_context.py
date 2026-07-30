from __future__ import annotations

import json
import math
import re
from dataclasses import dataclass
from typing import Any

from .models import CriteriaDocument, RefinementMessage, RefinementSession, SetupProposal

# The model needs substantial headroom for OpenCode's own system prompt, tool calls,
# file reads, and the structured response.  This budget applies only to the text
# Goal Agent sends on stdin; it intentionally stays far below a typical 64k-token
# provider context window.
_NORMAL_INPUT_CHAR_BUDGET = 48_000
_COMPACT_INPUT_CHAR_BUDGET = 24_000
_RECENT_MESSAGE_COUNT = 8
_MAX_SUMMARY_CHARS = 12_000
_MAX_MESSAGE_CHARS = 4_000
_MAX_GOAL_CHARS = 8_000
_MAX_PROPOSAL_CHARS = 22_000
_MAX_CRITERIA_CHARS = 16_000
_MAX_LEGACY_CONTEXT_CHARS = 3_000


@dataclass(slots=True)
class RefinementPromptContext:
    transcript: str
    prompt_char_budget: int
    estimated_input_tokens: int
    compacted_message_count: int
    mode: str


def _clean_text(value: str, limit: int) -> str:
    value = re.sub(r"[ \t]+", " ", value.strip())
    if len(value) <= limit:
        return value
    omitted = len(value) - limit
    return value[:limit].rstrip() + f"\n… [{omitted} characters omitted]"


def _bounded_json(value: Any, limit: int) -> str:
    text = json.dumps(value, indent=2, ensure_ascii=False)
    if len(text) <= limit:
        return text
    # It is acceptable for this diagnostic context block to be truncated because
    # it is not parsed by Goal Agent.  The model is explicitly told the block may
    # be abbreviated and must return a complete replacement proposal.
    return text[:limit].rstrip() + f"\n… [{len(text) - limit} JSON characters omitted]"


def _compact_proposal(proposal: SetupProposal | None) -> dict[str, Any] | None:
    if proposal is None:
        return None
    data = proposal.model_dump(mode="json")
    data["refined_goal"] = _clean_text(str(data.get("refined_goal", "")), 6_000)
    data["goal_rationale"] = _clean_text(str(data.get("goal_rationale", "")), 2_000)
    data["assistant_message"] = _clean_text(str(data.get("assistant_message", "")), 2_000)
    data["clarifying_questions"] = [
        _clean_text(str(item), 1_200) for item in data.get("clarifying_questions", [])[:6]
    ]
    data["assumptions"] = [
        _clean_text(str(item), 1_000) for item in data.get("assumptions", [])[:12]
    ]
    compact_criteria: list[dict[str, Any]] = []
    for criterion in data.get("criteria", [])[:30]:
        item = dict(criterion)
        for key, limit in {
            "description": 1_500,
            "command": 1_500,
            "path": 1_000,
            "contains": 1_500,
            "judge_prompt": 4_000,
        }.items():
            if item.get(key) is not None:
                item[key] = _clean_text(str(item[key]), limit)
        item["evidence_paths"] = [
            _clean_text(str(path), 700) for path in item.get("evidence_paths", [])[:20]
        ]
        compact_criteria.append(item)
    data["criteria"] = compact_criteria
    data["criteria_quality_issues"] = data.get("criteria_quality_issues", [])[:20]
    return data


def _compact_criteria(criteria: CriteriaDocument) -> dict[str, Any]:
    proposal = SetupProposal(refined_goal="placeholder", criteria=criteria.criteria)
    compact = _compact_proposal(proposal) or {}
    return {"revision": criteria.revision, "criteria": compact.get("criteria", [])}


def _summarize_older_messages(messages: list[RefinementMessage]) -> str:
    if not messages:
        return ""
    lines = [
        "Earlier refinement decisions (deterministically compacted; newer messages appear below):"
    ]
    # User statements carry the authoritative decisions.  Assistant statements are
    # retained only in shorter form so unanswered questions and rationale survive.
    for message in messages:
        if message.role == "user":
            label = "USER DECISION / CORRECTION"
            limit = 1_400
        elif message.role == "assistant":
            label = "AI QUESTION / SUMMARY"
            limit = 650
        else:
            label = "SYSTEM NOTE"
            limit = 500
        lines.append(f"- {label}: {_clean_text(message.content, limit)}")
    text = "\n".join(lines)
    if len(text) <= _MAX_SUMMARY_CHARS:
        return text
    # Keep both the beginning (original intent) and the end (latest resolved
    # decisions) rather than keeping only one side.
    head = text[: _MAX_SUMMARY_CHARS // 2]
    tail = text[-(_MAX_SUMMARY_CHARS // 2) :]
    return head.rstrip() + "\n… [middle of compacted history omitted] …\n" + tail.lstrip()


def refresh_session_compaction(session: RefinementSession) -> RefinementSession:
    """Update the persisted rolling summary without deleting UI-visible messages."""

    cut = max(0, len(session.messages) - _RECENT_MESSAGE_COUNT)
    if cut:
        session.conversation_summary = _summarize_older_messages(session.messages[:cut])
        session.compacted_message_count = cut
    else:
        session.conversation_summary = ""
        session.compacted_message_count = 0
    return session


def build_refinement_context(
    *,
    session: RefinementSession,
    saved_goal: str,
    saved_criteria: CriteriaDocument,
    mode: str,
    legacy_context: str = "",
    aggressive: bool = False,
) -> RefinementPromptContext:
    """Build a bounded prompt context for one refinement turn.

    OpenCode adds its own system/tool history after this input.  Keeping Goal
    Agent's contribution bounded leaves room for project inspection and prevents
    iterative refinement from consuming the provider's entire context window.
    """

    refresh_session_compaction(session)
    char_budget = _COMPACT_INPUT_CHAR_BUDGET if aggressive else _NORMAL_INPUT_CHAR_BUDGET
    recent_start = session.compacted_message_count
    recent = session.messages[recent_start:]
    recent_transcript = "\n\n".join(
        f"{message.role.upper()}: {_clean_text(message.content, _MAX_MESSAGE_CHARS)}"
        for message in recent
    )

    blocks: list[str] = []
    if session.conversation_summary:
        blocks.append("COMPACTED EARLIER CONVERSATION\n" + session.conversation_summary)
    blocks.append("RECENT CONVERSATION\n" + (recent_transcript or "(none yet)"))

    if legacy_context.strip():
        blocks.append(
            "LEGACY BROWSER CONTEXT (may be abbreviated)\n"
            + _clean_text(legacy_context, _MAX_LEGACY_CONTEXT_CHARS)
        )

    compact_proposal = _compact_proposal(session.current_proposal)
    if compact_proposal is not None:
        blocks.append(
            "CURRENT DRAFT PROPOSAL (may be abbreviated; return a complete replacement)\n"
            + _bounded_json(compact_proposal, _MAX_PROPOSAL_CHARS if not aggressive else 12_000)
        )

    if mode == "criteria":
        blocks.append(
            "CURRENT SAVED CRITERIA TO IMPROVE (may be abbreviated)\n"
            + _bounded_json(
                _compact_criteria(saved_criteria),
                _MAX_CRITERIA_CHARS if not aggressive else 9_000,
            )
            + "\nFocus this turn on making the success criteria more concrete. "
            "Keep the goal meaning stable unless a change is necessary to remove ambiguity."
        )

    transcript = "\n\n".join(blocks)
    # The fixed setup instructions and goal are added later. Reserve space for
    # them here, then trim the variable transcript if necessary.
    reserved = min(_MAX_GOAL_CHARS, len(saved_goal)) + 11_000
    available = max(8_000, char_budget - reserved)
    if len(transcript) > available:
        head = transcript[: available // 3]
        tail = transcript[-(available - len(head)) :]
        transcript = (
            head.rstrip()
            + "\n\n… [older refinement context omitted to preserve model headroom] …\n\n"
            + tail.lstrip()
        )

    estimated_chars = len(transcript) + min(len(saved_goal), _MAX_GOAL_CHARS) + 11_000
    session.last_prompt_chars = estimated_chars
    session.last_estimated_input_tokens = math.ceil(estimated_chars / 4)
    session.last_context_mode = "compact_retry" if aggressive else "bounded"

    return RefinementPromptContext(
        transcript=transcript,
        prompt_char_budget=char_budget,
        estimated_input_tokens=session.last_estimated_input_tokens,
        compacted_message_count=session.compacted_message_count,
        mode=session.last_context_mode,
    )
