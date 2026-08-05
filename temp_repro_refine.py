import time
from fastapi.testclient import TestClient
from goal_agent.web import create_app

project = r"C:/Users/mattp/projects/rlm_1bit_student_poc"
app = create_app(project)
with TestClient(app) as client:
    r = client.post('/api/goals/test/proposal-jobs', json={
        'mode': 'goal',
        'feedback': 'Do we have any criteria to test some common benchmarks for LLMs?'
    })
    print('create', r.status_code)
    data = r.json()
    print('job', data.get('id'), data.get('status'), data.get('stage'), data.get('detail'))
    jid = data['id']
    for i in range(900):
        j = client.get(f'/api/proposal-jobs/{jid}').json()
        if i % 20 == 0:
            print('poll', i, j.get('status'), j.get('stage'), j.get('detail'))
        if j.get('status') in {'completed','failed','cancelled'}:
            print('final', j.get('status'))
            print('stage', j.get('stage'))
            print('detail', j.get('detail'))
            print('error', j.get('error'))
            break
        time.sleep(0.2)
