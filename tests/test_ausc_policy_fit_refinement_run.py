import json
import pytest
from sim.ausc_policy.fit_refinement_run import run_fit_diagnostic

TARGET=dict(name='R4',position_world_m=[0.,0.,0.],normal_world=[0.,0.,1.])


def test_partial_native_failure_keeps_full_attempt_ledger(tmp_path):
    class Oracle:
        def first_contact(self,*args):raise RuntimeError('native query failed')
    with pytest.raises(RuntimeError,match='native query failed'):
        run_fit_diagnostic(Oracle(),[TARGET],tmp_path,dict(top_k=8,max_calls=10000))
    rows=[json.loads(line) for line in (tmp_path/'R4_fit_attempts.jsonl').read_text().splitlines()]
    assert len(rows)==1 and 'native query failed' in rows[0]['exception']['message']
    assert not (tmp_path/'COMPLETE.json').exists()


def test_rejects_target_filename_escape_before_open(tmp_path):
    with pytest.raises(ValueError,match='target'):
        run_fit_diagnostic(object(),[dict(TARGET,name='../R4')],tmp_path,dict(top_k=8,max_calls=10000))
    assert not list(tmp_path.iterdir())
