import copy
import pytest
from sim.auscultation_pi05_v10 import train as T


@pytest.mark.parametrize('step',[-1,True,4001,1.0])
def test_invalid_clock(step):
    with pytest.raises(ValueError):T.clock(step)


def test_repair_clock_is_not_parent_optimizer_age():
    assert T.clock(500)==dict(local_step=500,optimizer_step=500,schedule_cursor=2000,cumulative_trainable_update_age=7800)


def test_complete_requires_current_terminal_receipt(tmp_path):
    run=tmp_path.resolve();seg=run/'training_segments'/'current.json';seg.parent.mkdir()
    r=dict(schema=T.VERSION,training_run_path=str(run),segment=str(seg),training_complete=True,interrupted=False,error=None,**T.clock(4000))
    raw=T.json_bytes(r);seg.write_bytes(raw);(run/'training_run.json').write_bytes(raw)
    assert T.verify_terminal(run)==r
    seg.with_suffix('.cancelled.json').symlink_to(run/'absent')
    with pytest.raises(ValueError):T.verify_terminal(run)


def test_old_terminal_segment_cannot_replace_current(tmp_path):
    run=tmp_path.resolve();folder=run/'training_segments';folder.mkdir()
    seg=folder/'current.json';old=folder/'old.json'
    r=dict(schema=T.VERSION,training_run_path=str(run),segment=str(seg),training_complete=True,interrupted=False,error=None,**T.clock(4000))
    raw=T.json_bytes(r);old.write_bytes(raw);seg.write_bytes(raw+b' ');(run/'training_run.json').write_bytes(raw)
    with pytest.raises(ValueError):T.verify_terminal(run)


@pytest.mark.parametrize('when',['before','save','restore'])
def test_cancel_during_any_final_boundary_rejects(monkeypatch,when):
    stop=[when=='before'];r=dict(training_complete=True,error=None,**T.clock(4000));saved=[];sidecars=[]
    def save():
        saved.append(copy.deepcopy(r))
        if when=='save':stop[0]=True
    def restore(*args):
        if when=='restore':stop[0]=True
    monkeypatch.setattr(T.signal,'signal',restore)
    with pytest.raises(InterruptedError):T.finish_training(r,stop,{2:None,15:None},save,sidecars.append)
    assert sidecars and sidecars[0]['interrupted']
    assert not saved[-1]['training_complete']
