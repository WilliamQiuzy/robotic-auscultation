"""Shutdown ordering with CPU fakes; never starts Kit or advances physics."""
import importlib
import json
from types import SimpleNamespace

import pytest


def module():
    assert importlib.util.find_spec('sim.ausc_policy.native_lifecycle') is not None
    return importlib.import_module('sim.ausc_policy.native_lifecycle')


def fixture(*,absent=False,fail=None):
    events=[];stage=object();physics=[]
    def call(label):
        events.append(label)
        if label==fail:raise RuntimeError(label+' failed')
    viewport=SimpleNamespace(updates_enabled=True)
    window=SimpleNamespace(viewport_api=viewport,destroy=lambda:call('viewport_destroy'))
    def close():
        call('app_close')
        assert 'viewport_destroy' in events or absent
    app=SimpleNamespace(update=lambda:call('app_update'),close=close,
        step=lambda:physics.append('forbidden'))
    context=SimpleNamespace(get_stage=lambda:stage)
    orchestrator=SimpleNamespace(set_capture_on_play=lambda value:call(('capture_on_play',value)),
        stop=lambda:call('rep_stop'),wait_until_complete=lambda:call('rep_wait'))
    return app,dict(context=context,replicator=SimpleNamespace(orchestrator=orchestrator),
        get_viewport_window=lambda:None if absent else window),events,physics,viewport


@pytest.mark.parametrize('absent',[False,True])
def test_quiescence_precedes_normal_close_and_preserves_absent_viewport(absent):
    m=module();app,dependencies,events,physics,viewport=fixture(absent=absent)
    result=m.close_shared_runtime(app,**dependencies)
    expected=[('capture_on_play',False),'rep_stop','rep_wait']
    if not absent:expected+=['viewport_destroy']
    assert events==expected+['app_update']*m.CLEANUP_UPDATES+['app_close']
    assert not physics and (absent or viewport.updates_enabled is False)
    assert result['app_close_returned'] is True and result['fast_shutdown'] is False
    assert result['viewport_present'] is (not absent)
    assert m.shared_app_settings()==dict(headless=True,fast_shutdown=False,disable_viewport_updates=True,limit_cpu_threads=6)
    assert result['configured_limit_cpu_threads']==6


@pytest.mark.parametrize('failure',['rep_stop','rep_wait','viewport_destroy','app_update','app_close'])
def test_cleanup_error_still_attempts_close_but_cannot_return_success(failure):
    m=module();app,dependencies,events,physics,_=fixture(fail=failure)
    with pytest.raises(RuntimeError,match=failure):m.close_shared_runtime(app,**dependencies)
    assert events.count('app_close')==1 and not physics


def test_stage_must_remain_present_through_quiescence():
    m=module();app,dependencies,events,_,_=fixture()
    dependencies['context'].get_stage=lambda:None
    with pytest.raises(RuntimeError,match='stage'):m.close_shared_runtime(app,**dependencies)
    assert events[-1]=='app_close'


@pytest.mark.parametrize('failure',[None,'rep_wait','app_close'])
def test_progress_is_bounded_fsynced_and_disambiguates_enter_from_return(tmp_path,monkeypatch,failure):
    m=module();app,dependencies,_,_,_=fixture(fail=failure);syncs=[]
    monkeypatch.setattr(m.os,'fsync',lambda fd:syncs.append(fd))
    path=tmp_path/'runtime_shutdown_progress.jsonl'
    if failure:
        with pytest.raises(RuntimeError):m.close_shared_runtime(app,progress_path=path,**dependencies)
    else:m.close_shared_runtime(app,progress_path=path,**dependencies)
    rows=[json.loads(line) for line in path.read_text().splitlines()];events=[row['event'] for row in rows]
    assert len(rows)==len(syncs)<=32 and [row['index'] for row in rows]==list(range(len(rows)))
    assert events.index('rep_stop_enter')<events.index('rep_stop_return')<events.index('rep_wait_enter')
    assert 'app_close_enter' in events
    if failure=='rep_wait':assert 'rep_wait_return' not in events
    if failure is None:
        assert events[-1]=='app_close_return'
        assert events.index('viewport_destroy_return')<events.index('app_update_0_enter')
    if failure=='app_close':assert 'app_close_return' not in events
