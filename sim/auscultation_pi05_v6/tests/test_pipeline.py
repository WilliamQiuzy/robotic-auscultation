"""A completed process/offline evaluation must never become physical acceptance."""
import json
from pathlib import Path
from types import SimpleNamespace
import pytest
from sim.auscultation_pi05_v6 import pipeline


def fixture(monkeypatch,tmp_path):
    manifest=tmp_path/'manifest.json';manifest.write_text('{}')
    dataset=SimpleNamespace(indices={'test':[0]},data_hash='d'*64,
                            dynamics={'recipe_sha256':'e'*64})
    monkeypatch.setattr(pipeline,'load_dataset',lambda path:dataset)
    return manifest,tmp_path/'run'


def test_clean_early_stop_is_not_training_completion(monkeypatch,tmp_path):
    manifest,output=fixture(monkeypatch,tmp_path)
    monkeypatch.setattr(pipeline.subprocess,'run',lambda *a,**k:SimpleNamespace(returncode=0))
    with pytest.raises(ValueError,match='final checkpoint'):
        pipeline.run(manifest,output,steps=1000)
    report=json.loads((output/'pipeline.json').read_text())
    assert not report['training_complete']
    assert not report['closed_loop_validated'] and not report['policy_qualified']
    assert len(report['stages'])==1


def test_offline_completion_never_awards_isaac_acceptance(monkeypatch,tmp_path):
    manifest,output=fixture(monkeypatch,tmp_path)
    def simulate_process(command,**kwargs):
        if command[4].endswith('train_openpi'):
            model=Path(command[command.index('--output')+1]);final=model/'step-000000001000'
            final.mkdir(parents=True);(final/'metadata.json').write_text('{"step":1000}')
            (model/'best.json').write_text(json.dumps(dict(directory=final.name,
                metadata_sha256=pipeline._sha(final/'metadata.json'))))
        else:
            evaluation=Path(command[command.index('--output')+1]);evaluation.mkdir()
            selected=Path(command[command.index('--checkpoint')+1])
            report=dict(data_hash='d'*64,checkpoint=str(selected),checkpoint_step=1000,
                checkpoint_metadata_sha256=pipeline._sha(selected/'metadata.json'),
                splits={'validation':{},'test':{}},closed_loop_validated=False)
            if '--selection-only' in command:
                from sim.auscultation_pi05_v6.action_contract import PHASES
                report.update(selection_only=True,seed=917,sampling_steps=10,per_phase=20,
                    sampling_policy='fixed deployment seed per query',splits={'validation':dict(
                        records=[{'frame':1}],first_action_bound_pass_fraction={'baseline':1.},
                        baseline_by_phase={phase:{'first_action_joint_mae_rad':.01} for phase in PHASES})})
            (evaluation/'report.json').write_text(json.dumps(report))
        return SimpleNamespace(returncode=0)
    monkeypatch.setattr(pipeline.subprocess,'run',simulate_process)
    report=pipeline.run(manifest,output,steps=1000)
    assert report['training_complete'] and report['offline_evaluation_complete']
    assert not report['closed_loop_validated'] and not report['policy_qualified']


def test_failure_retains_stage_and_does_not_run_evaluation(monkeypatch,tmp_path):
    manifest,output=fixture(monkeypatch,tmp_path)
    monkeypatch.setattr(pipeline.subprocess,'run',lambda *a,**k:SimpleNamespace(returncode=17))
    with pytest.raises(RuntimeError,match='training failed'):
        pipeline.run(manifest,output,steps=1000)
    report=json.loads((output/'pipeline.json').read_text())
    assert len(report['stages'])==1 and report['stages'][0]['exit_code']==17
    assert not report['training_complete'] and not report['offline_evaluation_complete']
