from types import SimpleNamespace
import pytest
from sim.auscultation_pi05_v6 import run_acceptance_case as module


def args(tmp_path):
    return SimpleNamespace(case='R4_learned_development',checkpoint_sha256='a'*64,points=['R4'],scope='full',
        source_root='/home/robot/wam-sim-media/pi05-v6-results',mirror=str(tmp_path/'new'),
        service_directory='/home/william/wam-vla-v2/service-v6',client_config='/private/path',
        dynamics_contract='/declared/path',execute=False)


def test_prepared_case_never_connects_starts_isaac_or_claims_acceptance(tmp_path,monkeypatch):
    def unexpected(*a,**kw):raise AssertionError('No remote action in preparation')
    monkeypatch.setattr(module,'remote',unexpected);monkeypatch.setattr(module,'linux_python',unexpected)
    result=module.run(args(tmp_path))
    assert result['prepared_only'] and result['scope']=='full'
    assert not (tmp_path/'new').exists() and 'case_passed' not in result


@pytest.mark.parametrize('field,value',[('case','../escape'),('points',['R4','R4']),
                                      ('source_root','relative'),('checkpoint_sha256','latest')])
def test_rejects_unbound_or_ambiguous_execution_spec(tmp_path,field,value):
    value_args=args(tmp_path);setattr(value_args,field,value)
    with pytest.raises(ValueError):module.run(value_args)


def test_contact_component_cannot_be_declared_as_sequence(tmp_path):
    value=args(tmp_path);value.scope='contact';value.points=['R4','L4']
    with pytest.raises(ValueError,match='component'):module.run(value)
