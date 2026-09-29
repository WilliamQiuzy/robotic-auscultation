from types import SimpleNamespace as S
import numpy as np
import pytest
from sim.auscultation_pi05_v5 import run_isaac
from sim.auscultation_pi05_v5.learned_controller import LearnedController, SafetyController


REQUIRED=['--source-project','/source','--output','/output','--annotation','/annotation',
          '--client-config','/client']


def test_cli_preserves_single_point_and_ordered_sequence():
    single=run_isaac.parse_args(REQUIRED+['--point','R4'])
    assert single.points==['R4']
    sequence=run_isaac.parse_args(REQUIRED+['--points','R4','L2','R1','--hall-condition','/condition.json'])
    assert sequence.points==['R4','L2','R1']
    assert str(sequence.hall_condition)=='/condition.json'


@pytest.mark.parametrize('options',[
    ['--point','R4','--points','L2'],['--points','R4','R4'],
    ['--points','R4','L2','--task-scope','contact'],
])
def test_cli_rejects_ambiguous_or_classical_contact_sequences(options):
    with pytest.raises(SystemExit):run_isaac.parse_args(REQUIRED+options)


def bundle(name,base=None,geometry=b'geometry',binding='binding'):
    return S(target=S(name=name),plan=dict(world_from_left_base=np.eye(4).tolist() if base is None else base,
        revalidation=dict(collision_regeneration_sha256=binding)),
        inputs={'collision_metadata':('meta',geometry),'collision_arrays':('arrays',b'arrays')})


def test_sequence_requires_matching_bases_geometry_binding_and_order():
    good=[bundle('R4'),bundle('L2')]
    run_isaac.validate_sequence_bundles(good,['R4','L2'],'binding')
    shifted=np.eye(4);shifted[0,3]=.001
    for invalid in ([bundle('R4'),bundle('L2',base=shifted.tolist())],
                    [bundle('R4'),bundle('L2',geometry=b'changed')],
                    [bundle('R4'),bundle('L2',binding='changed')],list(reversed(good))):
        with pytest.raises(ValueError):run_isaac.validate_sequence_bundles(invalid,['R4','L2'],'binding')


def test_run_rejects_multiple_contact_components_before_parent_motion():
    actor=LearnedController.__new__(LearnedController);actor.task_scope='contact'
    with pytest.raises(ValueError,match='contact'):
        actor.run([(S(name='R4'),{}),(S(name='L2'),{})])


def test_full_interpoint_transfer_uses_only_learned_queries(tmp_path,monkeypatch):
    from sim.auscultation_expert_layout_v1.phase_manager import Phase
    actor=LearnedController.__new__(LearnedController)
    actor.task_scope='full';actor.sequence_index=1;actor.q=np.zeros(7);actor.joint_speed_limit=.25
    actor.target=S(name='L2');actor.pre_gap=.03;actor.manager=S(state=S(phase=Phase.APPROACH))
    state=dict(time_s=2.,q=np.zeros(7));actor.env=S(state=lambda:state,rows=[{}],out=tmp_path)
    actor.sensor=S(rows=[dict(target='L2',time_s=2.,state=np.r_[np.zeros(14),.2,0,0,np.zeros(15)])])
    actor._canonical=lambda row:(np.zeros((3,7)),1)
    def forbidden(*args,**kwargs):raise AssertionError('Classical transfer/hold entered learned interpoint path')
    monkeypatch.setattr(SafetyController,'_move_to_prepose',forbidden)
    actor._hold=forbidden;actor._joint_segment=forbidden;actor._continuous_path=forbidden
    calls=[]
    def infer(phase,*,active):
        calls.append((phase,active));actor.sensor.rows[-1]['state'][14]=.03
    actor._infer_step=infer
    actor._tracked_metrics=lambda state:(dict(gap_m=.03 if calls else .2,tangent_m=0.,angle_deg=0.),None,None)
    actor._move_to_prepose({})
    assert calls==[('FREE',False)]
