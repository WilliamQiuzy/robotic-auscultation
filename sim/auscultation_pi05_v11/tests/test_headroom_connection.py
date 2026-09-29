"""AST physical equivalence; no model, simulator, heldout arrays or private config."""
import ast
from pathlib import Path
import pytest
HERE=Path(__file__).parents[1];OLD=HERE.parent/'auscultation_pi05_v8'


def tree(name):
    p=HERE/name;assert p.is_file(),'Missing V11 physical connection';return ast.parse(p.read_text())


def test_nine_only_runtime_modules_exist():
    for n in ('actor','policy_contract','policy_client','policy_service','admission_bridge','runtime_contract','run_headroom_isaac','audit_actions','assess_trials'):tree(n+'.py')


def test_physical_classes_are_actual_original_v7():
    t=tree('run_headroom_isaac.py');f=next(n for n in t.body if isinstance(n,ast.FunctionDef) and n.name=='physical_classes')
    old=next(n for n in ast.parse((OLD/'run_headroom_isaac.py').read_text()).body if isinstance(n,ast.FunctionDef) and n.name=='physical_classes')
    assert ast.dump(f)==ast.dump(old)


def test_no_new_training_imports_in_runtime_modules():
    for name in ('actor.py','policy_contract.py','policy_client.py','policy_service.py','admission_bridge.py','runtime_contract.py','run_headroom_isaac.py','audit_actions.py','assess_trials.py'):
        for node in ast.walk(tree(name)):
            if isinstance(node,ast.ImportFrom):
                assert node.module not in ('train','training_data','evaluation','parity','sim.auscultation_pi05_v11.train','sim.auscultation_pi05_v11.evaluation','sim.auscultation_pi05_v11.parity')


class IdentityOnly(ast.NodeTransformer):
    """Only these declared V11 identity substitutions may differ in core bodies."""
    def visit_Constant(self,node):
        changes={'wam_pi05_v11_bounded_headroom_replay_v1':'wam_v8_horizon_pair_headroom_replay_v1','Actual V11 recovery client type differs':'Actual V8 pair client type differs','pi05_v11':'pi05_v8'}
        if type(node.value) is str and node.value in changes:node.value=changes[node.value]
        return node
    def visit_keyword(self,node):
        self.generic_visit(node)
        if node.arg=='recovery_runtime_binding':node.arg='pair_runtime_binding'
        return node


def funcs(path):return {n.name:n for n in ast.parse(path.read_text()).body if isinstance(n,ast.FunctionDef)}


def equal_body(old,new):
    import copy
    assert ast.dump(old)==ast.dump(IdentityOnly().visit(copy.deepcopy(new)))


def test_entire_numeric_physical_and_acceptance_bodies_have_only_declared_identity_changes():
    old=funcs(OLD/'audit_actions.py');new=funcs(HERE/'audit_actions.py')
    for name in ('require','vector','close','digest','f32_digest','valid_hash','request_digest','verify_sensor','inherited_audit_source_paths','audit_original_gates','verify_runtime_evidence','verify_registration','_audit','audit_records','_audit_episode_replay'):
        equal_body(old[name],new[name])
    old=funcs(OLD/'run_headroom_isaac.py');new=funcs(HERE/'run_headroom_isaac.py')
    for name in ('physical_classes','validate_sequence_bundles','capture'):equal_body(old[name],new[name])
    # Complete native SDK execution/teardown tail, not selected statements.
    def tail(f):
        index=next(i for i,n in enumerate(f.body) if isinstance(n,ast.ImportFrom) and n.module=='isaacsim')
        return ast.Module(body=f.body[index:],type_ignores=[])
    equal_body(tail(old['main']),tail(new['main']))
    equal_body(funcs(OLD/'assess_trials.py')['inherited_check_documents'],funcs(HERE/'assess_trials.py')['inherited_check_documents'])


def pure_numeric_auditor():
    """Execute exact dependency-free audit functions, not mocked replacement code."""
    import types
    import numpy as np
    from sim.auscultation_pi05_v11 import action_encoding as codec
    from sim.auscultation_pi05_v7 import coverage_pilot_evaluation as E
    names=('vector','exact_f32','verify_experiment_requests');parsed=tree('audit_actions.py')
    body=[n for n in parsed.body if isinstance(n,ast.FunctionDef) and n.name in names]
    scope=dict(np=np,E=E,require=E.need,pair_codec=codec,C=types.SimpleNamespace(verify_policy_against_admission=lambda *args:None,B=types.SimpleNamespace(read_bound=lambda p,h:E.read(p))),Path=Path)
    exec(compile(ast.Module(body=body,type_ignores=[]),str(HERE/'audit_actions.py'),'exec'),scope)
    return types.SimpleNamespace(**scope)


@pytest.mark.parametrize('index',list(range(70)))
def test_every_raw_coordinate_is_checked_without_clipping(tmp_path,index):
    import json
    import numpy as np
    from sim.auscultation_pi05_v11 import action_encoding as codec
    from sim.auscultation_pi05_v11.tests.test_admission_bridge import synthetic_normalization
    from sim.auscultation_pi05_v7 import coverage_pilot_evaluation as E
    a=pure_numeric_auditor();norm=synthetic_normalization();stats=norm['residual_actions']
    (tmp_path/'metadata.json').write_text(json.dumps({'normalization':norm}))
    b=dict(normalization_sha256=E.digest(norm),checkpoint_metadata_sha256='a'*64,admission_binding={'path':'/synthetic','sha256':'b'*64},training_source_sha256={'sim/synthetic':'c'*64})
    admission={'checkpoint_binding':dict(checkpoint_path=str(tmp_path),metadata_sha256='unused-in-this-pure-numeric-fixture')}
    state=np.arange(41,dtype=np.float32)/100;z=np.linspace(-3,3,70,dtype=np.float32).reshape(10,7);raw=codec.decode_chunk(z,state[:7],'FREE','R4',stats)
    receipt=dict(b,request_id='synthetic',codec_target='R4',codec_phase='FREE',model_latent_residual=codec.denormalize_chunk(z,'FREE','R4',stats).tolist(),observation_anchor_q=state[:7].tolist(),model_normalized_residual=z.tolist(),raw_actions=raw.tolist())
    rows=[dict(phase='FREE',target='R4',policy_state=state.tolist(),inference=receipt,raw_actions=raw.tolist())];server=[dict(receipt,phase='FREE',target='R4')]
    assert a.verify_experiment_requests(rows,server,b,admission)==1
    rows[0]['raw_actions'][index//7][index%7]=float(np.nextafter(raw[index//7,index%7],np.float32(np.inf)))
    with pytest.raises(ValueError):a.verify_experiment_requests(rows,server,b,admission)


@pytest.mark.parametrize('mutation',['receipt_target','server_target','receipt_phase','server_phase','latent_client','latent_server'])
def test_latent_and_explicit_query_target_are_audited(tmp_path,mutation):
    import json,numpy as np
    from sim.auscultation_pi05_v11 import action_encoding as codec
    from sim.auscultation_pi05_v11.tests.test_admission_bridge import synthetic_normalization
    from sim.auscultation_pi05_v7 import coverage_pilot_evaluation as E
    a=pure_numeric_auditor();norm=synthetic_normalization();stats=norm['residual_actions'];(tmp_path/'metadata.json').write_text(json.dumps({'normalization':norm}))
    b=dict(normalization_sha256=E.digest(norm),checkpoint_metadata_sha256='a'*64,admission_binding={'path':'/synthetic','sha256':'b'*64},training_source_sha256={'sim/synthetic':'c'*64})
    admission={'checkpoint_binding':dict(checkpoint_path=str(tmp_path),metadata_sha256='unused-pure-fixture')}
    state=np.arange(41,dtype=np.float32)/100;z=np.zeros((10,7),np.float32);raw=codec.decode_chunk(z,state[:7],'FREE','R4',stats);latent=codec.denormalize_chunk(z,'FREE','R4',stats)
    receipt=dict(b,request_id='synthetic',codec_target='R4',codec_phase='FREE',model_latent_residual=latent.tolist(),observation_anchor_q=state[:7].tolist(),model_normalized_residual=z.tolist(),raw_actions=raw.tolist())
    import copy
    rows=[dict(phase='FREE',target='R4',policy_state=state.tolist(),inference=receipt,raw_actions=raw.tolist())];server=[dict(copy.deepcopy(receipt),phase='FREE',target='R4')]
    if mutation=='receipt_target':receipt['codec_target']='L4'
    elif mutation=='server_target':server[0]['target']='L4'
    elif mutation=='receipt_phase':receipt['codec_phase']='CONTACT'
    elif mutation=='server_phase':server[0]['phase']='CONTACT'
    elif mutation=='latent_client':receipt['model_latent_residual'][9][6]=float(np.nextafter(latent[9,6],np.inf))
    else:server[0]['model_latent_residual'][9][6]=float(np.nextafter(latent[9,6],np.inf))
    with pytest.raises(ValueError):a.verify_experiment_requests(rows,server,b,admission)


def test_all_aggregate_roles_still_require_ten_singles_and_one_ordered_sequence():
    old=funcs(OLD/'assess_trials.py')['assess'];new=funcs(HERE/'assess_trials.py')['assess']
    # The body includes identity-specific report/source names; the decisive role
    # requirement itself must remain byte-equivalent AST.
    def gate(f):
        return next(n for n in ast.walk(f) if isinstance(n,ast.Call) and any(isinstance(x,ast.Constant) and isinstance(x.value,str) and x.value=='Exactly ten full singles and the ordered continuous sequence required' for x in n.args))
    assert ast.dump(gate(old))==ast.dump(gate(new))
