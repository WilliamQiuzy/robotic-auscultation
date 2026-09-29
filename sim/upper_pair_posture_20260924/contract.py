"""Pinned comparison contract: only upper-point joint paths and audited roll vary."""
import argparse
import hashlib
import json
import math
from pathlib import Path

ROOT = Path(__file__).resolve().parent
PROJECT = ROOT.parents[1]
BASELINE = PROJECT/'sim/curobo_expert_v1/results/physics_revised_all10_v1'
PLANS = PROJECT/'sim/auscultation_expert_layout_v1/results/expert_v2/plans'
RUNTIME = PROJECT/'sim/auscultation_expert_layout_v1'
CAMERA = PROJECT/'sim/camera_framing_20260923/recommended_camera.json'
TARGETS = tuple(side+str(level) for level in range(1,6) for side in ('R','L'))
FIXED_ROW_KEYS = ('name','position_world','normal_world','posture_limits',
    'loaded_compression_m','free_approach_start_gap_m','free_approach_end_gap_m',
    'geometric_clearance_m','loaded_noncontact_clearance_m','joint_edge_sample_step_rad',
    'command_normal_world','expected_contact_gap_m','expected_contact_center_world')
INPUT_LABELS = ('audit','targets','runtime','collision_metadata','collision_arrays','kinematics_urdf')
COLLISION_BINDING = 'sim/auscultation_expert_layout_v1/results/collision_regeneration_verification.json'
REVALIDATION_SOURCES = frozenset((
    'sim/upper_pair_posture_20260924/revalidate.py', 'sim/upper_pair_posture_20260924/contract.py',
    'sim/curobo_expert_v1/fast_collision.py',
    *('sim/auscultation_expert_layout_v1/'+name for name in (
        'collision.py','kinematics.py','tool_frames.py','posture.py','replan_layout.py',
        'controller.py','revalidate_plans.py','targets.py','scene.py'))))


def sha(path):
    h=hashlib.sha256()
    with Path(path).open('rb') as stream:
        for block in iter(lambda:stream.read(1048576),b''):h.update(block)
    return h.hexdigest()


def read(path):
    return json.loads(Path(path).read_text())


def require(condition,message):
    if not condition:raise ValueError(message)


def validate_candidate_plan(candidate,reference,audit):
    """Semantic invariants only; actual FK/meshes require independent revalidation."""
    require(len(candidate.get('points',[]))==1,'one upper point required')
    row,old=candidate['points'][0],reference['points'][0]
    require(row.get('name') in ('R1','L1'),'only upper points may change')
    for key in ('world_from_left_base','targets_file','audit_file','provenance'):
        require(candidate.get(key)==reference.get(key),'fixed plan field changed: '+key)
    for key in FIXED_ROW_KEYS:
        require(key in row and row[key]==old[key],'fixed point field changed: '+key)
    for key in ('training_accepted','contact_success'):
        require(row.get(key) is not True,'geometry cannot claim '+key)
    require(row.get('success') is True,'candidate geometry failed')
    index=row.get('selected_audit_candidate_index')
    require(type(index) is int and 0<=index<len(audit.get('candidates',[])),'invalid audit candidate index')
    fit=audit['candidates'][index]
    require(row.get('selected_fit')==fit,'selected fit differs from hashed audit')
    require(fit['normal_world']==old['selected_fit']['normal_world'],'selected contact normal changed')
    require(fit['gap_m']==old['selected_fit']['gap_m'],'selected gap changed')
    require(audit['name']==row['name'] and audit['compression_m']==old['loaded_compression_m'],
            'audit identity or compression changed')
    require(row.get('command_rotation_world')==fit['wrist_rotation_world'],'rotation differs from selected audit')


def candidate_motion_changed(candidate,reference):
    """Ignore reporting/receipt changes and sub-microradian serialization noise."""
    def differs(a,b):
        if isinstance(a,list) and isinstance(b,list):
            return len(a)!=len(b) or any(differs(x,y) for x,y in zip(a,b))
        if isinstance(a,(int,float)) and isinstance(b,(int,float)):
            return not math.isclose(a,b,abs_tol=1e-7,rel_tol=0.)
        return a!=b
    new,old=candidate['points'][0],reference['points'][0]
    return any(differs(new.get(key),old.get(key)) for key in (
        'command_rotation_world','transfer_q','approach_q','loaded_q',
        'contact_corridor_posture_q','lower_staging_q'))


def validate_geometry_receipt(evidence,name,reference,contract,before_digest,contract_digest):
    """Consume the independent receipt; booleans alone never establish freshness."""
    require(evidence.get('accepted') is True and evidence.get('independent_upper_pair_check') is True,
            'new candidate requires independent revalidation: '+name)
    require(evidence.get('schema')=='upper_pair_independent_geometry_v1' and evidence.get('point')==name,
            'wrong independent receipt schema or point: '+name)
    require(evidence.get('input_plan_sha256')==before_digest,'receipt input plan changed: '+name)
    require(evidence.get('collision_regeneration_sha256')==contract['fixed_files'][COLLISION_BINDING],
            'receipt collision binding differs from pinned baseline: '+name)
    receipt=evidence.get('collider_receipt',{})
    require(receipt.get('implementation')=='sim.curobo_expert_v1.fast_collision.FastCollisionChecker'
            and receipt.get('urdf_planning_margin_deg')==3.,'wrong full-mesh checker or planning margin: '+name)
    require(all(type(receipt.get(key)) is int and receipt[key]>0 for key in
                ('collider_count','pair_count','dynamic_collider_count')),'missing collider inventory: '+name)
    require(receipt.get('contract_sha256')==contract_digest,'receipt contract changed: '+name)
    expected={'plan':before_digest,**{label:reference['provenance'][label+'_sha256'] for label in INPUT_LABELS}}
    require(receipt.get('inputs_sha256')==expected,'receipt input bindings differ: '+name)
    sources=receipt.get('sources_sha256')
    require(isinstance(sources,dict) and set(sources)==REVALIDATION_SOURCES,
            'independent checker source set differs: '+name)
    for relative,digest in sources.items():
        path=(PROJECT/relative).resolve()
        require(PROJECT in path.parents and sha(path)==digest,'independent checker source changed: '+relative)


def create_contract(destination):
    destination=Path(destination)
    require(not destination.exists(),'contract destination must be new')
    snapshot=read(BASELINE/'input_snapshot/manifest.json')
    outcome=read(BASELINE/'outcome.json')
    require(outcome.get('success') is True and outcome.get('candidate_policy')=='revised_shortcut',
            'baseline must be completed latest B')
    files={}; baseline_plans={}
    def pin(path):
        path=Path(path).resolve()
        relative=str(path.relative_to(PROJECT))
        digest=sha(path)
        require(relative not in files or files[relative]==digest,'source changed during pinning')
        files[relative]=digest
        return relative,digest
    for name in TARGETS:
        path=PLANS/f'planned_paths_{name}.json'
        relative,digest=pin(path)
        require(snapshot['files'][name+'/plan']['sha256']==digest,'canonical plan differs from baseline: '+name)
        baseline_plans[name]={'path':relative,'sha256':digest}
        plan=read(path)
        for label in ('audit','targets','runtime','collision_metadata','collision_arrays','kinematics_urdf'):
            filename=plan['provenance'][label+'_file']
            require(Path(filename).name==filename,'unsafe canonical input')
            parent=RUNTIME if label in ('runtime','kinematics_urdf') else PLANS
            _,digest=pin(parent/filename)
            require(digest==plan['provenance'][label+'_sha256']==snapshot['files'][name+'/'+label]['sha256'],
                    'canonical input differs from baseline: '+name+'/'+label)
    for path in (BASELINE/'input_snapshot/manifest.json',BASELINE/'outcome.json',CAMERA,
                 RUNTIME/'results/collision_regeneration_verification.json',RUNTIME/'layout_inputs.json'):
        pin(path)
    for name in ('posture_limits.json','solver_posture_limits.json','reference_posture_limits.json'):
        if (PLANS/name).is_file():pin(PLANS/name)
    result={'schema':'upper_pair_fixed_comparison_v1','baseline_episode':str(BASELINE.relative_to(PROJECT)),
        'canonical_plans':str(PLANS.relative_to(PROJECT)),'runtime_directory':str(RUNTIME.relative_to(PROJECT)),
        'camera':str(CAMERA.relative_to(PROJECT)),'fixed_files':files,'baseline_plans':baseline_plans,
        'target_order':list(TARGETS),'changed_points':['R1','L1'],
        'baseline_phase_config':outcome['phase_config'],'speed_scale':outcome['speed_scale'],
        'controller_policy':'revised_shortcut','scope':'Same scene/contact constraints; only upper joint posture and audited axial roll may change. No hardware or clinical acceptance.'}
    verify_contract(result)
    destination.parent.mkdir(parents=True,exist_ok=True)
    destination.write_text(json.dumps(result,indent=2,allow_nan=False)+'\n')
    return result


def verify_contract(contract):
    require(contract.get('schema')=='upper_pair_fixed_comparison_v1','unexpected contract')
    require(contract.get('target_order')==list(TARGETS) and contract.get('changed_points')==['R1','L1'],
            'comparison scope changed')
    for relative,digest in contract['fixed_files'].items():
        path=(PROJECT/relative).resolve()
        require(PROJECT in path.parents and sha(path)==digest,'pinned input changed: '+relative)


def validate_materialized_set(directory,contract,*,require_revalidation=True,contract_path=ROOT/'contract.json'):
    directory=Path(directory).resolve()
    verify_contract(contract)
    require(directory!=PLANS.resolve() and PLANS.resolve() not in directory.parents,
            'candidate plans must be outside sealed baseline')
    if require_revalidation:
        require(read(contract_path)==contract,'contract bytes do not describe supplied contract')
        contract_digest=sha(contract_path)
    results={}
    for name in TARGETS:
        path=directory/f'planned_paths_{name}.json'
        reference_path=PROJECT/contract['baseline_plans'][name]['path']
        reference=read(reference_path)
        plan=read(path)
        for label in ('audit','targets','collision_metadata','collision_arrays'):
            filename=reference['provenance'][label+'_file']
            require(sha(directory/filename)==reference['provenance'][label+'_sha256'],
                    'candidate fixed input changed: '+name+'/'+label)
        if name not in ('R1','L1'):
            require(sha(path)==contract['baseline_plans'][name]['sha256'],'unchanged point plan altered: '+name)
        else:
            audit=read(directory/reference['audit_file'])
            validate_candidate_plan(plan,reference,audit)
            if require_revalidation:
                evidence=plan.get('revalidation',{})
                before=path.with_suffix('.before_revalidation.json')
                require(before.is_file() and sha(before)==evidence.get('input_plan_sha256'),
                        'missing candidate prevalidation bytes: '+name)
                without=dict(plan);without.pop('revalidation',None)
                require(read(before)==without,'candidate changed after revalidation: '+name)
                validate_geometry_receipt(evidence,name,reference,contract,sha(before),contract_digest)
        results[name]=sha(path)
    verify_contract(contract)
    return results


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--create',type=Path)
    parser.add_argument('--contract',type=Path,default=ROOT/'contract.json')
    parser.add_argument('--input-dir',type=Path)
    args=parser.parse_args()
    if args.create:
        result=create_contract(args.create)
        print(json.dumps({'fixed_files':len(result['fixed_files']),'baseline_points':len(result['baseline_plans'])}))
    elif args.input_dir:
        print(json.dumps(validate_materialized_set(args.input_dir,read(args.contract),contract_path=args.contract),indent=2))
    else:
        verify_contract(read(args.contract));print('CONTRACT_VERIFIED')


if __name__=='__main__':main()
