"""Independent, predeclared fifty-trajectory experiment; frozen V7 is ancestry.

No model, simulator, test prediction or dataset fabrication occurs here.
"""
import hashlib
import json
from pathlib import Path
import re

VERSION='wam_pi05_v7_expanded_experiment_v1'
EXPERIMENT_RECIPE='pi05_v7_start_family_50_v1'
PROTOCOL_SHA256='12449ae2177b1da66ac1556a5f0bae6600cfedb61dac8578b7a8d55c87948c04'
PROTOCOL_HELPER_SHA256='48cf82a53e83c4f51be194a67ebc08d63d1b981511ce7c97d48883fc95352b4a'
PHASES=('FREE','APPROACH','CONTACT','HOLD','RETRACT')
TARGETS=tuple(side+str(i) for i in range(1,6) for side in 'RL')
FAMILIES={'legacy_nominal_v6_diagnostic':'train','train_yaw_positive_v1':'train',
          'train_yaw_negative_v1':'train','validation_fold_v1':'validation','test_fold_v1':'test'}
SOURCE_NAMES=('expanded_contract.py','expanded_training_data.py','expanded_train_openpi.py',
              'expanded_actor.py','expanded_policy_service.py','expanded_policy_client.py',
              'expanded_evaluation.py','expanded_selection.py','expanded_final_test.py')

def json_bytes(value):return (json.dumps(value,sort_keys=True,separators=(',',':'),allow_nan=False)+'\n').encode()
def sha256_json(value):return hashlib.sha256(json_bytes(value)).hexdigest()
def file_sha(path):
    digest=hashlib.sha256()
    with Path(path).open('rb') as stream:
        for block in iter(lambda:stream.read(8*1024*1024),b''):digest.update(block)
    return digest.hexdigest()
def is_sha(value):return isinstance(value,str) and re.fullmatch('[0-9a-f]{64}',value) is not None and value!='0'*64

def training_plan():
    return dict(schema='wam_pi05_v7_expanded_training_plan_v1',initialization='same official OpenPI base; fresh optimizer',
        resume_policy='only latest complete same-run checkpoint with exact data/config/source/normalization; restore optimizer and both RNG states; no locked/test-started run',
        seed=61,batch_size=4,learning_rate=5e-5,total_steps=12000,save_every=1000,evaluate_every=1000,
        candidate_steps=list(range(1000,12001,1000)),diagnostic_splits=['train','validation'],
        normalization_fit_split='train',validation=dict(episodes=10,phases=list(PHASES),
            quota_per_episode_phase=20,selection='min(20, available) unique evenly spaced including first and last',empty_bin='reject',
            aggregation='equal weight for all 50 episode-phase buckets',
            ranking=['all validation and known-start sentinel raw chunks within URDF bounds',
                     'macro first-action joint MAE','macro full-chunk joint MAE','earlier step']),
        known_start_sentinels=dict(split='train',episodes=30,selection='first valid FREE window in each episode',
            purpose='raw full-chunk URDF feasibility veto only; no train-error ranking'),
        final_test=dict(after='immutable selected-checkpoint lock',prediction_runs=1,
            sampling='each of 10 test episodes x 5 phases: min(20, available) unique evenly spaced including first and last; empty bin rejects',
            scope='at most 1000 offline sampled windows; neither all-window nor closed-loop acceptance',
            retraining_after_inspection='test exposed; require new independent test cohort'),
        sampling_steps=10,sampling_seed=917,closed_loop_acceptance=False)

def experiment_source_paths():
    return {'v7/'+name:Path(__file__).with_name(name) for name in SOURCE_NAMES}

def experiment_source_hashes():return {name:file_sha(path) for name,path in experiment_source_paths().items()}

def verify_source_map(value,*,local=False):
    if not isinstance(value,dict) or set(value)!=set(experiment_source_paths()) or not all(is_sha(v) for v in value.values()):
        raise ValueError('Incomplete expanded experiment source binding')
    if local and value!=experiment_source_hashes():raise ValueError('Expanded experiment source bytes changed')

def experiment_binding(manifest_sha256,preparation_sha256,experiment_sha256):
    if not all(is_sha(v) for v in (manifest_sha256,preparation_sha256,experiment_sha256)):
        raise ValueError('Concrete manifest and preparation hashes required')
    plan=training_plan()
    return dict(schema=VERSION,experiment_recipe=EXPERIMENT_RECIPE,protocol_sha256=PROTOCOL_SHA256,
        manifest_sha256=manifest_sha256,preparation_sha256=preparation_sha256,experiment_sha256=experiment_sha256,
        training_plan=plan,training_plan_sha256=sha256_json(plan),
        cohort=dict(episodes=50,splits={'train':30,'validation':10,'test':10},heldout_axes=['motion_family']),
        source_scope='17 inherited training and 25 inherited service sources plus separate expanded experiment sources')

def verify_experiment_binding(binding):
    if not isinstance(binding,dict):raise ValueError('Missing expanded experiment binding')
    expected=experiment_binding(binding.get('manifest_sha256'),binding.get('preparation_sha256'),binding.get('experiment_sha256'))
    if binding!=expected:raise ValueError('Expanded experiment contract differs from predeclared plan')

def read_json(path):
    def unique(pairs):
        result={}
        for key,value in pairs:
            if key in result:raise ValueError('Duplicate JSON key: '+key)
            result[key]=value
        return result
    return json.loads(Path(path).read_text(),object_pairs_hook=unique,
        parse_constant=lambda _:(_ for _ in ()).throw(ValueError('Nonfinite JSON')))

def bound_file(root,binding):
    root=Path(root).resolve()
    if not isinstance(binding,dict) or set(binding)!={'path','sha256'}:raise ValueError('Exact path/hash evidence binding required')
    name=binding['path']
    if not isinstance(name,str) or not name or Path(name).is_absolute() or '..' in Path(name).parts:
        raise ValueError('Contained relative evidence path required')
    path=root/name
    if (not is_sha(binding['sha256']) or not path.is_file() or any(p.is_symlink() for p in [path,*path.parents] if p!=root and root in p.parents)
            or root not in path.resolve().parents or file_sha(path)!=binding['sha256']):
        raise ValueError('Evidence bytes/path differ: '+name)
    return path
