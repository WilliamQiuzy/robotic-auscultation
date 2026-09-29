"""V10 wire identity; original training and new inference sources stay distinct."""
import copy
import hashlib
import json
from pathlib import Path
import re
from sim.auscultation_pi05_v8 import action_encoding as codec
from sim.auscultation_pi05_v6.training_data import MODALITIES
from sim.auscultation_pi05_v6.task_observation import observation_binding

TRAINING_SCHEMA='wam_pi05_v10_recovery_training_v1'
CHECKPOINT_SCHEMA='wam_pi05_v10_recovery_checkpoint_binding_v1'
POLICY_SCHEMA='wam_pi05_v10_recovery_policy_binding_v1'
RUNTIME_SCHEMA='wam_pi05_v10_recovery_runtime_admission_v1'
RECIPE=TRAINING_SCHEMA
INFERENCE_MODULES=('actor.py','policy_contract.py','policy_client.py','policy_service.py')
CONNECTION_MODULES=('admission_bridge.py','runtime_contract.py','run_headroom_isaac.py','audit_actions.py','assess_trials.py')
DYNAMICS_FIELDS=('dynamics_recipe_sha256','dynamics_contract_sha256','execution_contract_sha256')
SAMPLING_POLICY='same fixed common Gaussian seed at every receding-horizon query'
CONFIG_FIELDS=('checkpoint_metadata_sha256','checkpoint_binding','admission_binding','selection_lock_binding','training_source_sha256','postlock_producer_source_binding','inference_source_sha256','connection_source_sha256','runtime_inventory_binding','data_hash','normalization_sha256','normalization_object_sha256','normalization_file_binding','dataset_binding','dataset_codec_binding',*DYNAMICS_FIELDS)


def need(ok,message):
    if not ok:raise ValueError(message)


def is_sha(x):return type(x) is str and re.fullmatch('[0-9a-f]{64}',x) is not None and x!='0'*64

def same(a,b):
    """JSON equality that does not equate bool/int or int/float aliases."""
    if type(a) is not type(b):return False
    if type(a) is dict:return set(a)==set(b) and all(same(a[k],b[k]) for k in a)
    if type(a) is list:return len(a)==len(b) and all(same(x,y) for x,y in zip(a,b))
    return a==b


def relative(name):
    p=Path(name);need(type(name) is str and not p.is_absolute() and '..' not in p.parts and str(p)==name and name not in ('','.'),'Safe repository-relative source required')
    return p


def file_binding(value):
    need(type(value) is dict and set(value)=={'path','sha256'} and type(value['path']) is str and Path(value['path']).is_absolute() and is_sha(value['sha256']),'Exact canonical path/SHA binding required')


def source_map(values,absolute=False):
    need(type(values) is dict and values,'Complete nonempty source map required')
    for k,v in values.items():
        if absolute:need(Path(k).is_absolute(),'Original absolute producer source required')
        else:relative(k)
        need(is_sha(v),'Invalid source digest')


def project():return Path(__file__).resolve().parents[2]


def producer_snapshot_key(path):
    need(type(path) is str and Path(path).is_absolute(),'Original absolute producer path required')
    return 'producer/'+hashlib.sha256(path.encode()).hexdigest()

def paths(names):return {'sim/auscultation_pi05_v10/'+n:Path(__file__).with_name(n) for n in names}
def inference_source_paths():return paths(INFERENCE_MODULES)
def connection_source_paths():return paths(CONNECTION_MODULES)
def hash_paths(values):return {k:hashlib.sha256(p.read_bytes()).hexdigest() for k,p in values.items()}
def inference_source_hashes():return hash_paths(inference_source_paths())
def connection_source_hashes():return hash_paths(connection_source_paths())


def verify_policy_binding(b,expected_metadata_sha256):
    need(type(b) is dict and is_sha(expected_metadata_sha256),'Explicit V10 binding required')
    expected=dict(schema=POLICY_SCHEMA,training_schema=TRAINING_SCHEMA,training_recipe=RECIPE,checkpoint_metadata_sha256=expected_metadata_sha256,
        action_semantics='absolute_joint_command_v1',model_action_semantics='future_command_minus_fixed_observation_measured_q',sampling_steps=10,sampling_seed=917,sampling_policy=SAMPLING_POLICY,
        modalities=MODALITIES,audio='absent',observation_contract=observation_binding(),action_encoding_binding=codec.action_encoding_binding(codec.B))
    for k,v in expected.items():need(same(b.get(k),v),'Incompatible V10 policy field: '+k)
    need(not any(k.startswith('pair_') for k in b),'Legacy pair identity forbidden')
    cp=b.get('checkpoint_binding',{});step=cp.get('local_step')
    need(cp.get('schema')==CHECKPOINT_SCHEMA and type(step) is int and step in range(500,4001,500),'Positive V10 selected checkpoint required')
    for k,v in dict(optimizer_step=step,schedule_cursor=4*step,cumulative_trainable_update_age=7300+step).items():need(same(cp.get(k),v),'V10 checkpoint clock differs')
    for alias,k in (('checkpoint','checkpoint_path'),('checkpoint_step','local_step'),('checkpoint_optimizer_step','optimizer_step'),('cumulative_trainable_update_age','cumulative_trainable_update_age')):need(same(b.get(alias),cp.get(k)),'Checkpoint identity alias differs: '+alias)
    for k in ('metadata_sha256','arrays_sha256','complete_sha256','data_hash','normalization_sha256','config_sha256','schedule_sha256'):need(is_sha(cp.get(k)),'Incomplete checkpoint hash: '+k)
    for k,c in (('checkpoint_metadata_sha256','metadata_sha256'),('data_hash','data_hash'),('normalization_sha256','normalization_sha256'),('training_source_sha256','source_sha256')):need(same(b.get(k),cp.get(c)),'Checkpoint/wire differs: '+k)
    for k in ('admission_binding','selection_lock_binding','normalization_file_binding','runtime_inventory_binding'):file_binding(b.get(k))
    need(b['normalization_file_binding']['sha256']==b['normalization_sha256'] and is_sha(b.get('normalization_object_sha256')),'Normalization digests differ')
    d=b.get('dataset_binding',{});need(d.get('schema')=='wam_pi05_v10_recovery_dataset_binding_v1' and d.get('data_hash')==b['data_hash'] and same(d.get('normalization_file_binding'),b['normalization_file_binding']),'V10 dataset identity differs')
    need(same(b.get('dataset_codec_binding'),dict(dataset_binding=d,normalization_sha256=b['normalization_sha256'],normalization_object_sha256=b['normalization_object_sha256'],action_encoding_binding=b['action_encoding_binding'])),'Dataset codec differs')
    for k in DYNAMICS_FIELDS:need(is_sha(b.get(k)),'Missing dynamics hash: '+k)
    for k in ('training_source_sha256','inference_source_sha256','connection_source_sha256'):source_map(b.get(k))
    source_map(b.get('postlock_producer_source_binding'),absolute=True)
    need(set(b['inference_source_sha256'])==set(inference_source_paths()) and set(b['connection_source_sha256'])==set(connection_source_paths()),'Exact V10 connection source sets required')


def verify_runtime_sources(b):
    verify_policy_binding(b,b.get('checkpoint_metadata_sha256'))
    need(b['inference_source_sha256']==inference_source_hashes() and b['connection_source_sha256']==connection_source_hashes(),'Actual V10 connection sources differ')
    for name,h in b['training_source_sha256'].items():
        p=project()/relative(name);need(not p.is_symlink() and p.is_file() and hashlib.sha256(p.read_bytes()).hexdigest()==h,'Frozen training source differs: '+name)
    for name,h in b['postlock_producer_source_binding'].items():
        p=Path(name);need(p==p.resolve() and p.is_file() and not p.is_symlink() and hashlib.sha256(p.read_bytes()).hexdigest()==h,'Original scientific producer changed')


def verify_config_binding(config,b):
    verify_policy_binding(b,config.get('expected_checkpoint_metadata_sha256'))
    for k in CONFIG_FIELDS:need(same(config.get('expected_'+k),b.get(k)) and b.get(k) is not None,'Explicit V10 config differs: '+k)


def verify_against_admission(b,a,admission_sha):
    need(a.get('schema')==RUNTIME_SCHEMA and a.get('candidate_admitted') is True and a.get('service_eligible') is True,'Positive reconstructed V10 admission required')
    cp=a['checkpoint_binding'];verify_policy_binding(b,cp['metadata_sha256'])
    need(same(b['checkpoint_binding'],cp) and same(b['admission_binding'],a['admission_binding']) and b['admission_binding']['sha256']==admission_sha,'Exact admitted checkpoint differs')
    for k in ('dataset_binding','selection_lock_binding','normalization_object_sha256','postlock_producer_source_binding','runtime_inventory_binding','training_source_sha256'):need(same(b.get(k),a.get(k)),'Policy differs from admission: '+k)


def make_binding(a,admission_sha,metadata):
    cp=a['checkpoint_binding'];d=a['dataset_binding'];c=metadata['config']
    need(c.get('training_schema')==TRAINING_SCHEMA and same(c.get('source_sha256'),cp['source_sha256']),'V10 admitted configuration required')
    b=dict(schema=POLICY_SCHEMA,training_schema=TRAINING_SCHEMA,training_recipe=RECIPE,checkpoint=cp['checkpoint_path'],checkpoint_step=cp['local_step'],checkpoint_optimizer_step=cp['optimizer_step'],cumulative_trainable_update_age=cp['cumulative_trainable_update_age'],checkpoint_binding=cp,checkpoint_metadata_sha256=cp['metadata_sha256'],data_hash=cp['data_hash'],normalization_sha256=cp['normalization_sha256'],normalization_file_binding=d['normalization_file_binding'],dataset_binding=d,
        admission_binding=a['admission_binding'],selection_lock_binding=a['selection_lock_binding'],normalization_object_sha256=a['normalization_object_sha256'],postlock_producer_source_binding=a['postlock_producer_source_binding'],runtime_inventory_binding=a['runtime_inventory_binding'],training_source_sha256=cp['source_sha256'],inference_source_sha256=inference_source_hashes(),connection_source_sha256=connection_source_hashes(),
        sampling_seed=917,sampling_steps=10,sampling_policy=SAMPLING_POLICY,modalities=MODALITIES,audio='absent',observation_contract=observation_binding(),action_semantics='absolute_joint_command_v1',model_action_semantics='future_command_minus_fixed_observation_measured_q',action_encoding_binding=codec.action_encoding_binding(codec.B),**{k:c[k] for k in DYNAMICS_FIELDS})
    b['dataset_codec_binding']=dict(dataset_binding=d,normalization_sha256=b['normalization_sha256'],normalization_object_sha256=b['normalization_object_sha256'],action_encoding_binding=b['action_encoding_binding'])
    verify_against_admission(b,a,admission_sha);return copy.deepcopy(b)
