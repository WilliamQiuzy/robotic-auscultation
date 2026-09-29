"""Split-scoped qualification of source-rooted episodes with exact saved B stats.

The composite describes all eighteen identities. Only loaded_scopes are actually
array-qualified; qualification_scope_binding records that distinction. No old
full-dataset loader, normalization fitting, model or simulator is called here.
"""
from collections import OrderedDict
from dataclasses import dataclass, field
import copy
import hashlib
import json
from pathlib import Path
import numpy as np
from sim.auscultation_pi05_v8 import action_encoding as codec

SCHEMA='wam_pi05_v10_recovery_composite_manifest_v1'
BINDING_SCHEMA='wam_pi05_v10_recovery_dataset_binding_v1'
MODE='V10_recovery_data_B7300_v1'
DESIGN_SHA256='d4ab529ce6982433c447531b78b9d1f28de2a5726e22d001d5e704bc211f1f04'
ORIGINAL_MANIFEST_SHA256='ce5da81634870e79cc8442d4ec166387966cc1e7c125942a16ebe98d73dcc9b0'
NORMALIZATION_SHA256='0d374775d5b85b9b22a833b1b20afc69540b8b7c2f7ddd80e63de16d089e6794'
NORMALIZATION_OBJECT_SHA256='313b47e59585a4d248102ca2babe1cffd19c7375e224702267d34ebdee756170'
SPLITS=('train','validation','test')
PHASES=('FREE','APPROACH','CONTACT','HOLD','RETRACT')


def need(value,message):
    if not value:raise ValueError(message)


def json_bytes(value):return (json.dumps(value,sort_keys=True,separators=(',',':'),allow_nan=False)+'\n').encode()
def digest(value):return hashlib.sha256(json_bytes(value)).hexdigest()
def object_digest(value):return hashlib.sha256(json_bytes(value)[:-1]).hexdigest()

def file_sha(path):
    h=hashlib.sha256()
    with Path(path).open('rb') as stream:
        for data in iter(lambda:stream.read(8*1024*1024),b''):h.update(data)
    return h.hexdigest()


def _regular(path):
    p=Path(path)
    need(p.is_absolute() and p==p.resolve() and p.is_file() and not p.is_symlink(),'Canonical regular file required: '+str(p))
    return p


def _pin(path,expected,artifacts):
    p=_regular(path);need(type(expected) is str and len(expected)==64 and file_sha(p)==expected,'Bound file SHA mismatch: '+str(p))
    need(str(p) not in artifacts or artifacts[str(p)]==expected,'Conflicting artifact identity')
    artifacts[str(p)]=expected;return p


def _read_pinned(path,expected,artifacts):
    p=_pin(path,expected,artifacts)
    def unique(items):
        d={}
        for k,v in items:need(k not in d,'Duplicate JSON field');d[k]=v
        return d
    result=json.loads(p.read_text(),object_pairs_hook=unique,parse_constant=lambda v:(_ for _ in ()).throw(ValueError('Nonfinite JSON')))
    _pin(p,expected,artifacts);return result


def _relative(root,name,sha,artifacts):
    need(type(name) is str and name and not Path(name).is_absolute() and '..' not in Path(name).parts,'Relative source-root artifact required')
    p=Path(root)/name;need(p.resolve().is_relative_to(Path(root)),'Artifact escaped original source root')
    return _pin(p,sha,artifacts)


def _legacy():
    from sim.auscultation_pi05_v6 import train_openpi
    return train_openpi


def core_sources():
    root=Path(__file__).resolve().parents[2]
    result={}
    for name,expected in EXPECTED_SOURCES.items():
        p=root/name;need(file_sha(p)==expected,'Frozen source changed: '+name);result[name]=expected
    for name in ('__init__.py','training_data.py'):
        result['sim/auscultation_pi05_v10/'+name]=file_sha(Path(__file__).with_name(name))
    return result


def _recheck(artifacts):
    for p,h in artifacts.items():_pin(p,h,{})


def _normalization(path,expected,artifacts):
    need(expected==NORMALIZATION_SHA256,'Exact B7300 saved normalization required')
    n=_read_pinned(path,expected,artifacts)
    need(json_bytes(n)==Path(path).read_bytes() and object_digest(n)==NORMALIZATION_OBJECT_SHA256,'B normalization serialization/object changed')
    codec.validate_stats(n['residual_actions'],mode=codec.B)
    need(n['mode']==codec.B and n['action_encoding_binding']==codec.action_encoding_binding(codec.B),'B normalization mode/source differs')
    return n


def _source_metadata(path,sha,artifacts):
    old=_legacy();m=_read_pinned(path,sha,artifacts);root=Path(path).parent
    need(m.get('schema')=='wam_vla_training_manifest_v1' and m.get('sealed') is True and type(m.get('horizon')) is int and m['horizon']==10 and m.get('action_period_s')==.05,'Sealed original source manifest required')
    old.validate_modalities(m.get('modalities'))
    from sim.auscultation_pi05_v6 import phase_contract
    need(m.get('observation_contract')==old.observation_binding() and m.get('phase_contract_version')==phase_contract.VERSION and m.get('phase_contract_sha256')==file_sha(phase_contract.__file__),'Original task/phase source differs')
    need(type(m.get('episodes')) is list and m['episodes'] and all(type(e) is dict for e in m['episodes']),'Source records required')
    if 'protocol_sha256' in m:_relative(root,'protocol.json',m['protocol_sha256'],artifacts)
    # Dynamics consumes JSON metadata only, including unvisited split descriptors.
    _relative(root,m['dynamics_contract_path'],m['dynamics_contract_sha256'],artifacts)
    for rec in m['episodes']:
        for key in ('dynamics_contract','outcome','qualification'):_relative(root,rec[key+'_path'],rec[key+'_sha256'],artifacts)
    from sim.auscultation_pi05_v6.dynamics_binding import verify_manifest_dynamics
    dynamics=verify_manifest_dynamics(path,m);_recheck(artifacts)
    return m,dynamics


def _capture_record(root,record,report,artifacts):
    for key in ('derivation','outcome','dynamics_contract','qualification'):
        if key+'_path' in record:_relative(root,record[key+'_path'],record[key+'_sha256'],artifacts)
    phase=report['evidence']['joint_action_derivation']['phase_derivation']
    audit=phase['independent_supervisor_audit'];_relative(root,audit['path'],audit['sha256'],artifacts)
    task=report['evidence']['task_observation_derivation'];parent=task['parent_episode'];_relative(root,parent['path'],parent['sha256'],artifacts)
    evidence=root/task['evidence_root']
    need(evidence==evidence.resolve() and evidence.is_dir() and evidence.is_relative_to(root),'Task evidence root escaped or aliased')
    for name,h in task['input_sha256'].items():_relative(evidence,name,h,artifacts)


def _qualify_record(root,record,source_manifest,normalization,artifacts):
    """All old per-record gates, then fixed-B roundtrip; no split-wide loader."""
    old=_legacy();path=_relative(root,record['path'],record['sha256'],artifacts)
    qp=_relative(root,record['qualification_path'],record['qualification_sha256'],artifacts)
    report=_read_pinned(qp,record['qualification_sha256'],artifacts)
    need(report.get('schema')=='wam_vla_episode_qualification_v1' and report.get('episode_id')==record['episode_id'] and report.get('arrays_sha256')==record['sha256'] and report.get('data_kind')=='closed_loop_rollout' and report.get('qualified') is True,'Qualified source episode required')
    need(report.get('phase_contract_version')==source_manifest['phase_contract_version'] and report.get('phase_contract_sha256')==source_manifest['phase_contract_sha256'],'Qualification phase differs')
    need((report.get('action_contract_version'),report.get('action_contract_sha256')) in old.COMPATIBLE_QUALIFICATIONS|{(old.ACTION_CONTRACT_VERSION,file_sha(Path(old.__file__).with_name('action_contract.py')))},'Qualification action source differs')
    need(all(report.get('checks',{}).get(k) is True for k in old.QUALIFICATION_CHECKS),'Missing original qualification check')
    old.validate_modalities(report.get('modalities'));_capture_record(root,record,report,artifacts)
    with np.load(path,allow_pickle=False) as z:arrays={name:z[name] for name in z.files}
    starts=old._validate_arrays(arrays)
    # The old fit checked this domain. Preserve it with the saved state codec,
    # without fitting either state or action statistics on the new episodes.
    old.normalize(arrays['state'],normalization['state'])
    from sim.auscultation_pi05_v6.action_encoding import inspect_support
    support=inspect_support(arrays['actions'])
    need(all(support.get(k) is True for k in ('strict_interior_supported','roundtrip_within_declared_tolerance')),'Original legacy label support failed')
    from sim.auscultation_pi05_v6.derive_task_dataset import verify_task_derivation
    need(verify_task_derivation(root,record,report,arrays) is True,'Task derivation failed')
    phase=report['evidence']['joint_action_derivation']['phase_derivation']
    need(phase.get('phase_contract_sha256')==source_manifest['phase_contract_sha256'] and phase.get('contact_supervisor_equivalence_verified') is True,'Shared phase/contact replay required')
    old.verify_observed_audit(root,phase,record)
    starts=old.phase_safe_starts(starts,phase.get('cross_boundary_label_indices'),len(arrays['state']))
    errors=[]
    for start in starts:
        p=str(arrays['phases'][start]);q=arrays['state'][start,:7];labels=arrays['actions'][start:start+10]
        z=codec.encode_chunk(labels,q,p,normalization['residual_actions'],mode=codec.B)
        restored=codec.decode_chunk(z,q,p,normalization['residual_actions'],mode=codec.B)
        errors.append(float(np.max(abs(restored.astype(float)-labels))))
    need(max(errors)<=2e-6,'Fixed B label roundtrip exceeds original tolerance')
    _recheck(artifacts)
    del arrays['rgb']
    for array in arrays.values():array.setflags(write=False)
    return dict(record,**arrays,_path=path),starts,dict(legacy_action_support=support,fixed_B_roundtrip_max_rad=max(errors))


def _test_authorization(auth,identity,artifacts):
    need(type(auth) is dict and set(auth)=={'selection_lock_binding','reservation_binding'},'Explicit locked test authorization required')
    from .evaluation import verify_selection_lock
    lb=auth['selection_lock_binding'];rb=auth['reservation_binding']
    for b in (lb,rb):need(type(b) is dict and set(b)=={'path','sha256'},'Exact test binding required');_pin(b['path'],b['sha256'],artifacts)
    lock=verify_selection_lock(lb['path'],expected_sha=lb['sha256'])
    need(lock.get('schema')=='wam_pi05_v10_recovery_r5_selection_lock_v1','V10 selection lock required')
    selected=lock['selection']['selected'];run=Path(lock['selection']['run'])
    need(type(selected) is dict and selected['checkpoint_binding']['data_hash']==identity['data_hash'] and lock['selection']['dataset_binding']==identity,'Selected dataset differs')
    need(Path(rb['path'])==run/'FINAL_TEST_STARTED.json','Original test reservation path required')
    reservation=_read_pinned(rb['path'],rb['sha256'],artifacts)
    expected=dict(schema='wam_pi05_v10_recovery_final_test_reservation_v1',run=str(run),selection_lock_binding=lb,selected_checkpoint_binding=selected['checkpoint_binding'],dataset_binding=identity,test_split='test',episode_id='L5_observed_armature02_v1',status='reserved')
    need(reservation==expected,'Exclusive selected-only test reservation differs');_recheck(artifacts)


def _array_digest(episodes,indices):
    values=[]
    for e in episodes:
        if 'state' in e:
            values.append((e['episode_id'],[(k,str(e[k].dtype),list(e[k].shape),hashlib.sha256(e[k].tobytes()).hexdigest()) for k in sorted(_legacy().ARRAY_KEYS-{'rgb'})]))
    return digest(dict(arrays=values,indices=indices))


@dataclass
class Dataset:
    manifest:dict
    episodes:list
    indices:dict
    normalization:dict
    dynamics:dict
    loaded_scopes:tuple
    qualification_scope_binding:dict
    _identity:dict
    _identity_artifacts:dict
    _scoped_artifacts:dict
    _array_sha:str
    _rgb_cache:OrderedDict=field(default_factory=OrderedDict,repr=False)
    mode:str=MODE
    @property
    def data_hash(self):return self._identity['data_hash']
    @property
    def source_sha256(self):return self._identity['source_sha256']
    def rgb(self,episode_index):
        need(type(episode_index) is int and 0<=episode_index<len(self.episodes),'Known episode required')
        e=self.episodes[episode_index];need(e['split'] in self.loaded_scopes and 'state' in e,'RGB scope not qualified')
        if episode_index not in self._rgb_cache:
            _pin(e['_path'],e['sha256'],{})
            with np.load(e['_path'],allow_pickle=False) as z:value=z['rgb']
            value.setflags(write=False);self._rgb_cache[episode_index]=value
            while sum(x.nbytes for x in self._rgb_cache.values())>2*2**30 and len(self._rgb_cache)>1:self._rgb_cache.popitem(last=False)
        self._rgb_cache.move_to_end(episode_index);return self._rgb_cache[episode_index]


def load_dataset(manifest,*,scopes,normalization_path,expected_normalization_sha256,test_authorization=None):
    need(isinstance(scopes,(list,tuple)) and scopes and len(set(scopes))==len(scopes) and set(scopes)<=set(SPLITS),'Explicit unique dataset scopes required')
    scopes=tuple(s for s in SPLITS if s in scopes)
    need(('test' in scopes)==(test_authorization is not None),'Test scope requires exactly its explicit authorization')
    artifacts={};manifest=_regular(Path(manifest));m=_read_pinned(manifest,file_sha(manifest),artifacts)
    need(set(m)=={'schema','sealed','design_binding','sources','episodes'} and m['schema']==SCHEMA and m['sealed'] is True,'Exact sealed composite manifest required')
    b=m['design_binding'];need(set(b)=={'path','sha256'} and b['sha256']==DESIGN_SHA256,'Approved design required')
    design=_read_pinned(b['path'],b['sha256'],artifacts);norm=_normalization(normalization_path,expected_normalization_sha256,artifacts);sources=core_sources()
    originals=design['data']['train_original8'];added=design['data']['train_existing5']+design['data']['train_new3'];allids=originals+[design['data']['validation_episode'],design['data']['test_episode']]+added
    need(norm['residual_actions']['fit_episode_ids']==originals,'Fixed stats must retain original8 fitting IDs')
    need(type(m['sources']) is list and [x.get('id') for x in m['sources']]==['original10']+added,'Exact nine ordered source manifests required')
    containers={};records={};dynamics={};source_bindings=[]
    for source in m['sources']:
        need(set(source)=={'id','manifest'} and set(source['manifest'])=={'path','sha256'},'Exact source binding required')
        sb=source['manifest'];sid=source['id']
        if sid=='original10':need(sb['sha256']==ORIGINAL_MANIFEST_SHA256,'Original10 source manifest changed')
        sm,dyn=_source_metadata(sb['path'],sb['sha256'],artifacts);expected=allids[:10] if sid=='original10' else [sid]
        need([r['episode_id'] for r in sm['episodes']]==expected,'Unexpected source episode set/order')
        containers[sid]=(Path(sb['path']).parent,sm);dynamics[sid]=dyn;source_bindings.append(source)
        for rec in sm['episodes']:records[rec['episode_id']]=(sid,copy.deepcopy(rec))
    need(type(m['episodes']) is list and all(set(x)=={'episode_id','source_id'} for x in m['episodes']) and [x['episode_id'] for x in m['episodes']]==allids,'Exact18 unique source-record mappings required')
    resolved=[]
    for item in m['episodes']:
        sid,rec=records[item['episode_id']];need(sid==item['source_id'],'Source record remapped')
        expected='validation' if rec['episode_id']==design['data']['validation_episode'] else 'test' if rec['episode_id']==design['data']['test_episode'] else 'train'
        need(rec['split']==expected and rec['target']==rec['episode_id'].split('_')[0],'Episode split/target differs');resolved.append(rec)
    need(len({r['sha256'] for r in resolved})==18,'Duplicate episode content cannot be a new group')
    _legacy().validate_groups(resolved,heldout_axes=['motion_family'])
    from sim.auscultation_pi05_v6.dynamics_contract import require_compatible
    first=dynamics['original10']
    for own in dynamics.values():require_compatible(first['contract'],own['contract'],scope='simulator_recipe')
    identity=dict(schema=BINDING_SCHEMA,mode=MODE,manifest_binding=dict(path=str(manifest),sha256=file_sha(manifest)),source_manifests=source_bindings,design_binding=b,normalization_file_binding=dict(path=str(_regular(normalization_path)),sha256=expected_normalization_sha256),normalization_sha256=digest(norm),normalization_object_sha256=object_digest(norm),action_encoding_binding=norm['action_encoding_binding'],source_sha256=sources,training_episode_ids=originals+added,normalization_fit_episode_ids=originals,split_episode_ids={s:[r['episode_id'] for r in resolved if r['split']==s] for s in SPLITS})
    identity['data_hash']=digest(identity);identity_artifacts=dict(artifacts)
    if 'test' in scopes:_test_authorization(test_authorization,identity,artifacts)
    episodes=[];indices={s:[] for s in SPLITS};qualified=[];support={}
    for i,record in enumerate(resolved):
        if record['split'] in scopes:
            sid,_=records[record['episode_id']];root,source=containers[sid]
            episode,starts,sup=_qualify_record(root,record,source,norm,artifacts)
            indices[record['split']].extend((i,start) for start in starts);qualified.append(record['episode_id']);support[record['episode_id']]=sup
        else:episode=copy.deepcopy(record)
        episodes.append(episode)
    _recheck(artifacts);need(core_sources()==sources,'Source changed during qualification')
    scope=dict(schema='wam_pi05_v10_qualification_scope_v1',loaded_scopes=list(scopes),verified_episode_ids=qualified,artifact_sha256=dict(artifacts),per_episode_support=support,qualification_scope='source attestations plus unchanged original numeric/task/supervisor/dynamics checks; not a new acquisition or policy acceptance')
    return Dataset(m,episodes,indices,norm,first,scopes,scope,identity,identity_artifacts,artifacts,_array_digest(episodes,indices))


def _state_ok(dataset):
    need(isinstance(dataset,Dataset),'Qualified V10 dataset required')
    need(dataset.mode==MODE and digest(dataset.normalization)==NORMALIZATION_SHA256 and object_digest(dataset.normalization)==NORMALIZATION_OBJECT_SHA256,'Fixed B normalization changed')
    need(dataset.source_sha256==core_sources(),'V10 data sources changed')
    need(dataset.data_hash==digest({k:v for k,v in dataset._identity.items() if k!='data_hash'}),'Dataset identity changed')


def dataset_binding(dataset):
    _state_ok(dataset);_recheck(dataset._scoped_artifacts)
    need(_array_digest(dataset.episodes,dataset.indices)==dataset._array_sha,'Qualified arrays/indices changed')
    return copy.deepcopy(dataset._identity)


def make_batch(dataset,split,indices,tokenizer):
    _state_ok(dataset)
    need(split in dataset.loaded_scopes and isinstance(indices,(list,tuple,np.ndarray)) and len(indices)>0,'Known qualified nonempty split required')
    from sim.auscultation_pi05_v7 import training_data as original
    observations=[];actions=[]
    for index in indices:
        need(isinstance(index,(int,np.integer)) and not isinstance(index,(bool,np.bool_)) and 0<=int(index)<len(dataset.indices[split]),'Invalid batch index')
        ei,start=dataset.indices[split][int(index)];e=dataset.episodes[ei];need(e['split']==split,'Batch scope mismatch')
        raw=e['state'][start];phase=str(e['phases'][start]);target=str(e['targets'][start])
        observations.append(original.observation(dataset.rgb(ei)[start],raw,phase,target,tokenizer,dataset.normalization))
        actions.append(codec.encode_chunk(e['actions'][start:start+10],raw[:7],phase,dataset.normalization['residual_actions'],mode=codec.B))
    return original.stack_observations(observations),np.stack(actions)


def decode_prediction(dataset,normalized,state,phase):
    _state_ok(dataset);raw=codec.parent._array(state,(41,),'original41 observation')
    return codec.decode_chunk(normalized,raw[:7],phase,dataset.normalization['residual_actions'],mode=codec.B)


@dataclass(frozen=True)
class OriginalTrainView:
    parent:Dataset
    episodes:list
    indices:dict
    normalization:dict
    qualification_scope_binding:dict
    scope:str='original8 subset of qualified V10 TRAIN; not historical full dataset qualification'


def original_train_view(dataset):
    dataset_binding(dataset);need('train' in dataset.loaded_scopes,'Original subset requires qualified TRAIN')
    ids=dataset._identity['normalization_fit_episode_ids'];lookup={e['episode_id']:i for i,e in enumerate(dataset.episodes)}
    selected=[lookup[eid] for eid in ids];remap={old:new for new,old in enumerate(selected)}
    indices={s:[] for s in SPLITS};indices['train']=[(remap[ei],start) for ei,start in dataset.indices['train'] if ei in remap]
    return OriginalTrainView(dataset,[dataset.episodes[i] for i in selected],indices,dataset.normalization,dataset.qualification_scope_binding)


# Frozen dependencies consumed by per-record qualification and original B preprocessing.
EXPECTED_SOURCES = {'sim/auscultation_pi05_v6/action_contract.py': 'd21e73ec9e530b0a3785debb9e78cd42648f4de45880221317ba982aa44b7596', 'sim/auscultation_pi05_v6/action_encoding.py': '0b1fd40429365493e0684f39865e6569b02e3e75ae5e288d891dbd93a2e3985c', 'sim/auscultation_pi05_v6/derive_task_dataset.py': 'd92ae5ba32d3ab0f29dbc98b77c522747146fbd489805a40c430c0f07c5bd4a6', 'sim/auscultation_pi05_v6/dynamics_binding.py': 'ad5e6ba8de38e8ef6f83d52b427abb43fa81fbc0aaf4ea0b3ab4acf9bc888a80', 'sim/auscultation_pi05_v6/dynamics_contract.py': '2c622501c39c78fdcbb3ec072ea04517b36bf94522f08891d789b4b19356728a', 'sim/auscultation_pi05_v6/evaluation.py': '927d03e0db2810b4f468df8cb40f9460cd99030c7c1410ea2f1387ff4819ac49', 'sim/auscultation_pi05_v6/observed_supervisor.py': 'c493be864050f479c21c112cf68e76aa377f8d101419416ded245df767371ed4', 'sim/auscultation_pi05_v6/openpi_adapter.py': '023e74899c7b54260307e9b86e1e859e573a5e3947b4829fbe78cff9bddee444', 'sim/auscultation_pi05_v6/phase_contract.py': 'a6d7edb5723c09fd5e4976155bc0d6587869776ebb262adcb761e51c2ce358b2', 'sim/auscultation_pi05_v6/state_encoding.py': '50ccfc5f374cd412bff6aa45dc8fc7fd7d10b66776387c8056657eb9167dabaf', 'sim/auscultation_pi05_v6/supervision.py': 'd872f5d4256f6d4030d29dab3204015cee6378bc3286b4bcef0462e35e83ee49', 'sim/auscultation_pi05_v6/task_observation.py': '542fe25980121464a0258ac4884e1ff19deae98fa20ba368fdf6043d39e04b5f', 'sim/auscultation_pi05_v6/train_openpi.py': '6305fb8831894bdfb0fb17cc0dfa68efc3da90bd4ffa468b582696c60125112f', 'sim/auscultation_pi05_v6/training_data.py': 'e89025be7c00f828c87693533500d29f383241539d8971ba25d4680fb49abcbe', 'sim/auscultation_pi05_v7/action_encoding.py': '709e0d67631e3164678198c7e50d9abf702421183e1ea3cb60d1e6c44817b885', 'sim/auscultation_pi05_v7/coverage_pilot_evaluation.py': '87410026c3c07167badd6feee8c19c075ba2a90624289a1a0218dcf8070d820c', 'sim/auscultation_pi05_v7/coverage_pilot_schedule.py': 'ff8d09b8c4d66e51341021b3ba8485de4a1e735a6559a31e4e717420af6d3b65', 'sim/auscultation_pi05_v7/coverage_pilot_selection.py': 'c678909c90ede4b2c526bfa90bf3de742124cdc0fd2d1030ba76122803bb02eb', 'sim/auscultation_pi05_v7/model.py': 'ffd76935d42904f69000317dccc4bd280fcb3eba3202e0acf6dfcf14223ee2cc', 'sim/auscultation_pi05_v7/training_data.py': '4da99cb8bd4632df380480b92e76a872acb0bcf993aa6781b6d9e280da7dbe3f', 'sim/auscultation_pi05_v8/action_encoding.py': 'a694add3d8becbc99fdc18b9c4ad9d10823339f2a674fcc62130c9147331ae23', 'sim/auscultation_pi05_v8/evaluation.py': '1babc07f8f53c833d39d761efb077e74ec1e9a8f4a305669b1bc7c05b848c5b9', 'sim/auscultation_pi05_v8/train.py': '89dbf8238e4041966cb221865f2269624dd870b70393568526c51ef0056f1d39', 'sim/auscultation_pi05_v8/training_data.py': '86dc1e157012fec6983a68ee90940d5718b716ee6cd3a83f8faef447cc761172', 'sim/auscultation_pi05_v6/export_joint_dataset.py': '1f4da9baf7fcde2f9d7a92fc18f816632687565cb7097e857a8c4b37b68358ba', 'sim/auscultation_pi05_v6/audit_observed_supervisor.py': 'f032b25cf898403f9e6c8b4f92eefbfa1f33a1e62997e8bd598b66cd9f3add64', 'sim/auscultation_vla_v2/chunks.py': '88c6403f65660f8b23f1bdcbd020e80b4472cfac946130448f04b70661366cf6', 'sim/auscultation_vla_v2/multimodal.py': '712507a4bd7de1e15eb173c4ac1b5b3a71571cc4479180b97f9e2b10ad48d2f1', 'sim/auscultation_vla_v2/hall.py': '8ccf46f67d32d6229175ab31d14e5a7ef55d175b29da9ddf39a27e935c1f1b6e', 'sim/auscultation_expert_layout_v1/kinematics.py': 'b86d682a7cf7c0589757d59546a9dd619149460a431a75b2bf685fc9090c2f66', 'sim/auscultation_expert_layout_v1/tool_frames.py': '3dd6a5af8911f1b2772e67eef7a8c0406fa06eb8ff646b88fa3bc92d1b332a94', 'auscultation/mechanical_sensor/force_feedback/force_sensor.py': '09f656f179301728ec955d7aea385c13a8968589359974c15b91c6f40261abce'}
