"""V11 target-aware inverse-coordinate labels over the qualified V10 sources.

Acquisition/phase/Hall/dynamics checks are inherited unchanged. Only TRAIN fits
latent statistics. Heldout observation construction never inverse-encodes labels.
"""
import copy
from dataclasses import dataclass, field
from collections import OrderedDict
from pathlib import Path
import numpy as np
from sim.auscultation_pi05_v10 import training_data as parent_D
from . import action_encoding as codec

MODE='V11_target_interval_latent_residual_v1'
DESIGN_SHA256='aaaac1cdf4d46fa67db49360701e7dd5d8f3cef3960ea8442b8a1d194cb11c6e'
DESIGN=Path('/home/william/wam-vla-v2/diagnostics/pi05_v11_bounded_coordinate_design_v1/machine_spec_v1.json')
PHASES=parent_D.PHASES
SPLITS=parent_D.SPLITS
need=parent_D.need
file_sha=parent_D.file_sha
json_bytes=parent_D.json_bytes
digest=parent_D.digest
object_digest=parent_D.object_digest
_regular=parent_D._regular
_pin=parent_D._pin
_read_pinned=parent_D._read_pinned
_recheck=parent_D._recheck


def design(artifacts=None):
    return _read_pinned(DESIGN,DESIGN_SHA256,{} if artifacts is None else artifacts)


def core_sources():
    sources=parent_D.core_sources();root=Path(__file__).resolve().parents[2]
    sources['sim/auscultation_pi05_v10/schedule.py']=file_sha(root/'sim/auscultation_pi05_v10/schedule.py')
    for name in ('__init__.py','action_encoding.py','training_data.py'):
        sources['sim/auscultation_pi05_v11/'+name]=file_sha(Path(__file__).with_name(name))
    for name,h in codec.SOURCE_SHA256.items():
        need(file_sha(root/name)==h,'Approved codec source changed');sources[name]=h
    return sources


def original_ids(dataset):return list(dataset._identity['parent_dataset_binding']['normalization_fit_episode_ids'])


def train_views(parent):
    parent_D.dataset_binding(parent)
    need('train' in parent.loaded_scopes and len(parent.indices['train'])==24397,'All24397 qualified TRAIN starts required')
    starts={i:[] for i,e in enumerate(parent.episodes) if e['split']=='train'}
    for ei,s in parent.indices['train']:starts[ei].append(int(s))
    views=[dict(episode_id=e['episode_id'],split='train',actions=e['actions'],state=e['state'],phases=e['phases'],targets=e['targets'],starts=starts[i]) for i,e in enumerate(parent.episodes) if e['split']=='train']
    need(len(views)==16,'Exact16 TRAIN episodes required');return views


def fit_normalization(parent,bounds):
    expected=design()['data']['qualified_parent_dataset_binding']
    need(parent_D.dataset_binding(parent)==expected,'Qualified parent dataset changed')
    norm=dict(schema='wam_pi05_v11_bounded_normalization_v1',mode=MODE,
        state=copy.deepcopy(parent.normalization['state']),encoding_binding=copy.deepcopy(parent.normalization['encoding_binding']),
        residual_actions=codec.fit_stats(train_views(parent),bounds),action_encoding_binding=codec.action_encoding_binding(bounds))
    need(object_digest(norm['state'])==design()['normalization']['state_object_sha256'],'Original state normalization differs')
    return norm


@dataclass
class Dataset:
    parent:object
    manifest:dict
    episodes:list
    indices:dict
    normalization:dict
    dynamics:dict
    loaded_scopes:tuple
    qualification_scope_binding:dict
    _identity:dict
    _scoped_artifacts:dict
    _array_sha:str
    _rgb_cache:OrderedDict=field(default_factory=OrderedDict,repr=False)
    mode:str=MODE
    @property
    def _identity_artifacts(self):return self._scoped_artifacts
    @property
    def data_hash(self):return self._identity['data_hash']
    @property
    def source_sha256(self):return self._identity['source_sha256']
    rgb=parent_D.Dataset.rgb


def _test_authorization(auth,identity,artifacts):
    need(type(auth) is dict and set(auth)=={'selection_lock_binding','reservation_binding'},'Explicit V11 locked test authorization required')
    from .evaluation import verify_selection_lock
    lb=auth['selection_lock_binding'];rb=auth['reservation_binding']
    for b in (lb,rb):need(type(b) is dict and set(b)=={'path','sha256'},'Exact test binding required');_pin(b['path'],b['sha256'],artifacts)
    lock=verify_selection_lock(lb['path'],expected_sha=lb['sha256'])
    need(lock.get('schema')=='wam_pi05_v11_bounded_r5_selection_lock_v1','V11 selection lock required')
    selected=lock['selection']['selected'];run=Path(lock['selection']['run'])
    need(type(selected) is dict and selected['checkpoint_binding']['data_hash']==identity['data_hash'] and lock['selection']['dataset_binding']==identity,'Selected V11 dataset differs')
    need(Path(rb['path'])==run/'FINAL_TEST_STARTED.json','Original test reservation path required')
    reservation=_read_pinned(rb['path'],rb['sha256'],artifacts)
    expected=dict(schema='wam_pi05_v11_bounded_final_test_reservation_v1',run=str(run),selection_lock_binding=lb,selected_checkpoint_binding=selected['checkpoint_binding'],dataset_binding=identity,test_split='test',episode_id='L5_observed_armature02_v1',status='reserved')
    need(reservation==expected,'Exclusive selected-only V11 test reservation differs');_recheck(artifacts)


def load_dataset(manifest,*,scopes,normalization_path,expected_normalization_sha256,test_authorization=None):
    need(isinstance(scopes,(list,tuple)) and scopes and len(set(scopes))==len(scopes) and set(scopes)<=set(SPLITS),'Explicit unique scopes required')
    need(('test' in scopes)==(test_authorization is not None),'Only authorized test scope may open L5')
    # TRAIN is always explicitly qualified to independently refit/verify the
    # immutable statistics. This extra loaded scope is recorded, never hidden.
    loaded=tuple(s for s in SPLITS if s in scopes or s=='train')
    artifacts={};spec=design(artifacts);binding=spec['data']['qualified_parent_dataset_binding']
    need(str(_regular(Path(manifest)))==binding['manifest_binding']['path'],'Original V10 composite path required')
    _pin(manifest,binding['manifest_binding']['sha256'],artifacts)
    nb=binding['normalization_file_binding']
    parent=parent_D.load_dataset(manifest,scopes=[s for s in loaded if s!='test'],normalization_path=nb['path'],expected_normalization_sha256=nb['sha256'])
    need(parent_D.dataset_binding(parent)==binding,'Exact qualified parent identity required')
    artifacts.update(parent.qualification_scope_binding['artifact_sha256'])
    norm=_read_pinned(normalization_path,expected_normalization_sha256,artifacts)
    need(json_bytes(norm)==Path(normalization_path).read_bytes(),'Canonical V11 normalization serialization required')
    bounds=codec.build_bounds_contract(Path(__file__).resolve().parents[2])
    expected=fit_normalization(parent,bounds)
    need(norm==expected,'V11 latent statistics differ from complete TRAIN-only fit')
    sources=core_sources();identity=dict(schema='wam_pi05_v11_bounded_dataset_binding_v1',mode=MODE,
        manifest_binding=copy.deepcopy(binding['manifest_binding']),parent_dataset_binding=binding,
        design_binding=dict(path=str(DESIGN),sha256=DESIGN_SHA256),
        normalization_file_binding=dict(path=str(_regular(Path(normalization_path))),sha256=expected_normalization_sha256),
        normalization_sha256=digest(norm),normalization_object_sha256=object_digest(norm),
        action_encoding_binding=norm['action_encoding_binding'],source_sha256=sources,
        training_episode_ids=binding['training_episode_ids'],normalization_fit_episode_ids=binding['training_episode_ids'],split_episode_ids=binding['split_episode_ids'])
    identity['data_hash']=digest(identity)
    episodes=list(parent.episodes);indices=copy.deepcopy(parent.indices);support={}
    if 'test' in loaded:
        # Authorization precedes the first L5 NPZ read. Parent itself never
        # receives fabricated V10 authorization or mutates its qualified scope.
        _test_authorization(test_authorization,identity,artifacts)
        originals=binding['source_manifests'][0]['manifest'];sm,dyn=parent_D._source_metadata(originals['path'],originals['sha256'],artifacts)
        for ei,e in enumerate(episodes):
            if e['split']!='test':continue
            need(e['episode_id']=='L5_observed_armature02_v1','Only reserved L5 allowed')
            record=next(r for r in sm['episodes'] if r['episode_id']==e['episode_id'])
            episodes[ei],starts,sup=parent_D._qualify_record(Path(originals['path']).parent,record,sm,parent.normalization,artifacts)
            indices['test']=[(ei,int(s)) for s in starts];support[e['episode_id']]=sup
    _recheck(artifacts)
    scope=dict(schema='wam_pi05_v11_bounded_qualification_scope_v1',requested_scopes=list(scopes),loaded_scopes=list(loaded),
        verified_episode_ids=[e['episode_id'] for e in episodes if 'state' in e],artifact_sha256=artifacts,
        parent_qualification=parent.qualification_scope_binding,test_original_qualification=support,
        normalization_fit='all24397 TRAIN starts once; heldout reference support checked separately; not policy acceptance')
    return Dataset(parent,parent.manifest,episodes,indices,norm,parent.dynamics,loaded,scope,identity,artifacts,parent_D._array_digest(episodes,indices))


def _state_ok(dataset):
    need(isinstance(dataset,Dataset) and dataset.mode==MODE,'Qualified V11 dataset required')
    need(digest(dataset.normalization)==dataset._identity['normalization_sha256'],'V11 normalization changed')
    need(dataset.data_hash==digest({k:v for k,v in dataset._identity.items() if k!='data_hash'}),'V11 identity changed')


def dataset_binding(dataset):
    _state_ok(dataset);need(parent_D.dataset_binding(dataset.parent)==dataset._identity['parent_dataset_binding'],'Parent changed')
    need(core_sources()==dataset.source_sha256,'V11 source changed');_recheck(dataset._scoped_artifacts)
    need(parent_D._array_digest(dataset.episodes,dataset.indices)==dataset._array_sha,'Qualified arrays/indices changed')
    return copy.deepcopy(dataset._identity)


def _rows(dataset,split,indices):
    _state_ok(dataset);need(split in dataset.loaded_scopes and isinstance(indices,(list,tuple,np.ndarray)) and len(indices)>0,'Nonempty qualified split required')
    for index in indices:
        need(isinstance(index,(int,np.integer)) and not isinstance(index,(bool,np.bool_)) and 0<=int(index)<len(dataset.indices[split]),'Invalid batch index')
        ei,s=dataset.indices[split][int(index)];e=dataset.episodes[ei]
        phase=str(e['phases'][s]);target=str(e['targets'][s])
        need(e['split']==split and e['target']==target and target==e['episode_id'].split('_')[0],'Requested row/target mismatch')
        need(all(str(x)==phase for x in e['phases'][s:s+10]) and all(str(x)==target for x in e['targets'][s:s+10]),'Phase/target chunk mismatch')
        yield ei,s,e,phase,target


def make_observation_batch(dataset,split,indices,tokenizer):
    from sim.auscultation_pi05_v7 import training_data as O
    observations=[O.observation(dataset.rgb(ei)[s],e['state'][s],phase,target,tokenizer,dataset.normalization) for ei,s,e,phase,target in _rows(dataset,split,indices)]
    return O.stack_observations(observations)


def make_batch(dataset,split,indices,tokenizer):
    need(split=='train','Only TRAIN labels may become optimizer targets')
    obs=make_observation_batch(dataset,split,indices,tokenizer)
    actions=[codec.encode_chunk(e['actions'][s:s+10],e['state'][s,:7],phase,target,dataset.normalization['residual_actions']) for ei,s,e,phase,target in _rows(dataset,split,indices)]
    return obs,np.stack(actions)


def decode_prediction(dataset,normalized,state,phase,target):
    _state_ok(dataset);raw=np.asarray(state,dtype=np.float64)
    need(raw.shape==(41,) and np.isfinite(raw).all(),'Original finite41 observation required')
    return codec.decode_chunk(normalized,raw[:7],phase,target,dataset.normalization['residual_actions'])


def original_train_view(dataset):
    dataset_binding(dataset)
    return parent_D.original_train_view(dataset.parent)
