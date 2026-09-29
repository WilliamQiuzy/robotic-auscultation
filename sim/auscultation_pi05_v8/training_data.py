"""Qualified V7 parent references with an explicit paired action-only view.

Only load_dataset establishes the fixed qualified acquisition identity. The pure
view_parent helper accepts an already-loaded parent; it never certifies a new
acquisition. Original arrays, indices, RGB cache and observations are delegated.
"""
from dataclasses import dataclass
import hashlib
import json
from pathlib import Path
import numpy as np
from . import action_encoding as codec

MANIFEST_SHA256='ce5da81634870e79cc8442d4ec166387966cc1e7c125942a16ebe98d73dcc9b0'
PARENT_DATA_HASH='9c40deac72e4b54c2c921867a488022c18c36c34e2cf310ed27bace8d40cf34f'
EXPECTED_PARENT_SOURCES={'v6/action_contract.py': 'd21e73ec9e530b0a3785debb9e78cd42648f4de45880221317ba982aa44b7596',
 'v6/action_encoding.py': '0b1fd40429365493e0684f39865e6569b02e3e75ae5e288d891dbd93a2e3985c',
 'v6/derive_task_dataset.py': 'd92ae5ba32d3ab0f29dbc98b77c522747146fbd489805a40c430c0f07c5bd4a6',
 'v6/dynamics_binding.py': 'ad5e6ba8de38e8ef6f83d52b427abb43fa81fbc0aaf4ea0b3ab4acf9bc888a80',
 'v6/dynamics_contract.py': '2c622501c39c78fdcbb3ec072ea04517b36bf94522f08891d789b4b19356728a',
 'v6/evaluation.py': '927d03e0db2810b4f468df8cb40f9460cd99030c7c1410ea2f1387ff4819ac49',
 'v6/observed_supervisor.py': 'c493be864050f479c21c112cf68e76aa377f8d101419416ded245df767371ed4',
 'v6/openpi_adapter.py': '023e74899c7b54260307e9b86e1e859e573a5e3947b4829fbe78cff9bddee444',
 'v6/phase_contract.py': 'a6d7edb5723c09fd5e4976155bc0d6587869776ebb262adcb761e51c2ce358b2',
 'v6/state_encoding.py': '50ccfc5f374cd412bff6aa45dc8fc7fd7d10b66776387c8056657eb9167dabaf',
 'v6/supervision.py': 'd872f5d4256f6d4030d29dab3204015cee6378bc3286b4bcef0462e35e83ee49',
 'v6/task_observation.py': '542fe25980121464a0258ac4884e1ff19deae98fa20ba368fdf6043d39e04b5f',
 'v6/train_openpi.py': '6305fb8831894bdfb0fb17cc0dfa68efc3da90bd4ffa468b582696c60125112f',
 'v6/training_data.py': 'e89025be7c00f828c87693533500d29f383241539d8971ba25d4680fb49abcbe',
 'v7/action_encoding.py': '709e0d67631e3164678198c7e50d9abf702421183e1ea3cb60d1e6c44817b885',
 'v7/model.py': 'ffd76935d42904f69000317dccc4bd280fcb3eba3202e0acf6dfcf14223ee2cc',
 'v7/training_data.py': '4da99cb8bd4632df380480b92e76a872acb0bcf993aa6781b6d9e280da7dbe3f'}


def _parent_module():
    # Avoid native/state dependencies until actual qualification or batching.
    from sim.auscultation_pi05_v7 import training_data
    return training_data


def _need(condition,message):
    if not condition:raise ValueError(message)


def _sha(path):
    h=hashlib.sha256()
    with Path(path).open('rb') as f:
        for part in iter(lambda:f.read(8*1024*1024),b''):h.update(part)
    return h.hexdigest()


def _digest(value):
    return hashlib.sha256(json.dumps(value,sort_keys=True,separators=(',',':'),allow_nan=False).encode()).hexdigest()


def core_sources():
    inherited=_parent_module().core_sources()
    _need(inherited==EXPECTED_PARENT_SOURCES,'Original 17 training sources changed')
    own=Path(__file__).parent
    return dict(inherited,**{'v8/'+n:_sha(own/n) for n in ('action_encoding.py','training_data.py')})


def _records(base):
    _need(isinstance(base.indices,dict) and set(base.indices)=={'train','validation','test'},'Original split indices required')
    starts={i:[] for i,e in enumerate(base.episodes) if e['split']=='train'}
    seen=set()
    for row in base.indices['train']:
        _need(isinstance(row,(list,tuple)) and len(row)==2,'Invalid train window reference')
        ei,start=row
        _need(not isinstance(ei,(bool,np.bool_)) and isinstance(ei,(int,np.integer)) and int(ei) in starts,'Train reference points outside TRAIN')
        _need(not isinstance(start,(bool,np.bool_)) and isinstance(start,(int,np.integer)),'Integer observation start required')
        key=(int(ei),int(start));_need(key not in seen,'Duplicate original training window');seen.add(key)
        starts[int(ei)].append(int(start))
    return [dict(base.episodes[i],starts=values) for i,values in starts.items()]


@dataclass(frozen=True)
class Dataset:
    parent: object
    mode: str
    normalization: dict
    data_hash: str
    source_sha256: dict

    @property
    def episodes(self):return self.parent.episodes
    @property
    def indices(self):return self.parent.indices
    @property
    def dynamics(self):return self.parent.dynamics
    @property
    def manifest(self):return self.parent.manifest
    def rgb(self,episode_index):return self.parent.parent.rgb(episode_index)


def view_parent(base,mode):
    """Wrap a qualified parent's references; pure helper is not qualification."""
    sources=core_sources();stats=codec.fit_stats(_records(base),mode)
    _need(stats['parent_pooled_stats']==base.normalization['residual_actions'],'Original pooled parent normalization differs')
    _need(base.normalization.get('action_encoding_binding')==codec.parent.action_encoding_binding(),'Parent action source differs')
    normalization=dict(schema='wam_paired_horizon_dataset_normalization_v1',recipe=codec.recipe(mode),mode=mode,
        state=base.normalization['state'],encoding_binding=base.normalization['encoding_binding'],
        residual_actions=stats,action_encoding_binding=codec.action_encoding_binding(mode))
    data_hash=_digest(dict(parent_data_hash=base.data_hash,recipe=codec.recipe(mode),mode=mode,normalization=normalization,source_sha256=sources))
    return Dataset(base,mode,normalization,data_hash,sources)


def load_dataset(manifest,mode):
    """Run the unchanged full V6/V7 qualification chain on the fixed manifest."""
    codec.action_encoding_binding(mode)
    manifest=Path(manifest)
    _need(not manifest.is_symlink() and manifest.is_file() and _sha(manifest)==MANIFEST_SHA256,'Exact original qualified manifest required')
    core_sources()
    base=_parent_module().load_dataset(manifest)
    _need(base.data_hash==PARENT_DATA_HASH,'Original qualified parent data/normalization hash changed')
    _need(len([e for e in base.episodes if e['split']=='train'])==8 and len(base.indices['train'])==12759,'Original eight TRAIN and 12759 valid windows required')
    return view_parent(base,mode)


def dataset_binding(dataset):
    _need(isinstance(dataset,Dataset),'Explicit paired dataset required')
    _need(dataset.source_sha256==core_sources(),'Paired dataset source changed')
    norm=dataset.normalization
    _need(norm.get('mode')==dataset.mode and norm.get('recipe')==codec.recipe(dataset.mode)
          and norm.get('schema')=='wam_paired_horizon_dataset_normalization_v1','Dataset mode/recipe/schema mismatch')
    stats_hash=codec.validate_stats(norm['residual_actions'],mode=dataset.mode)
    expected=_digest(dict(parent_data_hash=dataset.parent.data_hash,recipe=codec.recipe(dataset.mode),mode=dataset.mode,normalization=norm,source_sha256=dataset.source_sha256))
    _need(dataset.data_hash==expected,'Paired dataset normalization/data hash changed')
    return dict(schema='wam_paired_horizon_dataset_binding_v1',recipe=codec.recipe(dataset.mode),mode=dataset.mode,
        parent_data_hash=dataset.parent.data_hash,data_hash=dataset.data_hash,normalization_sha256=_digest(norm),
        residual_stats_sha256=stats_hash,fit_windows_sha256=norm['residual_actions']['fit_windows_sha256'],
        fit_window_count=norm['residual_actions']['fit_window_count'],source_sha256=dataset.source_sha256,
        parent_action_encoding_binding=codec.parent.action_encoding_binding())


def validate_binding(dataset,expected):
    actual=dataset_binding(dataset)
    _need(isinstance(expected,dict) and expected==actual,'Legacy, other mode or mismatched dataset binding')
    return actual


def make_batch(dataset,split,indices,tokenizer):
    _need(isinstance(dataset,Dataset) and split in dataset.indices,'Known paired dataset/split required')
    _need(isinstance(indices,(list,tuple,np.ndarray)) and len(indices)>0,'Nonempty integer batch indices required')
    rows=[];actions=[];parent=_parent_module()
    for index in indices:
        _need(not isinstance(index,(bool,np.bool_)) and isinstance(index,(int,np.integer)) and 0<=int(index)<len(dataset.indices[split]),'Invalid batch index')
        ei,start=dataset.indices[split][int(index)];episode=dataset.episodes[ei]
        raw=episode['state'][start];phase=str(episode['phases'][start]);target=str(episode['targets'][start])
        rows.append(parent.observation(dataset.rgb(ei)[start],raw,phase,target,tokenizer,dataset.normalization))
        actions.append(codec.encode_chunk(episode['actions'][start:start+10],raw[:7],phase,dataset.normalization['residual_actions'],mode=dataset.mode))
    return parent.stack_observations(rows),np.stack(actions)


def decode_prediction(dataset,normalized,state,phase):
    _need(isinstance(dataset,Dataset),'Explicit paired dataset required')
    raw=codec.parent._array(state,(41,),'original observation state')
    return codec.decode_chunk(normalized,raw[:7],phase,dataset.normalization['residual_actions'],mode=dataset.mode)
