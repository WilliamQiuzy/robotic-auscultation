"""Residual-action view of independently qualified, immutable V6 observations.

Only the model's action coordinates change. Source arrays, labels, qualified
splits, sensor processing, RGB preparation and state tokenization are retained.
"""
from dataclasses import dataclass
import hashlib
from pathlib import Path
import numpy as np
from sim.auscultation_pi05_v6 import train_openpi as parent
from sim.auscultation_pi05_v6.openpi_adapter import prepare_images,tokenize_state
from sim.auscultation_pi05_v6.training_data import normalize
from .action_encoding import fit_stats,encode_chunk,decode_chunk,action_encoding_binding

RECIPE='pi05_v7_fixed_observation_joint_residual_v1'

@dataclass
class Dataset:
    parent: object
    normalization: dict
    data_hash: str

    @property
    def episodes(self):return self.parent.episodes
    @property
    def indices(self):return self.parent.indices
    @property
    def dynamics(self):return self.parent.dynamics
    @property
    def manifest(self):return self.parent.manifest

def load_dataset(manifest):
    base=parent.load_dataset(manifest)
    records=[]
    for ei,episode in enumerate(base.episodes):
        if episode['split']=='train':
            starts=[start for e,start in base.indices['train'] if e==ei]
            records.append(dict(episode,starts=starts))
    stats=fit_stats(records)
    normalization=dict(state=base.normalization['state'],encoding_binding=base.normalization['encoding_binding'],
                       residual_actions=stats,action_encoding_binding=action_encoding_binding())
    digest=hashlib.sha256(parent._json_bytes(dict(parent_data_hash=base.data_hash,recipe=RECIPE,normalization=normalization))).hexdigest()
    return Dataset(base,normalization,digest)

def observation(rgb,state,phase,target,tokenizer,normalization):
    raw=np.asarray(state,np.float32)
    state=normalize(raw,normalization['state'])
    ids,mask=tokenize_state(tokenizer,parent.training_instruction(phase,target),state)
    images,present=prepare_images(rgb)
    return dict(images=images,image_masks=present,state=state,tokenized_prompt=ids,tokenized_prompt_mask=mask)

def stack_observations(rows):
    if not rows:raise ValueError('Nonempty observation batch required')
    return {key:({name:np.stack([r[key][name] for r in rows]) for name in rows[0][key]}
                 if key in ('images','image_masks') else np.stack([r[key] for r in rows])) for key in rows[0]}

def make_batch(dataset,split,indices,tokenizer):
    rows=[];actions=[]
    for index in indices:
        ei,start=dataset.indices[split][int(index)];episode=dataset.episodes[ei]
        raw=episode['state'][start];phase=str(episode['phases'][start]);target=str(episode['targets'][start])
        rows.append(observation(dataset.parent.rgb(ei)[start],raw,phase,target,tokenizer,dataset.normalization))
        actions.append(encode_chunk(episode['actions'][start:start+10],raw[:7],phase,dataset.normalization['residual_actions']))
    return stack_observations(rows),np.stack(actions)

def decode_prediction(dataset,normalized,state,phase):
    return decode_chunk(normalized,np.asarray(state)[:7],phase,dataset.normalization['residual_actions'])

def core_sources():
    own=Path(__file__).parent;old=Path(parent.__file__).parent
    paths={**{'v7/'+name:own/name for name in ('training_data.py','action_encoding.py','model.py')},
           **{'v6/'+name:old/name for name in ('train_openpi.py','openpi_adapter.py','training_data.py','state_encoding.py',
               'action_contract.py','action_encoding.py','task_observation.py','dynamics_binding.py','dynamics_contract.py',
               'phase_contract.py','observed_supervisor.py','supervision.py','derive_task_dataset.py','evaluation.py')}}
    return {name:parent._sha(path) for name,path in paths.items()}
