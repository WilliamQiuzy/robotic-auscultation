"""Explicit frozen, noisy simulation target provider for M2 deployment.

Config schema m2_frozen_compact_estimate_provider_v2 has compact_config and
noise_receipt {file,sha256} references plus integer revision. The first file is
the frozen pilot config; the second is its builder receipt. This reads only
the supplied compact estimate, never an analytic plan's target/normal. Noise
is the declared synthetic 4 mm position bias, not calibrated normal noise.

Target information is fixed before the episode. Only q/time are sampled at
each call; timestamps describe that feedback acquisition, not a fresh RGB-D
target detection. Phase latches at estimated uncompressed TCP distance <9cm,
and remains context metadata, outside the 17 numerical model inputs.
"""
from copy import deepcopy
import hashlib
import json
from pathlib import Path
import re
import numpy as np
from . import m2_contract as contract, m2_stage as stage

SCHEMA='m2_frozen_compact_estimate_provider_v2'
APPROACH_DISTANCE_M=stage.APPROACH_DISTANCE_M


def need(ok,message):
    if not ok:raise ValueError(message)


def _read(ref):
    need(isinstance(ref,dict) and set(ref)=={'file','sha256'} and
         isinstance(ref['sha256'],str) and re.fullmatch('[0-9a-f]{64}',ref['sha256']) is not None,
         'Explicit estimate artifact file/SHA256 required')
    path=Path(ref['file']).resolve()
    need(path.stat().st_size<=2_000_000,'Estimate metadata artifact exceeds bound')
    data=path.read_bytes()
    need(hashlib.sha256(data).hexdigest()==ref['sha256'],'Frozen estimate artifact hash changed')
    return json.loads(data),data,path


class FrozenEstimateProvider:
    def __init__(self,config,*,metadata,episode_id,target_id,patient):
        need(isinstance(config,dict) and set(config)=={'schema','compact_config','noise_receipt','revision'} and
             config['schema']==SCHEMA,'Explicit frozen compact estimate provider required')
        contract.require_observation_contract(metadata)
        need(metadata['target_source']=='gt_plus_measured_noise','Frozen simulation estimate must retain explicit noise source')
        need(type(config['revision']) is int and config['revision']>=0,'Invalid estimate revision')
        need(isinstance(episode_id,str) and 0<len(episode_id)<=128,'Deployment episode identity required')
        doc,compact_bytes,compact_path=_read(config['compact_config'])
        receipt,noise_bytes,noise_path=_read(config['noise_receipt'])
        try:
            binding=doc['fresh_expert']['raw_multimodal_capture']
            need(binding['profile']=='compact','Frozen source must declare compact estimate')
            settings=binding['compact'];meta=settings['metadata'];contract.require_observation_contract(meta)
            need(meta['split'] in ('train','validation','test') and
                 meta['subject_id'] not in contract.QUARANTINED_IDENTITIES and
                 meta['identity_split_sha256']==contract.IDENTITY_SPLIT_SHA256,
                 'Frozen TRAIN/VALIDATION/TEST identity split must be explicit')
            split_bytes=contract.IDENTITY_SPLIT_PATH.read_bytes()
            need(hashlib.sha256(split_bytes).hexdigest()==contract.IDENTITY_SPLIT_SHA256,
                 'Frozen deployment identity split changed')
            splits=json.loads(split_bytes)['splits']
            need([name for name,ids in splits.items() if meta['subject_id'] in ids]==[meta['split']],
                 'Frozen target estimate identity membership differs')
            need(meta['target_source']=='gt_plus_measured_noise' and
                 all(meta[key]==metadata[key] for key in ('geometry_id','geometry_source_sha256','contract_id')),
                 'Frozen estimate deployment metadata mismatch')
            need(meta['target_id']==target_id and isinstance(patient,dict) and
                 meta['subject_id']==patient['identity']==receipt['subject_id'] and
                 meta['split']==patient['split']==receipt['split'] and
                 meta['patient_asset_sha256']==patient['asset_sha256'] and
                 meta['installed_patient_receipt_sha256']==patient['receipt_sha256']==receipt['patient_receipt']['sha256'],
                 'Frozen estimate differs from authenticated installed patient/target')
            need(meta['source_kind']=='isaac_live_physics' and meta['replay'] is False,
                 'Frozen estimate must declare live non-replay physics provenance')
            # The compact source episode can differ from this deployment episode.
            # Every identity exposed to inference must describe this actual run;
            # optional caller metadata cannot contradict authenticated sources.
            identity={key:deepcopy(meta[key]) for key in (
                'subject_id','split','target_id','identity_split_sha256',
                'patient_asset_sha256','installed_patient_receipt_sha256','source_kind','replay')}
            identity['episode_id']=episode_id
            for key,value in identity.items():
                need(key not in metadata or (metadata[key]==value and
                     (key!='replay' or metadata[key] is False)),
                     'Deployment metadata contradicts authenticated identity: '+key)
            canonical_metadata=dict(deepcopy(metadata),**identity)
            pose=contract._pose(settings['estimated_target_base'],'frozen estimated target')
            confidence=contract._scalar(settings['target_confidence'],'frozen confidence',maximum=1.)
            need(confidence>=.8,'Frozen confidence below existing L2 threshold')
            source=settings['target_estimate_source_id']
            need(isinstance(source,str) and 0<len(source)<=256,'Explicit frozen target source ID required')
            noise=receipt['target_estimate_noise']
            need(noise['schema']=='m2_fixed_visible_position_bias_v1' and
                 type(noise['seed']) is int and 0<=noise['seed']<2**63 and noise['prng']=='numpy_PCG64' and
                 noise['magnitude_m']==.004 and noise['normal_noise_applied'] is False and
                 noise['orientation_noise_applied'] is False and noise['applied_to_model_input'] is True and
                 noise['fixed_per_episode'] is True and noise['empirically_calibrated_noise'] is False,
                 'Unsupported or falsely calibrated noise provenance')
            clean=contract._pose(noise['clean_target_base'],'frozen clean target')
            recorded=contract._pose(noise['estimated_target_base'],'noise estimated target')
            bias=contract._array(noise['position_bias_base_m'],(3,),'recorded position bias')
            direction=np.random.Generator(np.random.PCG64(noise['seed'])).normal(size=3)
            expected=.004*direction/np.linalg.norm(direction)
            need(np.allclose(bias,expected,atol=1e-15,rtol=0) and np.array_equal(pose,recorded) and
                 np.array_equal(pose[:3,:3],clean[:3,:3]) and
                 np.allclose(pose[:3,3],clean[:3,3]+bias,atol=1e-12,rtol=0),
                 'Frozen model input does not contain the declared position-only noise')
        except (KeyError,TypeError) as error:
            raise ValueError('Missing explicit compact estimate/noise/patient provenance') from error
        self.metadata=canonical_metadata;self.episode_id=episode_id;self.target_id=target_id
        self.revision=config['revision'];self.pose=pose.copy();self.pose.setflags(write=False)
        self.confidence=confidence;self.source_id=source;self.phase='FREE';self._stage=stage.CausalStage()
        self._last_time=None;self._last_q=None;self.calls=0;self.transitions=[]
        self.source_paths=[compact_path,noise_path,Path(stage.__file__)]
        self.expected_source_hashes={str(compact_path):config['compact_config']['sha256'],
                                     str(noise_path):config['noise_receipt']['sha256'],
                                     str(Path(stage.__file__).resolve()):hashlib.sha256(Path(stage.__file__).read_bytes()).hexdigest()}
        self.artifact_bytes={'m2_frozen_compact_config.json':compact_bytes,'m2_target_noise_receipt.json':noise_bytes}
        self._receipt=dict(schema=SCHEMA,configuration=deepcopy(config),episode_id=episode_id,target_id=target_id,
            source_metadata=deepcopy(meta),deployment_metadata=deepcopy(self.metadata),
            revision=self.revision,patient=deepcopy(patient),estimated_target_base=pose.tolist(),
            target_confidence=confidence,target_estimate_source_id=source,target_estimate_noise=deepcopy(noise),
            phase_transition_distance_m=APPROACH_DISTANCE_M,phase_rule_id=stage.RULE_ID,
            observation_contract=contract.observation_contract(),phase_is_state17_feature=False,
            target_acquisition='fixed before episode; only measured q/time are acquired per callback',
            teacher_plan_target_substitution=False,training_eligible=False,physics_executed=False)

    def __call__(self,controller,state):
        need(controller.episode_id==self.episode_id and controller.target.name==self.target_id,
             'Estimate provider episode/target changed')
        stamp=contract._scalar(state['time_s'],'current estimate feedback time')
        q=contract._array(state['q'],(7,),'current estimate feedback q')
        actual=controller.env.state()
        need(abs(contract._scalar(actual['time_s'],'actual feedback time')-stamp)<=1e-8 and
             np.array_equal(q,actual['q']),'Estimate callback differs from same-time actual feedback')
        need(self._last_time is None or stamp>=self._last_time-1e-8,'Estimate feedback clock moved backwards')
        if self._last_time is not None and abs(stamp-self._last_time)<=1e-8:
            need(np.array_equal(q,self._last_q),'Feedback q changed at unchanged physics time')
        state17=contract.encode_state(q,self.pose,self.confidence,metadata=self.metadata)
        distance=float(np.linalg.norm(state17[7:10]))
        previous=self.phase;self.phase=self._stage.update(distance,stamp)
        if previous!=self.phase:
            self.transitions.append(dict(time_s=stamp,phase=self.phase,estimated_tcp_distance_m=distance))
        self._last_time=stamp;self._last_q=q.copy();self.calls+=1
        return dict(episode_id=self.episode_id,target_id=self.target_id,revision=self.revision,phase=self.phase,
            observed_at_s=stamp,measured_q=q,estimated_target_base=self.pose.copy(),confidence=self.confidence,source_id=self.source_id)

    def snapshot(self):
        return dict(deepcopy(self._receipt),callback_count=self.calls,last_feedback_time_s=self._last_time,
                    phase_transitions=deepcopy(self.transitions))
