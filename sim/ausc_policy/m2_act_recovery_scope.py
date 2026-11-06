"""Metadata-only pretraining freeze for a fresh ACT repair evaluation cohort."""
import hashlib
from datetime import datetime

from . import m2_given_contact_runtime as rt, m2_rebuild_capture_dataset as rebuild

SCHEMA='wam_act_recovery_new_identity_cohort_v1'


def validate(ref):
    value=rt.read_bound(ref);need=rt.need
    need(value.get('schema')==SCHEMA and value.get('frozen_before_training') is True and
         value.get('model_outcomes_consulted') is False and type(value.get('seed')) is int,
         'Pretraining deterministic new identity cohort required')
    timestamp=datetime.fromisoformat(value['timestamp_utc'].replace('Z','+00:00'))
    need(timestamp.tzinfo is not None,'Timezone-bound pretraining freeze required')
    cohorts=value.get('cohorts',{})
    need(set(cohorts)=={'train','selection_validation','frozen_evaluation'} and
         cohorts['train']==list(rebuild.TRAIN_IDS) and cohorts['selection_validation']==[rebuild.SELECTION_ID],
         'Original development identity cohort required')
    excluded=value.get('excluded_prior_cohort_ids',[])
    need(isinstance(excluded,list) and set(rebuild.TRAIN_IDS)|{rebuild.SELECTION_ID}|set(rebuild.FROZEN_IDS)<=set(excluded),
         'All prior reconstructed identities must be excluded')
    split=rt.pose.bound_json(value['identity_split'])
    need(value['identity_split']['sha256']==rt.shared.contract.IDENTITY_SPLIT_SHA256,
         'Original frozen identity split required')
    quality=rt.pose.bound_json(value['identity_quality'])
    need(quality.get('identity_split_sha256')==rt.shared.contract.IDENTITY_SPLIT_SHA256,
         'Eligible identity inventory must bind frozen split')
    ids=cohorts['frozen_evaluation']
    need(isinstance(ids,list) and 2<=len(ids)<=8 and len(ids)==len(set(ids)),'Explicit two through eight new identities required')
    pool=[i for i in split['splits']['validation'] if i in quality['eligible_shape_betas'] and i not in excluded and
        i not in quality.get('quarantined_identities',[]) and i not in rt.shared.contract.QUARANTINED_IDENTITIES]
    selected=sorted(pool,key=lambda i:hashlib.sha256(f'{value["seed"]}:act-recovery-new-heldout:{i}'.encode()).digest())[:len(ids)]
    need(ids==selected,'Fresh cohort must follow frozen deterministic selection; old or outcome-selected IDs rejected')
    need(value.get('pose_ids')==list(rebuild.POSE_IDS) and isinstance(value.get('target_ids'),list) and
         len(value['target_ids'])==2 and len(set(value['target_ids']))==2 and
         set(value['target_ids'])<={s+str(i) for s in 'LR' for i in range(1,6)} and
         value.get('fixed_denominator')==len(ids)*len(value['pose_ids'])*len(value['target_ids']),
         'All predeclared identity/pose/target slots must remain in denominator')
    return cohorts
