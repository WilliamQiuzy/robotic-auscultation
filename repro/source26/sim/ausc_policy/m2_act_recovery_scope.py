"""Metadata-only pretraining freeze for a fresh ACT repair evaluation cohort."""
import hashlib
from datetime import datetime

from . import m2_given_contact_runtime as rt, m2_rebuild_capture_dataset as rebuild

SCHEMA='wam_act_recovery_new_identity_cohort_v1'
TRAIN16_SCHEMA='wam_act_recovery_train16_scope_v2'
TRAIN16_IDS=('p001120','p000758','p001031','p000833','p000961','p000742','p000513','p001182',
    'p001109','p000835','p000847','p000878','p000810','p000451','p000595','p000908')
FRESH8_IDS=('p000519','p000843','p000893','p000455','p000864','p000540','p000570','p000891')
PARENT_COHORT_SHA256='f7ff36152de8b987596880512621c5ce8c9ec149eae20b07dca71d4f613a4ff8'


def validate_train16_metadata(value,*,actual_train_ids=None):
    """Identity-only gate; old development outcomes may inform this repair.

    model_outcomes_consulted describes cohort selection, not the already closed
    development diagnoses. The fresh evaluation payloads/outcomes stay opaque.
    """
    need=rt.need;cohorts=value.get('cohorts',{})
    need(value.get('schema')==TRAIN16_SCHEMA and value.get('frozen_before_training') is True and
         value.get('model_outcomes_consulted') is False and value.get('fresh_payloads_opened') is False and
         value.get('development_outcomes_used_for_repair') is True and value.get('fresh_evaluation_outcomes_consulted') is False,
         'Explicit pretraining TRAIN16 scope without fresh payloads/outcomes required')
    need(cohorts==dict(train=list(TRAIN16_IDS),selection_validation=[rebuild.SELECTION_ID],
         frozen_evaluation=list(FRESH8_IDS)),'Exact TRAIN16 pool and unchanged opaque fresh8 cohort required')
    actual=value.get('optimized_train_ids');unavailable=value.get('unavailable_train_ids')
    need(isinstance(actual,list) and actual and actual==sorted(set(actual)) and set(actual)<=set(TRAIN16_IDS) and
         isinstance(unavailable,list) and unavailable==sorted(set(TRAIN16_IDS)-set(actual)),
         'Truthful optimized TRAIN identities and explicit unavailable pool identities required')
    need(not (set(actual)&({rebuild.SELECTION_ID}|set(FRESH8_IDS))),
         'Optimization, selection and fresh evaluation identities overlap')
    if actual_train_ids is not None:
        need(set(actual)==set(actual_train_ids),'Declared optimized identities differ from actual contributing TRAIN records')
    for subject in TRAIN16_IDS:rt.shared.require_membership('train',subject)
    for subject in (rebuild.SELECTION_ID,*FRESH8_IDS):rt.shared.require_membership('validation',subject)
    parent=value.get('parent_cohort',{})
    need(parent.get('sha256')==PARENT_COHORT_SHA256,'Original pretraining parent cohort SHA256 required')
    timestamp=datetime.fromisoformat(value['timestamp_utc'].replace('Z','+00:00'))
    need(timestamp.tzinfo is not None,'Timezone-bound pretraining freeze required')
    need(value.get('pose_ids')==list(rebuild.POSE_IDS) and value.get('target_ids')==['R3','L3'] and
         value.get('fixed_denominator')==32,'Unchanged two poses, R3/L3 and fresh32 denominator required')
    return cohorts


def validate(ref):
    value=rt.read_bound(ref)
    if value.get('schema')==TRAIN16_SCHEMA:
        cohorts=validate_train16_metadata(value)
        parent=value['parent_cohort'];rt._reference_shape(parent)
        old=_validate_legacy(parent)
        frozen=rt.read_bound(parent)
        rt.need(old['frozen_evaluation']==cohorts['frozen_evaluation'] and
            frozen['pose_ids']==value['pose_ids'] and frozen['target_ids']==value['target_ids'],
            'TRAIN16 scope must reuse the exact metadata-only parent evaluation cohort')
        return cohorts
    return _validate_legacy(ref)


def _validate_legacy(ref):
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
