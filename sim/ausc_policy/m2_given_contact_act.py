"""Distinct prescribed-pose ACT admission using the existing numerical backbone.

Sample/statistics wrappers retain the given-task contracts. The explicit
formal factory delegates only to the immutable simulation permit and owned
CPU bridge; no qualification is performed by this adapter. The injected
predictor path remains unqualified and separate.
"""
from . import m2_act as shared
from .m2_given_contact_pose import TASK_ID, require_route, observation_contract, need
from .m2_given_contact_intent import ACTION_SEMANTICS_ID, validate_native_intents, raw_cell

DATASET_SCHEMA = 'm2_given_contact_act_dataset_v2'
NORMALIZATION_SCHEMA = 'm2_given_contact_act_normalization_v2'
CHECKPOINT_SCHEMA = 'm2_given_contact_act_state17_history2_raw_target_h20_e10_v2'
DEPLOYMENT_SCHEMA = 'm2_given_contact_act_deployment_v2'
ADMISSION_SCHEMA = 'm2_given_contact_act_admission_v2'
PROTOCOL_SCHEMA = 'm2_given_contact_act_protocol_v2'


def assemble_samples(records, native_audit, metadata):
    from .m2_given_contact_pose import NATIVE_SOURCE
    ref = require_route(metadata)
    records = list(records)
    validate_native_intents(native_audit, metadata)
    for row in records:
        need(shared._scalar(row.get('target_estimate_source_id')) == 'given-contact-'+ref.sha256,
             'Given-reference observation source mismatch')
    import numpy as np
    for i,row in enumerate(records):
        if 'q_command' not in row: continue
        endpoint = 12*(i+2)-1
        need(endpoint < len(native_audit) and np.array_equal(
            np.asarray(row['q_command']),np.asarray(native_audit[endpoint]['q_command'])),
            'Compact applied endpoint differs from actual native command audit')
    result = shared.assemble_samples(records, native_audit, metadata, phase_source=NATIVE_SOURCE, route=TASK_ID)
    keep = []; actions = []
    for i, window in enumerate(result['derived_index']['windows']):
        first = window['inclusive_native_ticks'][0]+1
        owner_span=native_audit[window['history_native_ticks'][0]:window['inclusive_native_ticks'][1]+1]
        # Shared L2 provenance is insufficient: teacher/student ownership must
        # remain the same over BOTH real history frames and all future cells.
        actual_owners={event['actual_control_owner'] for event in owner_span}
        raw_owners={event['raw_intent']['setter']['control_owner']
                    for event in owner_span if event.get('raw_intent') is not None}
        if len(actual_owners)!=1 or actual_owners!=raw_owners: continue
        cells = [raw_cell(native_audit, first+k*12, metadata) for k in range(20)]
        if any(value is None for value in cells): continue
        keep.append(i); actions.append(np.asarray(cells)-result['origins'][i])
        window.update(action_semantics_id=ACTION_SEMANTICS_ID,
            actual_control_owner=next(iter(actual_owners)),
            raw_intent_native_ticks=[first,first+239], future_label_times='cell end t+(k+1)*.05')
    for key in shared.ARRAY_KEYS: result[key] = result[key][keep]
    result['actions'] = np.asarray(actions,dtype=float).reshape((-1,20,7))
    result['derived_index']['windows'] = [result['derived_index']['windows'][i] for i in keep]
    result.update(schema=DATASET_SCHEMA, task_id=TASK_ID, observation_contract=observation_contract(),
                  reference_sha256=ref.sha256, action_semantics_id=ACTION_SEMANTICS_ID,
                  metric_space='raw target-space MAE; not measured tracking')
    return result


def require_samples(samples):
    need(samples.get('schema') == DATASET_SCHEMA and samples.get('task_id') == TASK_ID and
         samples.get('observation_contract') == observation_contract() and
         samples.get('action_semantics_id') == ACTION_SEMANTICS_ID, 'Given-contact dataset route/action semantics mismatch')


def fit_normalization(samples, *, split):
    require_samples(samples)
    result = shared.fit_normalization(samples, split=split, observation_schema=observation_contract())
    result.update(schema=NORMALIZATION_SCHEMA, task_id=TASK_ID, action_semantics_id=ACTION_SEMANTICS_ID)
    return result


def pack_batch(samples, stats, indices, **kwargs):
    require_samples(samples)
    need(stats.get('schema') == NORMALIZATION_SCHEMA and stats.get('task_id') == TASK_ID and
         stats.get('action_semantics_id') == ACTION_SEMANTICS_ID,
         'Given-contact normalization route mismatch')
    return shared.pack_batch(samples, stats, indices, observation_schema=observation_contract(), **kwargs)


class ChunkPredictor(shared.ChunkPredictor):
    def __init__(self, torch, policy, normalization, metadata, device='cpu'):
        require_route(metadata)
        need(normalization.get('schema') == NORMALIZATION_SCHEMA and normalization.get('task_id') == TASK_ID and
             normalization.get('action_semantics_id') == ACTION_SEMANTICS_ID,
             'Given-contact checkpoint normalization required')
        super().__init__(torch, policy, normalization, metadata, device,
                         observation_schema=observation_contract(), checkpoint_schema=CHECKPOINT_SCHEMA)


def validate_checkpoint(checkpoint, *, allow_cpu_fixture=False):
    need(checkpoint.get('action_semantics_id') == ACTION_SEMANTICS_ID, 'Checkpoint raw action semantics required')
    require_route(checkpoint.get('metadata', {}))
    shared.validate_checkpoint(checkpoint, allow_cpu_fixture=allow_cpu_fixture,
        expected_schema=CHECKPOINT_SCHEMA, observation_schema=observation_contract())


def make_injected_controller(*args, predictor, metadata, episode_id, estimate_provider, policy_receipt, **kwargs):
    from .m2_given_contact_controller import GivenContactACT
    require_route(metadata)
    need(policy_receipt.get('schema') == DEPLOYMENT_SCHEMA and
         policy_receipt.get('action_semantics_id') == ACTION_SEMANTICS_ID and
         policy_receipt.get('task_id') == TASK_ID and policy_receipt.get('production_ready') is False,
         'Explicit unqualified injected-predictor receipt required')
    return GivenContactACT(*args, predictor=predictor, metadata=metadata, episode_id=episode_id,
        estimate_provider=estimate_provider, policy_receipt=policy_receipt, **kwargs)


def make_controller(*args, deployment, **kwargs):
    if not (isinstance(deployment,dict) and deployment.get('schema')==DEPLOYMENT_SCHEMA and
            deployment.get('production_ready') is True and deployment.get('simulation_only') is True and
            'permit' in deployment):
        raise RuntimeError('Given-contact formal CPU bridge qualification permit required')
    from .m2_given_contact_deployment import make_controller as formal_factory
    return formal_factory(*args,deployment=deployment,**kwargs)


def load_checkpoint(model, runtime_config):
    """Explicit formal admitted route; native execution uses its owned CPU child."""
    from .m2_given_contact_deployment import preflight
    from . import m2_given_contact_runtime as runtime
    binding=preflight(runtime_config)
    need(model==binding['permit']['model'], 'Only the exact admitted selected formal model may load')
    return runtime.load_checkpoint(model,device='cpu')
