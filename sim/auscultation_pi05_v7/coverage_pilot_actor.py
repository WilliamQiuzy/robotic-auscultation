"""Selected, fully bound diagnostic coverage continuation actor; absolute output."""
from pathlib import Path
import time
import numpy as np
from sim.auscultation_pi05_v6 import train_openpi as parent
from sim.auscultation_pi05_v6.training_data import MODALITIES
from sim.auscultation_pi05_v6.task_observation import observation_binding
from .action_encoding import action_encoding_binding
from .training_data import observation,stack_observations,decode_prediction
from . import model as implementation
from . import coverage_pilot_train as training
from . import coverage_pilot_schedule as schedule_api


def load_actor(manifest,checkpoint,*,selection_lock,sampling_steps=10,seed=917):
    if sampling_steps!=10 or seed!=917:raise ValueError('Declared sampler remains seed917/10 steps')
    from .coverage_pilot_selection import verify_selection_lock
    lock=verify_selection_lock(Path(selection_lock))
    selected=lock['selection']['selected'];checkpoint=Path(checkpoint).resolve()
    if selected is None or checkpoint!=Path(selected['checkpoint_path']).resolve():raise ValueError('Selected diagnostic checkpoint required')
    metadata=parent._read_json(checkpoint/'metadata.json');config=metadata['config']
    dataset,old,schedule=training.load_inputs(manifest,config['bootstrap_checkpoint'])
    training.verify_configuration(dataset,config,schedule)
    if (metadata['step']<=0 or metadata['step'] not in schedule['candidate_steps']
        or metadata['optimizer_step']!=4000+metadata['step']):raise ValueError('Selected positive local step required')
    if training.checkpoint_binding(checkpoint)!=selected['checkpoint_binding']:raise ValueError('Selection checkpoint bytes changed')
    with np.load(checkpoint/'arrays.npz',allow_pickle=False) as arrays:training.verify_optimizer_age(arrays,metadata['leaf_contract'],metadata['step'])
    import jax
    from openpi.models import model as model_api
    runtime=implementation.initialize(61,5e-5)
    if runtime['assets']!=config['assets']:raise ValueError('Official public assets changed')
    implementation.restore(runtime,checkpoint,dataset.data_hash,config,dataset.normalization)
    sample=implementation.sampler(runtime,steps=10)
    def predict(rgb,state,phase,target):
        raw=np.asarray(state,np.float32)
        row=observation(rgb,raw,phase,target,runtime['tokenizer'],dataset.normalization)
        obs=model_api.Observation(**jax.device_put(stack_observations([row])));started=time.monotonic()
        normalized=np.asarray(sample(runtime['trainable'],runtime['frozen'],obs,jax.random.key(917)))
        if normalized.shape!=(1,10,7) or not np.isfinite(normalized).all():raise ValueError('Malformed residual output')
        absolute=decode_prediction(dataset,normalized[0],raw,phase)
        if absolute.shape!=(10,7) or not np.isfinite(absolute).all():raise ValueError('Malformed absolute output')
        return absolute,dict(inference_seconds=time.monotonic()-started,observation_anchor_q=raw[:7].tolist(),
          model_normalized_residual=normalized[0].tolist(),decoder='fixed observed measured q added once to sampled residual; no clipping')
    info=dict(checkpoint=str(checkpoint),checkpoint_step=metadata['step'],checkpoint_optimizer_step=metadata['optimizer_step'],
        checkpoint_metadata_sha256=parent._sha(checkpoint/'metadata.json'),data_hash=dataset.data_hash,parent_data_hash=dataset.parent.data_hash,
        pilot_binding=config['pilot_binding'],pilot_source_sha256=schedule_api.source_hashes(),pilot_selection_sha256=parent._sha(selection_lock),
        sampling_steps=10,sampling_seed=917,sampling_policy='same fixed common Gaussian seed at every receding-horizon query',
        modalities=MODALITIES,audio='absent',action_semantics='absolute_joint_command_v1',training_recipe=training.RECIPE,
        model_action_semantics='future_command_minus_fixed_observation_measured_q',observation_contract=observation_binding(),
        action_encoding_binding=action_encoding_binding(),dynamics_contract_sha256=dataset.dynamics['contract_sha256'],
        dynamics_recipe_sha256=dataset.dynamics['recipe_sha256'],execution_contract_sha256=dataset.dynamics['execution_contract_sha256'])
    return predict,info
