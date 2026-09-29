"""Verified V7 pi05 actor; residual flow output decodes to absolute commands.

The simulator receives absolute joint targets and must never add measured q
again. The fixed observation anchor is used exactly once here, matching labels.
"""
from pathlib import Path
import time
import numpy as np
from sim.auscultation_pi05_v6 import train_openpi as parent
from sim.auscultation_pi05_v6.training_data import MODALITIES
from sim.auscultation_pi05_v6.task_observation import observation_binding
from .expanded_training_data import load_dataset,observation,stack_observations,decode_prediction,core_sources,RECIPE
from .expanded_contract import training_plan,verify_experiment_binding,verify_source_map
from .action_encoding import action_encoding_binding
from . import model as implementation

def verify_configuration(dataset,config):
    if not isinstance(config,dict) or 'experiment_binding' not in config:
        raise ValueError('Missing expanded experiment checkpoint binding')
    recorded=config.get('training_run_path')
    if not isinstance(recorded,str) or not Path(recorded).is_absolute() or str(Path(recorded).resolve())!=recorded:
        raise ValueError('Expanded checkpoint lacks canonical original training run path')
    verify_experiment_binding(config['experiment_binding'])
    verify_source_map(config.get('experiment_source_sha256'),local=True)
    plan=training_plan()
    for key in ('seed','batch_size','learning_rate','total_steps','save_every','evaluate_every'):
        if config.get(key)!=plan[key]:raise ValueError('Expanded predeclared plan changed: '+key)
    for key,expected in dict(recipe=RECIPE,
        experiment_binding=dataset.experiment_binding,experiment_source_sha256=dataset.experiment_source_sha256,action_dim=7,action_horizon=10,state_width=41,max_token_len=256,
        source_sha256=core_sources(),openpi_commit=parent.OPENPI_COMMIT,
        parent_data_hash=dataset.parent.data_hash,
        action_semantics='absolute_joint_command_v1',
        model_action_semantics='future_command_minus_fixed_observation_measured_q',
        canonical_phase_instructions=dict(parent.INSTRUCTIONS),instruction_policy='canonical-phase-and-target-v1',
        observation_contract=observation_binding(),action_encoding_binding=action_encoding_binding(),
        dynamics_contract_sha256=dataset.dynamics['contract_sha256'],
        dynamics_recipe_sha256=dataset.dynamics['recipe_sha256'],
        execution_contract_sha256=dataset.dynamics['execution_contract_sha256']).items():
        if config.get(key)!=expected:raise ValueError('V7 checkpoint binding mismatch: '+key)
    if config.get('trainer_sha256')!=parent._sha(Path(__file__).with_name('expanded_train_openpi.py')):
        raise ValueError('Expanded training source changed')

def load_actor(manifest,checkpoint,*,experiment,sampling_steps=10,seed=917):
    import jax
    from openpi.models import model as model_api
    if sampling_steps!=10 or seed!=917:raise ValueError('Declared deployment sampler is 10 steps, seed917')
    checkpoint=Path(checkpoint)
    dataset=load_dataset(manifest,experiment)
    metadata=parent._read_json(checkpoint/'metadata.json');config=metadata['config']
    verify_configuration(dataset,config)
    if metadata['step'] not in training_plan()['candidate_steps']:raise ValueError('Declared completed candidate step required')
    runtime=implementation.initialize(config['seed'],config['learning_rate'])
    if runtime['assets']!=config['assets']:raise ValueError('Public assets differ')
    implementation.restore(runtime,checkpoint,dataset.data_hash,config,dataset.normalization)
    sample=implementation.sampler(runtime,steps=sampling_steps)
    def predict(rgb,state,phase,target):
        raw=np.asarray(state,np.float32)
        row=observation(rgb,raw,phase,target,runtime['tokenizer'],dataset.normalization)
        obs=model_api.Observation(**jax.device_put(stack_observations([row])))
        started=time.monotonic()
        normalized=np.asarray(sample(runtime['trainable'],runtime['frozen'],obs,jax.random.key(seed)))
        if normalized.shape!=(1,10,7) or not np.isfinite(normalized).all():raise ValueError('Invalid sampled residual tensor')
        absolute=decode_prediction(dataset,normalized[0],raw,phase)
        if absolute.shape!=(10,7) or not np.isfinite(absolute).all():raise ValueError('Invalid decoded absolute commands')
        return absolute,dict(inference_seconds=time.monotonic()-started,
            decoder='fixed observed measured q added once to sampled residual; no clipping',
            observation_anchor_q=raw[:7].tolist(),model_normalized_residual=normalized[0].tolist())
    info=dict(checkpoint=str(checkpoint),checkpoint_step=metadata['step'],
        checkpoint_metadata_sha256=parent._sha(checkpoint/'metadata.json'),data_hash=dataset.data_hash,
        experiment_binding=dataset.experiment_binding,experiment_source_sha256=dataset.experiment_source_sha256,
        parent_data_hash=dataset.parent.data_hash,sampling_steps=sampling_steps,sampling_seed=seed,
        sampling_policy='same fixed common Gaussian seed at every receding-horizon query',
        modalities=MODALITIES,audio='absent',action_semantics='absolute_joint_command_v1',training_recipe=RECIPE,
        model_action_semantics=config['model_action_semantics'],
        observation_contract=observation_binding(),action_encoding_binding=action_encoding_binding(),
        dynamics_contract_sha256=dataset.dynamics['contract_sha256'],dynamics_recipe_sha256=dataset.dynamics['recipe_sha256'],
        execution_contract_sha256=dataset.dynamics['execution_contract_sha256'])
    return predict,info
