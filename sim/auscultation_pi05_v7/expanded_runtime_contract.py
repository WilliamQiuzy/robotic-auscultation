"""Explicit zero-home connection for selected expanded pi05 + unchanged Headroom.

The 17/25 original source maps and original five Headroom modules remain
inherited implementations. Four new connection files bind the executed runner,
experiment admission and additional audits; no controller identity is forged.
"""
from pathlib import Path
from . import expanded_contract as X
from . import expanded_evaluation as E
from .headroom_execution import runtime_adaptation_contract
from .expanded_policy_client import verify_policy_binding,verify_runtime_sources

MODULES=('expanded_runtime_contract.py','run_expanded_headroom_isaac.py',
         'audit_expanded_headroom_actions.py','assess_expanded_headroom_trials.py')
RUNNER='sim.auscultation_pi05_v7.run_expanded_headroom_isaac'
CONTROLLER='sim.auscultation_pi05_v7.headroom_learned_controller.HeadroomLearnedController'
CLIENT='sim.auscultation_pi05_v7.expanded_policy_client.PolicyClient'
ADMISSION_SCHEMA='wam_v7_expanded_selected_runtime_admission_v1'
POLICY_FIELDS=('checkpoint_metadata_sha256','data_hash','experiment_binding','experiment_source_sha256',
               'dynamics_recipe_sha256','dynamics_contract_sha256','execution_contract_sha256')


def connection_source_paths():
    return {'v7/'+name:Path(__file__).with_name(name) for name in MODULES}


def runtime_connection_contract():
    return dict(schema='wam_v7_expanded_headroom_connection_v1',scope='zero_home_full_single_or_continuous_v1',
        executed_runner_module=RUNNER,controller_class=CONTROLLER,client_class=CLIENT,
        connection_source_sha256={k:E.sha(p) for k,p in connection_source_paths().items()},
        experiment_source_sha256=X.experiment_source_hashes(),headroom_implementation=runtime_adaptation_contract(),
        action_numerics_changed=False,raw_actions_clipped=False,expert_fallback=False,
        start_family_preparation_supported=False,physical_gates_unchanged=True,
        selection_policy='Fresh CPU reconstruction of locked feasible-first expanded validation selection before Kit',
        acceptance='Additional experiment/selection/source audit AND unchanged Headroom action/ADC/physical gates')


def selected_admission(lock_path,lock_sha,manifest,experiment):
    """Complete fresh CPU verification; called before SimulationApp construction."""
    from .expanded_selection import verify_selection_lock
    lock,context=verify_selection_lock(lock_path,lock_sha)
    b=context['binding'];selected=lock['selection']['selected'];config=context['config']
    E.require(str(Path(manifest).resolve(strict=True))==b['manifest_path'] and
        str(Path(experiment).resolve(strict=True))==b['experiment_path'],'Selected experiment paths differ')
    artifacts=dict(lock['selection']['artifact_sha256'])
    inputs=[Path(lock_path).resolve(),Path(lock['selection_path']),Path(b['run'])/'SELECTION_LOCKED.json',
        Path(manifest).resolve(),Path(experiment).resolve(),Path(b['run'])/'configuration.json',
        Path(b['run'])/'training_run.json',Path(b['run'])/'predeclared_schedules.json']
    # Admission is evidence for a selection among the whole declared run, not
    # just the winner. Keep every candidate's frozen bytes in the post-run check.
    inputs.extend(Path(checkpoint)/name for checkpoint,inventory in b['checkpoint_inventory'].items()
        for name in inventory)
    artifacts.update({str(p):E.sha(p) for p in inputs})
    return dict(schema=ADMISSION_SCHEMA,selection_lock_path=str(Path(lock_path).resolve()),selection_lock_sha256=lock_sha,
        manifest_path=b['manifest_path'],manifest_sha256=b['manifest_sha256'],
        experiment_path=b['experiment_path'],experiment_sha256=b['experiment_sha256'],
        checkpoint_metadata_sha256=selected['checkpoint_sha256']['metadata.json'],
        selected_checkpoint=selected['checkpoint'],selected_step=selected['step'],data_hash=b['data_hash'],
        experiment_binding=b['experiment_binding'],experiment_source_sha256=b['experiment_source_sha256'],
        dynamics_recipe_sha256=config['dynamics_recipe_sha256'],dynamics_contract_sha256=config['dynamics_contract_sha256'],
        execution_contract_sha256=config['execution_contract_sha256'],artifact_sha256=artifacts,
        selection_freshly_reconstructed=True,closed_loop_accepted=False)


def verify_policy_against_admission(binding,admission):
    E.require(admission.get('schema')==ADMISSION_SCHEMA,'Expanded selected admission required')
    verify_policy_binding(binding,admission['checkpoint_metadata_sha256'])
    for key in POLICY_FIELDS:E.require(binding.get(key)==admission.get(key),'Selected experiment/source policy differs: '+key)


def verify_connection_outcome(outcome,admission):
    contract=runtime_connection_contract()
    E.require(outcome.get('executed_runner_module')==RUNNER and outcome.get('executed_controller_class')==CONTROLLER,
        'Actual expanded runner/controller identity differs')
    E.require(outcome.get('executed_runner_source_sha256')==contract['connection_source_sha256']['v7/run_expanded_headroom_isaac.py'],
        'Actual expanded runner source differs')
    E.require(outcome.get('executed_policy_client_class')==CLIENT,'Actual expanded client identity differs')
    E.require(outcome.get('runtime_connection_contract')==contract,'Expanded connection source/contract differs')
    E.require(outcome.get('runtime_connection_contract_sha256')==E.json_sha(contract),'Connection contract SHA differs')
    E.require(outcome.get('experiment_runtime_binding')==admission,'Saved selected experiment admission differs')
    E.require(outcome.get('task_scope')=='full' and outcome.get('initial_state_scope')=='zero_home',
        'Start-family or component task cannot impersonate zero-home full execution')
    verify_policy_against_admission(outcome.get('learned_policy_binding',{}),admission)


def verify_artifacts(artifacts):
    for path,expected in artifacts.items():E.require(E.sha(path)==expected,'Bound runtime input changed: '+path)
