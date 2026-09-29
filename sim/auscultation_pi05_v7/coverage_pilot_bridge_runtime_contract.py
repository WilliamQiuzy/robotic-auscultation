"""Truthful original-eight pilot + unchanged Headroom, zero-home/full only.

The original 17/25/5 source maps are ancestry. Seven pilot sources and these
four runtime sources are additional, independently bound executed code.
"""
from pathlib import Path
from . import coverage_pilot_admission_bridge as B
from . import coverage_pilot_runtime_contract as ORIGINAL
from . import coverage_pilot_evaluation as E
from . import coverage_pilot_schedule as S
from .headroom_execution import runtime_adaptation_contract
from .coverage_pilot_policy_client import verify_policy_binding,verify_runtime_sources

MODULES=('coverage_pilot_admission_bridge.py','coverage_pilot_bridge_runtime_contract.py','run_coverage_pilot_bridge_headroom_isaac.py',
         'audit_coverage_pilot_bridge_headroom_actions.py','assess_coverage_pilot_bridge_headroom_trials.py')
RUNNER='sim.auscultation_pi05_v7.run_coverage_pilot_bridge_headroom_isaac'
CONTROLLER='sim.auscultation_pi05_v7.headroom_learned_controller.HeadroomLearnedController'
CLIENT='sim.auscultation_pi05_v7.coverage_pilot_policy_client.PolicyClient'
ADMISSION_SCHEMA='wam_v7_coverage_pilot_selected_runtime_admission_v1'
POLICY_FIELDS=('checkpoint_metadata_sha256','data_hash','pilot_binding','pilot_source_sha256','pilot_selection_sha256',
               'dynamics_recipe_sha256','dynamics_contract_sha256','execution_contract_sha256')
PARENT_AUDIT_SHA='771a788d9041f12268631196504ddfdf7d5ca606542a80d259aa870a2ee552e2'
PARENT_ASSESSOR_SHA='d731ba715f713925086744588504d87589003ea29b7a6e588f91da1bdfad640e'


def connection_source_paths():
    return {'v7/'+name:Path(__file__).with_name(name) for name in MODULES}


def runtime_connection_contract():
    E.need(E.file_sha(Path(__file__).with_name('audit_headroom_learned_actions.py'))==PARENT_AUDIT_SHA and
           E.file_sha(Path(__file__).with_name('assess_headroom_trials.py'))==PARENT_ASSESSOR_SHA,'Copied audit/assessor ancestry changed')
    return dict(schema='wam_v7_coverage_pilot_bridge_headroom_connection_v1',scope='zero_home_full_single_or_continuous_v1',
        executed_runner_module=RUNNER,controller_class=CONTROLLER,client_class=CLIENT,
        connection_source_sha256={k:E.file_sha(p) for k,p in connection_source_paths().items()},
        pilot_source_sha256=S.source_hashes(),headroom_implementation=runtime_adaptation_contract(),
        original_connection_source_sha256={k:E.file_sha(v) for k,v in ORIGINAL.connection_source_paths().items()},
        admission_bridge_contract=B.source_contract(),
        action_numerics_changed=False,raw_actions_clipped=False,expert_fallback=False,
        start_family_preparation_supported=False,physical_gates_unchanged=True,
        copied_headroom_auditor_sha256=PARENT_AUDIT_SHA,copied_headroom_assessor_sha256=PARENT_ASSESSOR_SHA,
        audit_identity='New pilot recipe/schema/source/selection binding; complete copied physical/action replay remains required',
        selection_policy='Fresh CPU reconstruction of locked original-eight TRAIN fit after full14600; not heldout/generalization selection',
        acceptance='Pilot selection/source audit AND complete copied Headroom action/ADC/physical gates; actual trials still required')


def selected_admission_with_receipt(lock_path,lock_sha,manifest):
    """Original full selection algorithm executes in the bound CPU interpreter."""
    return B.selected_admission_with_receipt(lock_path,lock_sha,manifest)


def selected_admission(lock_path,lock_sha,manifest):
    return selected_admission_with_receipt(lock_path,lock_sha,manifest)[0]


def verify_policy_against_admission(binding,admission):
    E.need(admission.get('schema')==ADMISSION_SCHEMA,'New pilot selected admission required')
    verify_policy_binding(binding,admission['checkpoint_metadata_sha256'])
    for key in POLICY_FIELDS:E.need(binding.get(key)==admission.get(key),'Selected pilot/source policy differs: '+key)


def verify_connection_outcome(outcome,admission):
    contract=runtime_connection_contract()
    E.need(outcome.get('executed_runner_module')==RUNNER and outcome.get('executed_controller_class')==CONTROLLER,
        'Actual pilot runner/controller identity differs')
    E.need(outcome.get('executed_runner_source_sha256')==contract['connection_source_sha256']['v7/run_coverage_pilot_bridge_headroom_isaac.py'],
        'Actual pilot runner source differs')
    E.need(outcome.get('executed_policy_client_class')==CLIENT,'Actual pilot client identity differs')
    E.need(outcome.get('runtime_connection_contract')==contract,'Pilot connection source/contract differs')
    E.need(outcome.get('runtime_connection_contract_sha256')==E.digest(contract),'Connection contract SHA differs')
    E.need(outcome.get('pilot_runtime_binding')==admission,'Saved selected pilot admission differs')
    B.verify_receipt(outcome.get('admission_bridge_receipt',{}),admission)
    E.need(outcome.get('task_scope')=='full' and outcome.get('initial_state_scope')=='zero_home',
        'Start-family or component task cannot impersonate zero-home full execution')
    verify_policy_against_admission(outcome.get('learned_policy_binding',{}),admission)


def verify_artifacts(artifacts):
    for path,expected in artifacts.items():E.need(E.file_sha(path)==expected,'Bound runtime input changed: '+path)
