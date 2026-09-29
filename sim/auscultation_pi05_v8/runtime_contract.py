"""Admitted V8 B policy plus unchanged physical Headroom execution."""
from pathlib import Path
from . import admission_bridge as B, policy_contract as P
from .train import source_hashes

MODULES = ('admission_bridge.py', 'runtime_contract.py', 'run_headroom_isaac.py', 'audit_actions.py', 'assess_trials.py')
RUNNER = 'sim.auscultation_pi05_v8.run_headroom_isaac'
CONTROLLER = 'sim.auscultation_pi05_v7.headroom_learned_controller.HeadroomLearnedController'
CLIENT = 'sim.auscultation_pi05_v8.policy_client.PolicyClient'
PARENT_AUDIT_SHA = '771a788d9041f12268631196504ddfdf7d5ca606542a80d259aa870a2ee552e2'
PARENT_ASSESSOR_SHA = 'd731ba715f713925086744588504d87589003ea29b7a6e588f91da1bdfad640e'


def package_path(generation):
    B.need(generation in {'v6', 'v7', 'v8'}, 'Known runtime generation required')
    return Path(__file__).resolve().parent.parent / ('auscultation_pi05_' + generation)


def training_source_paths():
    return {key: package_path(key.split('/')[0]) / key.split('/')[1] for key in source_hashes()}


def connection_source_paths():
    return {'v8/' + name: Path(__file__).with_name(name) for name in MODULES}


def runtime_connection_contract():
    from sim.auscultation_pi05_v7.headroom_execution import runtime_adaptation_contract
    B.need(B.file_sha(package_path('v7') / 'audit_headroom_learned_actions.py') == PARENT_AUDIT_SHA and
           B.file_sha(package_path('v7') / 'assess_headroom_trials.py') == PARENT_ASSESSOR_SHA,
           'Original physical audit/assessor ancestry changed')
    return dict(schema='wam_horizon_pair_headroom_connection_v1', scope='zero_home_full_single_or_continuous_v1',
        executed_runner_module=RUNNER, controller_class=CONTROLLER, client_class=CLIENT,
        connection_source_sha256={k: B.file_sha(p) for k, p in connection_source_paths().items()},
        training_source_sha256=source_hashes(), inference_source_sha256=P.inference_source_hashes(),
        headroom_implementation=runtime_adaptation_contract(), admission_bridge_contract=B.source_contract(),
        action_numerics_changed=False, raw_actions_clipped=False, expert_fallback=False,
        start_family_preparation_supported=False, physical_gates_unchanged=True,
        copied_headroom_auditor_sha256=PARENT_AUDIT_SHA, copied_headroom_assessor_sha256=PARENT_ASSESSOR_SHA,
        selection_policy='Both fresh14600 budgets and fixed locked candidates; complete fixed160 paired admission revalidated on CPU',
        acceptance='Positive B pair admission AND exact action/sensor replay AND every original physical gate; actual trials required',
        physical_accepted=False, generalization_validated=False)


selected_admission_with_receipt = B.selected_admission_with_receipt
selected_admission = B.selected_admission


def verify_policy_against_admission(binding, admission):
    B.need(admission.get('schema') == B.ADMISSION_SCHEMA and admission.get('pair_freshly_reconstructed') is True,
           'Fresh V8 runtime admission required')
    P.verify_against_admission(binding, admission['pair_admission'], admission['pair_admission_sha256'])
    for key in ('checkpoint_metadata_sha256', 'data_hash', 'training_source_sha256',
                'dynamics_recipe_sha256', 'dynamics_contract_sha256', 'execution_contract_sha256'):
        B.need(binding.get(key) == admission.get(key), 'Policy differs from runtime admission: ' + key)


def verify_connection_outcome(outcome, admission):
    contract = runtime_connection_contract()
    B.need(outcome.get('executed_runner_module') == RUNNER and outcome.get('executed_controller_class') == CONTROLLER,
           'Actual runner/controller differs')
    B.need(outcome.get('executed_runner_source_sha256') == contract['connection_source_sha256']['v8/run_headroom_isaac.py'] and
           outcome.get('executed_policy_client_class') == CLIENT, 'Actual runner/client source differs')
    B.need(outcome.get('runtime_connection_contract') == contract and
           outcome.get('runtime_connection_contract_sha256') == B.digest(contract), 'Runtime connection differs')
    B.need(outcome.get('pair_runtime_binding') == admission, 'Saved V8 admission differs')
    B.verify_receipt(outcome.get('admission_bridge_receipt', {}), admission)
    B.need(outcome.get('task_scope') == 'full' and outcome.get('initial_state_scope') == 'zero_home',
           'Only complete zero-home task qualifies')
    verify_policy_against_admission(outcome.get('learned_policy_binding', {}), admission)


def verify_artifacts(artifacts):
    for path, expected in artifacts.items():
        B.need(B.file_sha(path) == expected, 'Runtime input changed: ' + path)
