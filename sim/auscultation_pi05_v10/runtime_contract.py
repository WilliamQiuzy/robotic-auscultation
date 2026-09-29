"""New V10 policy identity joined to the unchanged actual V7 Headroom physics."""
from pathlib import Path
from . import admission_bridge as B,policy_contract as P

MODULES=P.CONNECTION_MODULES
RUNNER='sim.auscultation_pi05_v10.run_headroom_isaac'
CONTROLLER='sim.auscultation_pi05_v7.headroom_learned_controller.HeadroomLearnedController'
CLIENT='sim.auscultation_pi05_v10.policy_client.PolicyClient'
PARENT_AUDIT_SHA='22e8f32c9c6d315e037573737d8b664417cd62e84d8ed04a89760c11f4ebe40a'


def package_path(generation):
    B.need(generation in {'v6','v7','v8','v10'},'Known physical source generation required')
    return Path(__file__).resolve().parent.parent/('auscultation_pi05_'+generation)


def training_source_paths(admission):
    return {k:B.root_path()/P.relative(k) for k in admission['training_source_sha256']}


def connection_source_paths():return P.connection_source_paths()


def runtime_connection_contract():
    from sim.auscultation_pi05_v7.headroom_execution import runtime_adaptation_contract
    bridge=B.source_contract()
    B.need(B.file_sha(package_path('v8')/'audit_actions.py')==PARENT_AUDIT_SHA,'Reviewed full physical replay ancestor changed')
    return dict(schema='wam_pi05_v10_recovery_headroom_connection_v1',scope='zero_home_full_single_or_continuous_v1',executed_runner_module=RUNNER,controller_class=CONTROLLER,client_class=CLIENT,
        connection_source_sha256=P.connection_source_hashes(),inference_source_sha256=P.inference_source_hashes(),training_inventory_binding=bridge['producer_inventory_binding'],runtime_inventory_binding=bridge['runtime_inventory_binding'],headroom_implementation=runtime_adaptation_contract(),admission_bridge_contract=bridge,
        action_numerics_changed=False,raw_actions_clipped=False,expert_fallback=False,start_family_preparation_supported=False,physical_gates_unchanged=True,
        copied_headroom_auditor_sha256=PARENT_AUDIT_SHA,selection_policy='V10 complete4000/R5 immutable selection plus selected fixed TRAIN probes and once-only L5, verified by original external scientific producer',acceptance='Positive V10 scientific admission AND exact action/sensor replay AND original physical gates; actual trials required',physical_accepted=False,generalization_validated=False)


selected_admission_with_receipt=B.selected_admission_with_receipt
selected_admission=B.selected_admission


def verify_policy_against_admission(binding,admission):
    B.need(admission.get('schema')==B.ADMISSION_SCHEMA and admission.get('scientific_admission_reconstructed') is True,'Fresh V10 scientific runtime admission required')
    P.verify_against_admission(binding,admission,admission['admission_binding']['sha256'])
    for k in ('checkpoint_metadata_sha256','data_hash','training_source_sha256',*P.DYNAMICS_FIELDS):B.need(P.same(binding.get(k),admission.get(k)),'Policy/runtime identity differs: '+k)


def verify_connection_outcome(outcome,admission):
    contract=runtime_connection_contract()
    B.need(outcome.get('executed_runner_module')==RUNNER and outcome.get('executed_controller_class')==CONTROLLER and outcome.get('executed_policy_client_class')==CLIENT,'Actual runner/controller/client differs')
    B.need(outcome.get('executed_runner_source_sha256')==contract['connection_source_sha256']['sim/auscultation_pi05_v10/run_headroom_isaac.py'],'Actual runner source differs')
    B.need(outcome.get('runtime_connection_contract')==contract and outcome.get('runtime_connection_contract_sha256')==B.digest(contract),'Runtime connection differs')
    B.need(outcome.get('recovery_runtime_binding')==admission,'Saved V10 runtime admission differs');B.verify_receipt(outcome.get('admission_bridge_receipt',{}),admission)
    B.need(outcome.get('task_scope')=='full' and outcome.get('initial_state_scope')=='zero_home','Only full zero-home scope qualifies')
    verify_policy_against_admission(outcome.get('learned_policy_binding',{}),admission)


def verify_artifacts(artifacts):
    B.add_artifacts({},artifacts)
