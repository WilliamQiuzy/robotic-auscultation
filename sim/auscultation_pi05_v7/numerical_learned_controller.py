"""New runtime identity with an additional certified numerical comparison.

The immutable V6 learner, observed supervisor, Ruckig and physical check chain
are reused. No exception is swallowed and no tick is retried.
"""
import hashlib
import json
from pathlib import Path
import numpy as np
from sim.auscultation_pi05_v6.learned_controller import LearnedController
from sim.auscultation_skill_v1.controller import ExpertController as SafetyController
from sim.auscultation_expert_layout_v1.controller import EpisodeRejected
from .numerical_command_gate import CertificateTracker,compare_command,numerical_gate_contract,contract_sha256

CLASS='sim.auscultation_pi05_v7.numerical_learned_controller.NumericalLearnedController'
SOURCE='pi05_absolute_joint_policy_with_certified_numerical_comparison'
LEDGER='numerical_gate_decisions.jsonl'
RECEIPT='numerical_gate_receipt.json'


class NumericalSafetyController(SafetyController):
    """Insert only the new comparison at the existing safety-controller layer."""
    def __init__(self,*args,**kwargs):
        self._numerical_tracker=CertificateTracker();self._numerical_buffer=[]
        self._numerical_event_count=0;self._numerical_interventions=0;self._numerical_compared=0
        self._numerical_ledger_initialized=False
        super().__init__(*args,**kwargs)

    def _flush_numerical(self):
        path=Path(self.env.out)/LEDGER
        if self._numerical_buffer or not self._numerical_ledger_initialized:
            with path.open('a' if self._numerical_ledger_initialized else 'x') as stream:
                for row in self._numerical_buffer:stream.write(json.dumps(row,allow_nan=False)+'\n')
            self._numerical_ledger_initialized=True;self._numerical_buffer.clear()

    def numerical_receipt(self):
        self._flush_numerical();path=Path(self.env.out)/LEDGER
        receipt=dict(schema='wam_numerical_command_gate_receipt_v1',controller_class=CLASS,
            numerical_gate_contract=numerical_gate_contract(),numerical_gate_contract_sha256=contract_sha256(),
            controller_source_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
            ledger_path=LEDGER,ledger_sha256=hashlib.sha256(path.read_bytes()).hexdigest(),
            compared_ticks=self._numerical_compared,raw_rejected_ticks=self._numerical_event_count,
            numerical_interventions=self._numerical_interventions,buffer_limit=256,
            event_scope='Every raw whole-stream comparison failure, including rejected unexecuted commands; successful unchanged raw comparisons are independently reconstructed',
            flushed_on_run_exit=True,nominal_jerk_rad_s3=8.,trajectory_changed=False)
        (Path(self.env.out)/RECEIPT).write_text(json.dumps(receipt,indent=2,allow_nan=False)+'\n')
        return receipt

    def _tick(self,command,subphase,active=False,recovery=False):
        # The surrounding operations match the immutable SafetyController._tick.
        # ObservedContactSupervisorMixin calls this layer with active=False, as
        # before, then performs its measured ADC/FK supervisor update afterward.
        state=self.env.state();event=None
        self._numerical_tracker.observe(getattr(self,'trajectory',None),command,len(self.env.rows),subphase)
        if len(self.env.rows)>=3:
            history=np.asarray([row['q_command'] for row in self.env.rows[-3:]],float)
            bounds=dict(velocity=self.joint_speed_limit,acceleration=.5 if self._contact_active else 1.)
            comparison=compare_command(history,command,**bounds)
            certificate=None
            if not comparison['raw_passed']:
                certificate=self._numerical_tracker.certificate()
                comparison=compare_command(history,command,certificate=certificate,**bounds)
                event=dict(index=self._numerical_event_count,physics_row=len(self.env.rows),
                    time_s=float(state['time_s']),subphase=subphase,active=bool(active),recovery=bool(recovery),contact_active=bool(self._contact_active),
                    history=history.tolist(),command=np.asarray(command,float).tolist(),certificate=certificate,
                    comparison=comparison,physics_executed=False)
                self._numerical_event_count+=1;self._numerical_interventions+=int(comparison['intervened'])
            self._numerical_compared+=1
            if not comparison['passed']:
                self._numerical_buffer.append(event)
                if len(self._numerical_buffer)>=256:self._flush_numerical()
                raise EpisodeRejected('whole-stream joint derivative gate: '+str(comparison['raw_peaks'])+'; numerical comparison: '+comparison['reason'])
        try:
            self._check_collision(command,state['compression_m'],self._contact_active)
            self.command_collision_samples+=1
            # Same parent target as super()._tick in the original SafetyController,
            # without calling that original comparison a second time.
            state=super(SafetyController,self)._tick(command,subphase,active=active,recovery=recovery)
            self._check_collision(state['q'],state['compression_m'],self._contact_active)
            self.observed_collision_samples+=1
            return state
        finally:
            if event is not None:
                event['physics_executed']=len(self.env.rows)>event['physics_row']
                self._numerical_buffer.append(event)
                if len(self._numerical_buffer)>=256:self._flush_numerical()


class NumericalLearnedController(LearnedController,NumericalSafetyController):
    def run(self,points):
        try:
            result=super().run(points)
        finally:
            receipt=self.numerical_receipt()
        result.update(executed_controller_class=CLASS,controller_source=SOURCE,
            runtime_controller_generation='pi05_v7_certified_numerical_comparison_over_pi05_v6',
            numerical_gate_contract=numerical_gate_contract(),numerical_gate_contract_sha256=contract_sha256(),
            numerical_gate_receipt=dict(path=RECEIPT,sha256=hashlib.sha256((Path(self.env.out)/RECEIPT).read_bytes()).hexdigest()),
            numerical_interventions=receipt['numerical_interventions'],original_failed_case_reclassified=False)
        return result
