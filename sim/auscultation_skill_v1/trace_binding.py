"""Bind mixed 240 Hz contact and 20 Hz free blocks to exact physics evidence."""
import numpy as np
from sim.auscultation_expert_layout_v1.dataset import _array

DT = 1/240.


def validate_control_trace(controls, trace):
    if not isinstance(controls,list) or not controls or not all(isinstance(x,dict) for x in controls):
        raise ValueError('empty or malformed control episode')
    if not isinstance(trace,list) or not trace or not all(isinstance(x,dict) for x in trace):
        raise ValueError('empty or malformed physics trace')
    times = np.asarray([float(_array(r.get('time_s'),(),'physics timestamp')) for r in trace])
    if not np.isclose(times[0],DT,atol=1e-7,rtol=0) or not np.allclose(np.diff(times),DT,atol=1e-7,rtol=0):
        raise ValueError('physics timestamp spacing or origin mismatch')
    fields = (('q',('q',),(7,)), ('dq',('dq',),(7,)), ('hall_N',('hall_N',),()),
        ('compression_m',('compression_m',),()), ('tcp_base_m',('tcp_base_m','tcp_base'),(3,)),
        ('tcp_rotation_base',('tcp_rotation_base','rotation_base'),(3,3)))
    comparisons = {key:0 for key,_,_ in fields}; cursor=0; lengths={}
    for block in controls:
        start,end = block.get('physics_row'),block.get('physics_row_end_exclusive')
        if (type(start) is not int or type(end) is not int or start!=cursor
                or not 1<=end-start<=12 or end>len(trace)):
            raise ValueError('control coverage has a hole, overlap, or invalid bounds')
        count=end-start; cursor=end; lengths[str(count)]=lengths.get(str(count),0)+1
        if block.get('command_block_complete') is not True or block.get('executed_physics_steps')!=count:
            raise ValueError('incomplete command block')
        if not np.isclose(block.get('action_duration_s',0),count*DT,atol=1e-10,rtol=0):
            raise ValueError('action duration differs from executed rows')
        target=block.get('target')
        if not isinstance(target,str) or not target or any(r.get('target')!=target for r in trace[start:end]):
            raise ValueError('control target differs from physics target')
        if block.get('phase')!=trace[start].get('phase') or any(r.get('subphase')!=block.get('subphase') for r in trace[start:end]):
            raise ValueError('control stage differs from physics trace')
        command=_array(block.get('q_command'),(7,),'q_command')
        if not np.allclose(command,_array(trace[end-1].get('q_command'),(7,),'physics q_command'),atol=1e-8,rtol=0):
            raise ValueError('control command endpoint differs from physics trace')
        prior=_array(block.get('previous_q_command'),(7,),'previous_q_command')
        observation=block.get('observation')
        if not isinstance(observation,dict): raise ValueError('missing control observation')
        if start:
            preceding=trace[start-1]
            if not np.allclose(prior,_array(preceding.get('q_command'),(7,),'preceding command'),atol=1e-8,rtol=0):
                raise ValueError('prior command continuity differs from physics trace')
            expected_time=times[start-1]
            for key,aliases,shape in fields:
                actual_key=next((a for a in aliases if a in preceding),None)
                if actual_key is None: raise ValueError('missing authoritative state field: '+key)
                if not np.allclose(_array(observation.get(key),shape,'observation '+key),
                                   _array(preceding[actual_key],shape,'preceding '+key),atol=1e-7,rtol=0):
                    raise ValueError('control observation '+key+' differs from preceding physics state')
                comparisons[key]+=1
        else:
            if not np.allclose(prior,_array(observation.get('q'),(7,),'initial observed q'),atol=1e-8,rtol=0):
                raise ValueError('initial prior command differs from initial observed q')
            expected_time=0.
        if not np.isclose(float(_array(block.get('time_s'),(),'control timestamp')),expected_time,atol=1e-7,rtol=0):
            raise ValueError('control timestamp differs from preceding physics observation')
        action=_array(block.get('action_joint_delta_rad'),(7,),'action_joint_delta_rad')
        if not np.allclose(command-prior,action,atol=1e-8,rtol=0):
            raise ValueError('action differs from traced command increment')
        commands=np.asarray([r['q_command'] for r in trace[start:end]],float)
        if np.max(np.abs(np.diff(np.vstack([prior,commands]),axis=0)))>.25*DT+1e-8:
            raise ValueError('physics command speed exceeds common bound')
    if cursor!=len(trace): raise ValueError('uncovered physics rows')
    return dict(control_blocks=len(controls),physics_rows_covered=len(trace),
        block_length_histogram=lengths,state_field_comparisons=comparisons,
        unverified_state_fields=[k for k in comparisons if comparisons[k]<len(controls)-1 or len(controls)==1],
        initial_state_binding='initial observed q equals prior command; no independent pre-step physics sample',
        pose_binding='same-frame authoritative observations; conditioned actor inputs stored separately',
        duration_binding='each 1..12-step block checked against actual covered physics rows')
