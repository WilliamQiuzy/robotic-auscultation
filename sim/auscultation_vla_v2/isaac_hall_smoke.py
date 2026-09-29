"""Isolated PhysX shadow sensor smoke check; hold the initial joint target.

No CAN/ROS/serial access. This does not validate contact or run a VLA policy.
"""
import argparse
import hashlib
import json
import os
from pathlib import Path
import sys
import time


def main():
    parser=argparse.ArgumentParser()
    parser.add_argument('--source-project',type=Path,required=True)
    parser.add_argument('--output',type=Path,required=True)
    args=parser.parse_args()
    args.output.mkdir(parents=True,exist_ok=True)
    sys.path.insert(0,str(Path(__file__).resolve().parents[2]))
    sys.path.append(str(args.source_project.resolve()))
    import numpy as np
    from sim.auscultation_vla_v2.hall import HallSimulator, default_calibration
    from isaacsim import SimulationApp
    app=SimulationApp({'headless':True,'fast_shutdown':False})
    result=dict(schema='isaac_hall_shadow_smoke_v1',isaac_executed=True,policy_executed=False,
        patient_contact_validated=False,robot_hardware_access=False,passed=False,close_returned=False,
        calibration_sha256=default_calibration().sha256,physics_steps=0,valid_hall_polls=0,
        sensor_input_rejections=0,errors=[])
    status=1
    try:
        from sim.auscultation_expert_layout_v1.env import IsaacContactEnv
        env=IsaacContactEnv(args.output/'scene')
        cal=default_calibration();sensor=HallSimulator(cal)
        hold=env.state()['q'].copy()
        compressions=[];volts=[];forces=[];sample_times=set()
        result['runtime_source_sha256']=hashlib.sha256((args.source_project/'sim/auscultation_expert_layout_v1/env.py').read_bytes()).hexdigest()
        for _ in range(480):
            env.step(hold,label='HALL_SHADOW_HOLD',target='none')
            state=env.state();t=float(state['time_s']);x=float(state['compression_m'])
            compressions.append(x);result['physics_steps']+=1
            try:
                sensor.update(t,x)
                packet=sensor.read(t)
                if packet is not None:
                    reading=cal.decode(packet,now_s=t)
                    volts.append(reading.hall_voltage_v);forces.append(reading.spring_force_estimate_n)
                    sample_times.add(packet.sample_time_s);result['valid_hall_polls']+=1
            except ValueError as error:
                result['sensor_input_rejections']+=1
                if len(result['errors'])<5:result['errors'].append(str(error))
        result.update(simulated_duration_s=t,distinct_adc_samples=len(sample_times),
            slider_range_m=[min(compressions),max(compressions)],
            hall_voltage_range_V=[min(volts),max(volts)] if volts else None,
            spring_estimate_range_N=[min(forces),max(forces)] if forces else None,
            commanded_joint_motion_rad=0.,
            maximum_actual_joint_change_rad=float(np.max(np.abs(env.state()['q']-hold))))
        result['sampling_schedule_passed']=abs(len(sample_times)-200)<=1
        result['hall_source_sha256']=hashlib.sha256(Path(__file__).with_name('hall.py').read_bytes()).hexdigest()
        result['passed']=(result['valid_hall_polls']>0 and result['sensor_input_rejections']==0
                          and result['sampling_schedule_passed'])
        status=0 if result['passed'] else 1
    except Exception as error:
        result['errors'].append(type(error).__name__+': '+str(error))
    finally:
        (args.output/'result.json').write_text(json.dumps(result,indent=2))
        app.close()
        result['close_returned']=True
        (args.output/'result.json').write_text(json.dumps(result,indent=2))
        print(json.dumps(result),flush=True)
    # This owns the isolated Kit worker; avoid known hangs in Python finalizers
    # only AFTER app.close returned and result was persisted. Parent checks exit.
    sys.stdout.flush();sys.stderr.flush();os._exit(status)


if __name__=='__main__':
    main()
