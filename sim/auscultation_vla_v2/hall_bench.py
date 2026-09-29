"""Deterministic software sensor bench; not an Isaac or physical calibration run."""
import csv
from dataclasses import asdict, replace
import hashlib
import json
from pathlib import Path

import numpy as np

from .hall import HallSimulator, default_calibration


def run(output):
    output.mkdir(parents=True, exist_ok=True)
    cal=default_calibration()
    cases={
        'nominal': {},
        'noise_delay_dropout': dict(noise_std_v=.002,latency_s=.02,seed=20260925),
        'curve_mismatch': dict(curve_warp=.2),
    }
    rows=[]; summaries={}
    for name,settings in cases.items():
        sim=HallSimulator(cal,**settings)
        valid=stale=missing=0; same_path=True; errors=[]
        for index in range(801):
            t=index*.005
            # Slider signal prescribed at the sensor interface, not a physics simulation.
            x=float(np.interp(t,[0.,.5,1.5,2.5,3.5,4.],[0.,0.,.0025,.0025,0.,0.]))
            drop=name=='noise_delay_dropout' and 2.<=t<2.15
            sim.update(t,x,drop=drop)
            packet=sim.read(t)
            row=dict(case=name,time_s=t,synthetic_slider_input_mm=x*1000,
                     input_kind='synthetic_displacement_function_not_Isaac',status='NO_DATA')
            if packet is None:
                missing+=1
            else:
                row.update(adc_raw=packet.raw,sample_time_s=packet.sample_time_s,
                           received_time_s=packet.received_time_s,age_ms=(t-packet.sample_time_s)*1000)
                try:
                    observed=cal.decode(packet,now_s=t)
                    real=cal.decode(replace(packet,source='real_adc'),now_s=t)
                    same_path=same_path and observed==real
                    row.update(asdict(observed));valid+=1
                    source_x=float(np.interp(packet.sample_time_s,[0.,.5,1.5,2.5,3.5,4.],[0.,0.,.0025,.0025,0.,0.]))
                    errors.append(abs(observed.displacement_estimate_m-source_x))
                except ValueError as error:
                    row['status']='REJECTED';row['reason']=str(error)
                    stale+=int('stale' in str(error))
            rows.append(row)
        summaries[name]=dict(settings=settings,steps=801,valid=valid,missing=missing,stale_rejections=stale,
            identical_real_sim_decode=same_path,max_displacement_error_m=max(errors),
            accuracy_scope='software consistency with prescribed displacement; NOT measured sensor accuracy')
    assert all(v['identical_real_sim_decode'] for v in summaries.values())
    assert summaries['nominal']['stale_rejections']==0
    assert summaries['noise_delay_dropout']['stale_rejections']>0
    assert summaries['curve_mismatch']['max_displacement_error_m']>summaries['nominal']['max_displacement_error_m']
    fields=sorted({k for row in rows for k in row})
    with (output/'hall_response.csv').open('w',newline='') as stream:
        writer=csv.DictWriter(stream,fieldnames=fields);writer.writeheader();writer.writerows(rows)
    result=dict(schema='hall_sensor_domain_bench_v1',synthetic_only=True,isaac_executed=False,
        physical_calibration_validated=False,calibration_sha256=cal.sha256,calibrated_flag=cal.calibrated_flag,
        bench_verified=cal.bench_verified,cases=summaries,adc_model='12-bit quantization -> RP2040 u16 scaling -> mean of 8 reads',
        unmodelled=['ADC INL','magnetic hysteresis','temperature dependence','measured voltage noise spectrum'],
        actor_features=['hall_voltage_V','hall_spring_estimate_N'],truth_separated=True,
        source_hashes={str(p):hashlib.sha256(p.read_bytes()).hexdigest() for p in [Path(__file__),Path(__file__).with_name('hall.py'),
                      Path(__file__).with_name('observations.py')]})
    (output/'summary.json').write_text(json.dumps(result,indent=2)+'\n')
    # Standard scientific plot: no Isaac/3D rendering on Mac.
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    fig,axes=plt.subplots(3,1,figsize=(10,8),sharex=True,layout='constrained')
    for name in cases:
        selected=[r for r in rows if r['case']==name]
        times=[r['time_s'] for r in selected]
        axes[0].plot(times,[r.get('hall_voltage_v',np.nan) for r in selected],label=name)
        axes[1].plot(times,[r.get('spring_force_estimate_n',np.nan) for r in selected],label=name)
        axes[2].plot(times,[r.get('age_ms',np.nan) for r in selected],label=name)
    axes[0].set_ylabel('Hall voltage (V)')
    axes[1].set_ylabel('Spring-load estimate (N)')
    axes[2].set_ylabel('Sample age (ms)');axes[2].set_xlabel('Time (s)')
    axes[2].axhline(50,color='black',ls='--',label='50 ms software budget')
    for ax in axes:ax.grid(alpha=.25);ax.legend(fontsize=8,loc='best')
    fig.suptitle('Synthetic Hall interface bench | NOT physical calibration or patient force')
    fig.savefig(output/'hall_response.png',dpi=150);plt.close(fig)
    print(json.dumps(result,indent=2))


if __name__=='__main__':
    run(Path(__file__).with_name('results')/'hall_modality_v1')
