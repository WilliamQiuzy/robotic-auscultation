"""Acceptance gates on independently recorded PhysX contact impulses and poses."""
import argparse,csv,json,statistics,math
from pathlib import Path
p=argparse.ArgumentParser();p.add_argument('results',type=Path);a=p.parse_args()
raw=list(csv.DictReader((a.results/'contact_trace.csv').open()))
assert raw,'Empty physics trace'
numeric=[k for k in raw[0] if k not in ('phase','probe')]
rows=[{k:float(v) if k in numeric else v for k,v in r.items()} for r in raw]
assert all(math.isfinite(r[k]) for r in rows for k in numeric),'Non-finite physics trace'
def phase(name):
 rs=[r for r in rows if r['phase']==name]
 assert len(rs)>=120,f'Missing or short simulation phase: {name}'
 return rs[-120:]
def mean(rs,key):return statistics.mean(r[key] for r in rs)
free=phase('settle');central=phase('central_hold');broad=phase('broad_hold');stop=phase('central_stop_hold')
frame=phase('frame_stop_hold');post=phase('post_stop_hold')
return_names=['broad_return','central_return','central_stop_return','broad_stop_return','frame_stop_return','post_stop_return']
return_errors={name:max(abs(r['compression_mm']) for r in phase(name)) for name in return_names}
expected=mean(central,'spring_inferred_N')-mean(central,'gravity_axis_N')
measured=mean(central,'carriage_axial_N')
checks={
 'free_rest_within_0p1mm':max(abs(r['compression_mm']) for r in free)<=.1,
 'travel_within_0p2mm':max(r['compression_mm'] for r in rows)<=7.45 and min(r['compression_mm'] for r in rows)>=-.2,
 'attachment_closure_within_0p2mm':max(r['closure_mm'] for r in rows)<=.2,
 'each_release_returns_within_0p2mm':all(v<=.2 for v in return_errors.values()),
 'central_probe_touches_diaphragm':mean(central,'diaphragm_N')>1.,
 'central_probe_does_not_touch_clamp':mean(central,'clamp_N')<.05,
 'broad_pad_touches_clamp':mean(broad,'clamp_N')>1.,
 'broad_pad_does_not_touch_recessed_diaphragm':mean(broad,'diaphragm_N')<.05,
 'independent_axial_force_balance':abs(measured-expected)<=max(.5,.15*expected),
 'stop_reached':mean(stop,'compression_mm')>=7.05,
 'hard_stop_reaction_exceeds_spring_force':mean(stop,'total_contact_N')>mean(stop,'spring_inferred_N')+3.,
 'offset_probe_touches_frame':mean(frame,'frame_N')>5.,
 'offset_probe_touches_post':mean(post,'post_N')>5.,
 'frame_probe_does_not_touch_diaphragm':mean(frame,'diaphragm_N')<.05,
 'post_probe_does_not_touch_diaphragm':mean(post,'diaphragm_N')<.05,
}
report={'all_pass':all(checks.values()),'checks':checks,'physics_samples':len(rows),
 'central':{'compression_mm':mean(central,'compression_mm'),'measured_axial_contact_N':measured,'spring_plus_gravity_N':expected,'force_residual_N':measured-expected},
 'broad':{'compression_mm':mean(broad,'compression_mm'),'measured_contact_N':mean(broad,'total_contact_N'),'diaphragm_N':mean(broad,'diaphragm_N')},
 'hard_stop':{'compression_mm':mean(stop,'compression_mm'),'measured_contact_N':mean(stop,'total_contact_N'),'spring_inferred_N':mean(stop,'spring_inferred_N'),'note':'A hard stop limits travel, not external force. Test actuator force capped at 20 N.'},
 'exterior':{'frame_N':mean(frame,'frame_N'),'guide_head_N':mean(post,'post_N')},
 'max_attachment_error_mm':max(r['closure_mm'] for r in rows),
 'peak_contact_N':max(r['total_contact_N'] for r in rows),'max_compression_mm':max(r['compression_mm'] for r in rows),
 'return_max_abs_mm_by_cycle':return_errors,
 'force_channels':'Signed impulse on carriage (actor0 positive), summed then projected onto current compression axis; norm totals shown separately.'}
(a.results/'acceptance.json').write_text(json.dumps(report,indent=2))
print(json.dumps(report,indent=2));raise SystemExit(0 if report['all_pass'] else 1)
