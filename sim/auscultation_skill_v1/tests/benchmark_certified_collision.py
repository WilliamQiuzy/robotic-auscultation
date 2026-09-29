import json,time,statistics,hashlib
from pathlib import Path
import numpy as np
from sim.auscultation_skill_v1.certified_collision import CertifiedCollisionChecker
from sim.auscultation_skill_v1.curve import FixedNormalCurve,force_derived_domain
from sim.curobo_expert_v1.fast_collision import FastCollisionChecker
from sim.curobo_expert_v1.static_world import build_static_world,install_fcl_static_world
from sim.auscultation_expert_layout_v1.kinematics import OpenArmKinematics
from sim.auscultation_expert_layout_v1.posture import PostureLimits,PostureChecker
ROOT=Path.cwd();PLANS=ROOT/'sim/upper_pair_posture_20260924/results/candidate_b_plans';GEO=ROOT/'sim/auscultation_expert_layout_v1/results/expert_v2/plans'
kin=OpenArmKinematics();bundle=build_static_world();cases=[];pairs={};source_paths=[];segments=[]
def setup(target):
 plan=json.loads((PLANS/f'planned_paths_{target}.json').read_text());point=plan['points'][0];base=np.array(plan['world_from_left_base']);limits=PostureLimits.from_dict(point['posture_limits'],kin)
 result={}
 for name,cls in [('original',FastCollisionChecker),('certified',CertifiedCollisionChecker)]:
  exact=cls(kin,base,GEO);install_fcl_static_world(exact,bundle);result[name]=(PostureChecker(exact,base,limits),exact)
 return result,plan,point,base,limits
for target,episode in [('L4','pilot_l4_v2'),('R1','qualify_v2/episodes/nominal_01')]:
 checkers,_,_,_,_=setup(target);pairs[target]=checkers
 path=ROOT/'sim/auscultation_skill_v1/results'/episode/'expert_trace.jsonl';source_paths.append(path);rows=[json.loads(line) for line in path.open()]
 groups=[('free',[i for i,r in enumerate(rows) if r['target']==target and r['subphase']=='transfer']),('contact_settle',[i for i,r in enumerate(rows) if r['target']==target and r['phase']=='CONTACT']),('hold',[i for i,r in enumerate(rows) if r['target']==target and r['phase']=='HOLD']),('unload_start',[i for i,r in enumerate(rows) if r['target']==target and r['subphase']=='normal_unload']),('unload_end',[i for i,r in enumerate(rows) if r['target']==target and r['subphase']=='normal_unload'])]
 for group,indices in groups:
  selected=indices[:48] if group=='unload_start' else indices[-48:]
  assert len(selected)==48 and np.all(np.diff(selected)==1)
  segments.append(dict(target=target,group=group,start_row=selected[0],end_row_exclusive=selected[-1]+1,physics_steps=48,source=str(path.relative_to(ROOT))))
  for index in selected:
   row=rows[index]
   for mode in ['command','observed']:
    cases.append(dict(target=target,group=group,row=index,mode=mode,q=np.array(row['q_command' if mode=='command' else 'q']),compression=rows[index-1]['compression_m'] if mode=='command' else row['compression_m'],contact=group!='free'))
assert len(cases)==960
expected=[];timings={'original':[],'certified':[]};cpu_timings={'original':[],'certified':[]};passes=[];differences=[]
implementation_path=ROOT/'sim/auscultation_skill_v1/certified_collision.py'
start_implementation_sha256=hashlib.sha256(implementation_path.read_bytes()).hexdigest()
for repeat in range(2):
 for label in (['original','certified'] if repeat==0 else ['certified','original']):
  if label=='certified':
   for checkers in pairs.values():
    checkers[label][1].invalidate_certificates();checkers[label][1].total_exact_distance_calls=0;checkers[label][1].total_certificate_reuses=0
  start=time.perf_counter(); cpu_start=time.process_time(); native=reused=0; group_s={};last=time.perf_counter()
  for index,c in enumerate(cases):
   checker,exact=pairs[c['target']][label];started=time.perf_counter()
   report=checker.check(c['q'],compression=max(0.,c['compression']),clearance=.0003 if c['contact'] else .003,allow_patient_contact=c['contact'],details=True)
   group=c['target']+'/'+c['group'];group_s[group]=group_s.get(group,0.)+time.perf_counter()-started
   if repeat==0 and label=='original':expected.append(report)
   elif report!=expected[index]:differences.append(dict(repeat=repeat,checker=label,index=index,original=expected[index],actual=report))
   assert exact.last_checked_aabb_pair_count==len(exact.pairs)
   if label=='certified':native+=exact.last_exact_distance_calls;reused+=exact.last_certificate_reuses
  elapsed=time.perf_counter()-start;cpu_elapsed=time.process_time()-cpu_start;timings[label].append(elapsed);cpu_timings[label].append(cpu_elapsed);record=dict(repeat=repeat,checker=label,seconds=elapsed,cpu_seconds=cpu_elapsed,exact_distance_calls=native if label=='certified' else None,certificate_reuses=reused if label=='certified' else None,group_seconds=group_s);passes.append(record);print('PASS',json.dumps(record),flush=True)
# Independently known rejected poses, with both reporting modes.
failures=[]
for target,reserve in [('L1',None),('L5',.00075)]:
 checkers,plan,p,base,limits=setup(target);normal=base[:3,:3].T@p['command_normal_world'];rotation=base[:3,:3].T@p['command_rotation_world'];target_p=(np.linalg.inv(base)@np.r_[p['position_world'],1])[:3]
 loaded=float((kin.tcp(p['loaded_q'])[:3,3]-target_p)@normal);center=loaded+p['loaded_compression_m'];minimum=loaded-.0015 if reserve is None else force_derived_domain(loaded,p['loaded_compression_m'],p['command_normal_world'],4.,reserve)['minimum_gap_m']
 curve=FixedNormalCurve.build(kin,base,limits,p['transfer_q'][-1],target_p,normal,rotation,minimum)
 gaps=[minimum,center-.0031] if target=='L1' else [minimum,minimum+.00006]
 for gap in gaps:
  for details in (False,True):
   q=curve(gap);opts=dict(clearance=.0003,compression=center-gap,allow_patient_contact=True,details=details)
   original=checkers['original'][0].check(q,**opts)
   for _ in range(2):actual=checkers['certified'][0].check(q,**opts);assert actual==original
   assert original['clear'] is False
   failures.append(dict(target=target,gap_m=gap,compression_m=center-gap,details=details,result=original))
medians={k:statistics.median(v) for k,v in timings.items()};speedup=medians['original']/medians['certified']
source_paths += [ROOT/'sim/auscultation_skill_v1/certified_collision.py',ROOT/'sim/auscultation_skill_v1/tests/test_certified_collision.py',GEO/'collision_geometry.json',GEO/'collision_geometry.npz',ROOT/'sim/curobo_expert_v1/static_world.py',ROOT/'sim/curobo_expert_v1/static_containment.py',ROOT/'sim/curobo_expert_v1/fast_collision.py']+[PLANS/f'planned_paths_{t}.json' for t in ('R1','L1','L4','L5')]
assert hashlib.sha256(implementation_path.read_bytes()).hexdigest()==start_implementation_sha256
r=dict(cpu_seconds=cpu_timings,cpu_speedup_median=statistics.median(cpu_timings['original'])/statistics.median(cpu_timings['certified']),implementation_file_unchanged_during_benchmark=True,schema='auscultation_optional_collision_certificate_benchmark_v1',scope='Mac CPU only, sequential saved physics rows; no simulated step, Linux job or runner integration. Guard is engineering margin, not formal floating-point error proof.',cases_per_pass=len(cases),segments=segments,timing_seconds=timings,median_seconds=medians,speedup_median=speedup,passes=passes,report_differences=differences,known_collisions=failures,all_original_pairs_aabb_checked_every_call=True,original_static_containment_installed=True,source_sha256={str(p.relative_to(ROOT)):hashlib.sha256(p.read_bytes()).hexdigest() for p in source_paths},script_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),eligible_for_further_review=not differences and speedup>1.)
Path('/tmp/certified_collision_benchmark.json').write_text(json.dumps(r,indent=2,allow_nan=False)+'\n');print('SUMMARY',json.dumps({k:r[k] for k in ('cases_per_pass','median_seconds','speedup_median','cpu_speedup_median','eligible_for_further_review')}),flush=True);print('MISMATCHES',len(differences),'KNOWN_COLLISIONS',len(failures),flush=True)
