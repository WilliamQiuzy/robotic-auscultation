"""Rebuild finite-head fits and low-elbow paths solely on current exported meshes.

This CPU experiment proposes geometry. It never creates accepted demonstrations
and never claims that a free approach plus loaded endpoint proves contact motion.
"""
import argparse
import hashlib
import json
import os
from pathlib import Path
import sys
import time
import numpy as np
from .geometry_epoch import contact_frame, require_geometry_artifacts


def write(path,value):
    Path(path).write_text(json.dumps(value,indent=2,allow_nan=False)+'\n')


def patient_provenance(config):
    """Bind varied-body targets to the actual patient installed in Isaac."""
    binding=config.get('patient_receipt')
    if binding is None:return None
    data=Path(binding['file']).read_bytes()
    if hashlib.sha256(data).hexdigest()!=binding['sha256']:
        raise ValueError('Installed patient receipt changed')
    receipt=json.loads(data)
    targets={row['name']:row for row in receipt['targets']}
    if (receipt.get('non_patient_scene_unchanged') is not True
            or any(row!=targets.get(row['name']) for row in config['targets'])):
        raise ValueError('Planning targets differ from the installed patient surface')
    return dict(binding=binding,identity=receipt['identity'],asset_sha256=receipt['asset_sha256'],
                identity_split_sha256=receipt['identity_split_sha256'])


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--config',type=Path,required=True)
    parser.add_argument('--output',type=Path,required=True)
    args=parser.parse_args();config=json.loads(args.config.read_text())
    if sys.platform!='linux' or os.environ.get('AUSC_OWNED_RUN')!='1':
        raise RuntimeError('Owned Linux experiment required')
    directory=Path(config['collision_directory'])
    hashes=require_geometry_artifacts(directory,config)
    source=Path(config['geometry_source'])
    if hashlib.sha256(source.read_bytes()).hexdigest()!=config['geometry_source_sha256']:
        raise ValueError('Geometry source changed')
    patient=patient_provenance(config)
    args.output.mkdir(parents=True,exist_ok=False)
    from .finite_head import FiniteHeadFit
    if config.get('fit_refinement_diagnostic') is not None:
        from .fit_refinement_run import run_fit_diagnostic
        rows=run_fit_diagnostic(FiniteHeadFit(directory,config),config['targets'],args.output,
                                config['fit_refinement_diagnostic'])
        require_geometry_artifacts(directory,config)
        if patient_provenance(config)!=patient:raise ValueError('Patient provenance changed during diagnostic')
        if hashlib.sha256(source.read_bytes()).hexdigest()!=config['geometry_source_sha256']:
            raise ValueError('Geometry source changed during diagnostic')
        write(args.output/'COMPLETE.json',dict(completed=True,geometry_id=config['geometry_id'],
            geometry_artifacts=hashes,patient_provenance=patient,targets=rows,
            physics_executed=False,training_eligible=False,
            scope='Bounded finite-head hypotheses and raw rejection evidence; no robot IK or physical qualification'))
        return
    from .contact_collision import ContactCollisionChecker
    from sim.auscultation_expert_layout_v1.kinematics import OpenArmKinematics
    from sim.auscultation_expert_layout_v1.posture import PostureLimits,PostureChecker
    from sim.auscultation_expert_layout_v1.plan_low_elbow import plan_target
    from sim.auscultation_expert_layout_v1.targets import Target,_base_transform
    kin=OpenArmKinematics(contact_frame=contact_frame)
    base=_base_transform(config['world_from_left_base'])
    limits=PostureLimits.from_dict(config['posture_limits'],kin)
    full=ContactCollisionChecker(kin,base,directory,config)
    checker=PostureChecker(full,base,limits)
    fit=FiniteHeadFit(directory,config);rows=[]
    if not checker.check(np.zeros(7))['clear']:
        raise ValueError('Current scene home is not clear')
    for target in config['targets']:
        start=time.monotonic();name=target['name']
        position=np.asarray(target['position_world_m']);normal=np.asarray(target['normal_world'])
        if name not in [s+str(i) for s in 'LR' for i in range(1,6)]:
            raise ValueError('Unknown retained target')
        candidates=fit.candidates(position,normal,compression=config['loaded_compression_m'])
        write(args.output/(name+'_fits.json'),candidates)
        print('FRESH_FITS',name,len(candidates['candidates']),flush=True)
        target_obj=Target(name,position,normal,base[:3,:3].T@(position-base[:3,3]),
                          base[:3,:3].T@normal,dict(geometry_artifacts=hashes))
        audit=dict(name=name,compression_m=config['loaded_compression_m'],candidates=[dict(
            normal_world=c['normal_world'],wrist_rotation_world=c['rotation_world'],
            gap_m=c['surface_gap_m'],angle_change_deg=c['normal_deviation_from_surface_deg'],
            gap_axis=c['gap_axis'],first_contact=c,
            finite_head_fit=c['fit']) for c in candidates['candidates']])
        plans=[]
        if audit['candidates']:
            for gap in config['fresh_pre_gap_search_m']:
                result=plan_target(target_obj,audit,base,kin,checker,limits,
                    max_candidates=len(audit['candidates']),seeds_per_candidate=config['seeds_per_candidate'],
                    max_seconds=config['per_gap_seconds'],max_routes=2,pre_gap=gap,
                    rng=np.random.default_rng(931+len(rows)))
                plans.append(result)
                write(args.output/(name+'_planning_attempts.json'),plans)
                if result['success']:break
                if (result.get('attempts') and
                        set(result.get('failure_counts',{}))=={'loaded_ik_or_posture'}):
                    # Pre-gap does not enter loaded IK. Repeating identical
                    # candidates/seeds at another pre-gap cannot change this.
                    break
        row=dict(name=name,fit_candidates=len(audit['candidates']),
                 geometry_path_found=any(p['success'] for p in plans),
                 elapsed_s=time.monotonic()-start,physics_executed=False,training_eligible=False)
        if row['geometry_path_found']:
            selected=min((p for p in plans if p['success']),key=lambda p:p['route_score'])
            selected.update(geometry_id=config['geometry_id'],geometry_source_sha256=config['geometry_source_sha256'],
                freshly_generated=True,geometry_artifacts=hashes,world_from_left_base=base.tolist(),
                gap_axis='command_normal',physics_executed=False,training_eligible=False)
            if patient is not None:selected['patient_provenance']=patient
            write(args.output/(name+'_fresh_plan.json'),selected)
        rows.append(row);write(args.output/'progress.json',rows)
        print('FRESH_PLAN',name,row,flush=True)
    require_geometry_artifacts(directory,config)
    if patient_provenance(config)!=patient:raise ValueError('Patient provenance changed during planning')
    write(args.output/'COMPLETE.json',dict(completed=True,geometry_id=config['geometry_id'],
        geometry_artifacts=hashes,targets=rows,physics_executed=False,training_eligible=False,
        scope='Fresh finite-head and sampled free paths; loaded approach and actual contact still require Isaac'))


if __name__=='__main__':main()
