"""Evidence binding for revised contact and independent planner candidates."""
import json
from pathlib import Path
import numpy as np

from sim.expert_quality_20260924.quality import sha256_file, verify_hashes

ROOT=Path(__file__).resolve().parent
CLASS='sim.curobo_expert_v1.controller.ExpertController'
POLICIES=('revised_shortcut','curobo')
REQUIRED_SOURCES=frozenset(('controller.py','planner_service.py','run_expert.py',
    'provenance.py','withdrawal.py','curobo_backend.py','robot_bridge.py','timing.py','static_world.py','fast_collision.py','static_containment.py',
    'inherited/candidate_controller.py','inherited/candidate_planner.py'))


def implementation_sources(bridge_dir=None, launcher=None):
    files={p.name:p for p in ROOT.glob('*.py')}
    inherited=ROOT.parent/'expert_quality_20260924'
    for name in ('candidate_controller.py','candidate_planner.py'):
        files['inherited/'+name]=inherited/name
    if launcher:files['backend_launcher']=Path(launcher).resolve()
    if bridge_dir:
        base=Path(bridge_dir).resolve()
        for p in sorted(base.rglob('*')):
            if p.is_file():files['bridge/'+str(p.relative_to(base))]=p
    if not REQUIRED_SOURCES.issubset(files):
        raise ValueError('implementation incomplete: '+str(sorted(REQUIRED_SOURCES-set(files))))
    return {'new_expert/'+name:path for name,path in files.items()}


def bind_manifest(output, policy):
    output=Path(output).resolve()
    if policy not in POLICIES:raise ValueError('unsupported policy')
    snapshot=json.loads((output/'input_snapshot/manifest.json').read_text())
    prefix='source/new_expert/'
    files={name[len(prefix):]:{'path':'input_snapshot/objects/'+entry['sha256'],
                              'sha256':entry['sha256']}
           for name,entry in snapshot['files'].items() if name.startswith(prefix)}
    if not REQUIRED_SOURCES.issubset(files):raise ValueError('incomplete captured implementation')
    generated={str(p.relative_to(output)):sha256_file(p)
        for p in sorted((output/'planner_requests').rglob('*')) if p.is_file()}
    manifest={'schema':'wam_independent_expert_sources_v2','candidate_policy':policy,
              'derived_controller':CLASS,'files':files,'generated_files':generated}
    destination=output/'candidate_sources_manifest.json'
    destination.write_text(json.dumps(manifest,indent=2,allow_nan=False)+'\n')
    outcome_path=output/'outcome.json';outcome=json.loads(outcome_path.read_text())
    outcome.update(candidate_method=CLASS,candidate_policy=policy,
                   candidate_sources_manifest=destination.name,
                   candidate_sources_manifest_sha256=sha256_file(destination))
    outcome_path.write_text(json.dumps(outcome,indent=2,allow_nan=False)+'\n')
    return manifest


def verify_candidate_sources(source,outcome):
    source=Path(source).resolve();hashes={};reasons=[]
    try:
        if outcome.get('candidate_method')!=CLASS:raise ValueError('derived controller mismatch')
        name=outcome['candidate_sources_manifest'];path=(source/name).resolve()
        if source not in path.parents:raise ValueError('unsafe manifest path')
        expected=outcome['candidate_sources_manifest_sha256']
        if sha256_file(path)!=expected:raise ValueError('candidate manifest hash mismatch')
        hashes[name]=expected;manifest=json.loads(path.read_text())
        if (manifest.get('schema')!='wam_independent_expert_sources_v2'
                or manifest.get('derived_controller')!=CLASS
                or manifest.get('candidate_policy')!=outcome.get('candidate_policy')
                or outcome.get('candidate_policy') not in POLICIES):
            raise ValueError('candidate manifest identity mismatch')
        files=manifest['files']
        if not isinstance(files,dict):raise ValueError('candidate files must be a mapping')
        if not REQUIRED_SOURCES.issubset(files):raise ValueError('missing required implementation')
        if outcome.get('candidate_policy')=='curobo' and not {
                'bridge/robot_config.json','bridge/world_config.json','bridge/bridge_manifest.json',
                'backend_launcher','guarded_run.py','preflight.py','preflight/fk_request.json',
                'preflight/fk_result.json'}.issubset(files):
            raise ValueError('cuRobo requires captured bridge, launcher and resource guard')
        snapshot_path=(source/outcome['input_snapshot']).resolve()
        if source not in snapshot_path.parents:raise ValueError('unsafe snapshot path')
        if sha256_file(snapshot_path)!=outcome['input_snapshot_manifest_sha256']:
            raise ValueError('snapshot hash mismatch')
        snapshot=json.loads(snapshot_path.read_text())
        captured={key[len('source/new_expert/'):]:record
            for key,record in snapshot['files'].items() if key.startswith('source/new_expert/')}
        if set(captured)!=set(files):raise ValueError('candidate source set differs from snapshot')
        hashes[outcome['input_snapshot']]=outcome['input_snapshot_manifest_sha256']
        for key,item in files.items():
            expected=captured[key]['sha256']
            if (item['sha256']!=expected or
                    (source/item['path']).resolve()!=(snapshot_path.parent/'objects'/expected).resolve()):
                raise ValueError('candidate source differs from snapshot:'+key)
            hashes[item['path']]=expected
        generated=manifest['generated_files']
        if not isinstance(generated,dict):raise ValueError('generated files must be a mapping')
        actual={str(p.relative_to(source)) for p in (source/'planner_requests').rglob('*') if p.is_file()}
        if actual!=set(generated):raise ValueError('planner evidence file set changed')
        hashes.update(generated)
        if outcome.get('candidate_policy')=='curobo':
            from .preflight import verify_fk_pair
            robot,_=_captured_bridge(source,files,captured)
            ref=json.loads((source/files['bridge/fk_reference.json']['path']).read_text())
            req=json.loads((source/files['preflight/fk_request.json']['path']).read_text())
            res=json.loads((source/files['preflight/fk_result.json']['path']).read_text())
            verify_fk_pair(req,res,robot,ref['joint_vectors'],files['preflight/fk_request.json']['sha256'],
                           files['curobo_backend.py']['sha256'],_captured_assets(robot,{},files,captured))
        verify_planner_records(source,outcome,generated,files,captured,snapshot)
        if outcome.get('candidate_policy')=='curobo' and outcome.get('success'):
            if len(outcome.get('planner_calls',[]))!=max(0,len(outcome['points'])-1):
                raise ValueError('completed cuRobo sequence has missing planner calls')
    except (OSError,ValueError,KeyError,TypeError) as error:
        reasons.append('independent_candidate_sources:'+str(error))
    reasons.extend(verify_hashes(source,hashes))
    return {'status':'failed' if reasons else 'passed','artifact_sha256':hashes,'reasons':reasons}


def _captured_bridge(source,files,captured):
    root=Path(captured['bridge/robot_config.json']['source_path']).parent
    robot=json.loads((source/files['bridge/robot_config.json']['path']).read_text())
    world=json.loads((source/files['bridge/world_config.json']['path']).read_text())
    robot['robot_cfg']['kinematics']['urdf_path']=str(root/robot['robot_cfg']['kinematics']['urdf_path'])
    robot['robot_cfg']['kinematics']['asset_root_path']=str(root)
    for mesh in world['mesh'].values():mesh['file_path']=str(root/mesh['file_path'])
    return robot,world


def _captured_assets(robot,world,files,captured):
    from .preflight import expected_assets
    paths={entry['source_path']:files[key]['sha256'] for key,entry in captured.items()
           if key.startswith('bridge/')}
    return expected_assets(robot,world,lambda path:paths[path])


def verify_planner_records(source,outcome,generated,files,captured,snapshot):
    """Check actual request origins, generated result identity, and executed routes."""
    calls=outcome.get('planner_calls',[]);seen=set();origins={}
    if not isinstance(calls,list):raise ValueError('planner calls must be a list')
    from .curobo_backend import EXPECTED_COMMIT
    for call in calls:
        directory=call['directory']
        if directory in seen or not directory.startswith('planner_requests/'):
            raise ValueError('duplicate or invalid planner evidence directory')
        seen.add(directory)
        for filename in ('request.json','call.json','backend.log'):
            if directory+'/'+filename not in generated:raise ValueError('planner call lacks saved evidence')
        request=json.loads((source/directory/'request.json').read_text())
        receipt=json.loads((source/directory/'call.json').read_text())
        if receipt.get('request_sha256')!=generated[directory+'/request.json']:
            raise ValueError('planner receipt request hash mismatch')
        if receipt.get('launcher_sha256')!=files['backend_launcher']['sha256']:
            raise ValueError('planner launcher differs from captured source')
        robot,world=_captured_bridge(source,files,captured)
        if request.get('robot_config')!=robot or request.get('world')!=world:
            raise ValueError('planner used a different resolved bridge config')
        if (not np.array_equal(request.get('start7'),call['measured_start_q'])
                or not np.array_equal(request.get('end7'),call['requested_goal_q'])):
            raise ValueError('planner request endpoints differ from recorded origin/goal')
        row=call['physics_row']
        if type(row)!=int or row<=0 or row-1 in origins:raise ValueError('invalid or repeated planner origin row')
        origins[row-1]=call
        result_name=directory+'/result.json'
        if result_name not in generated:
            if (receipt.get('returncode')==0 or call.get('proposal_accepted')
                    or call.get('motion_complete') or outcome.get('success')):
                raise ValueError('accepted planner call has no result')
            continue  # A captured launch/timeout failure is valid diagnostic evidence.
        if receipt.get('result_sha256')!=generated[result_name]:raise ValueError('planner result hash mismatch')
        result=json.loads((source/result_name).read_text())
        if not call.get('proposal_accepted'):
            if outcome.get('success'):raise ValueError('successful sequence contains rejected proposal')
            continue
        meta=result.get('metadata',{})
        if (receipt.get('returncode')!=0 or result.get('success') is not True
                or meta.get('backend')!='curobo' or meta.get('fallback_used') is not False
                or meta.get('curobo_commit')!=EXPECTED_COMMIT
                or meta.get('curobo_source_dirty') not in (False,'')
                or meta.get('backend_source_sha256')!=files['curobo_backend.py']['sha256']
                or meta.get('request_sha256')!=generated[directory+'/request.json']):
            raise ValueError('accepted proposal has invalid backend identity')
        from .preflight import verify_assets
        verify_assets(meta,_captured_assets(robot,world,files,captured))
        if (meta.get('solver_mode')!='native_graph_only'
                or any(meta.get('options',{}).get(k)!=v for k,v in request['options'].items())):
            raise ValueError('accepted backend solver options differ from request')
        from .planner_service import validate_proposal
        q=validate_proposal(result,request['start7'],request['end7'])
        routes=[r for r in outcome.get('inter_point_routes',[]) if r.get('planner_evidence_directory')==directory]
        if len(routes)!=1:raise ValueError('proposal must bind exactly one executed route')
        route=routes[0]
        if (route['from_target']!=call['from_target'] or route['to_target']!=call['to_target']
                or route['start_physics_row']!=row or route.get('planning_accepted') is not True):
            raise ValueError('planner route identity mismatch')
        expected=np.vstack([call['retained_command_q'],request['start7'],q,request['end7']])
        expected=expected[np.r_[True,np.max(np.abs(np.diff(expected,axis=0)),axis=1)>1e-12]]
        actual=np.asarray(route['joint_waypoints_q'])
        if actual.shape!=expected.shape or not np.array_equal(actual,expected):
            raise ValueError('executed route differs from proposal plus explicit connectors')
        if outcome.get('success') and (call.get('motion_complete') is not True or route.get('motion_complete') is not True):
            raise ValueError('successful sequence has incomplete cuRobo motion')
        point_plan=snapshot['files'][call['to_target']+'/plan']['sha256']
        plan=json.loads((source/'input_snapshot/objects'/point_plan).read_text())
        candidates=[r for r in plan['points'] if r.get('success') and r.get('name')==call['to_target']]
        if len(candidates)!=1 or not np.array_equal(candidates[0]['transfer_q'][-1],request['end7']):
            raise ValueError('cuRobo goal differs from pinned prepose')
    if origins:
        with (source/'expert_trace.jsonl').open() as stream:
            for index,line in enumerate(stream):
                if index not in origins:continue
                row=json.loads(line);call=origins.pop(index)
                if (row['time_s']!=call['simulation_time_s']
                        or not np.array_equal(row['q'],call['measured_start_q'])
                        or not np.array_equal(row['q_command'],call['retained_command_q'])):
                    raise ValueError('planner origin differs from actual last physics sample')
                if not origins:break
        if origins:raise ValueError('planner origin outside saved trace')
