"""Explicit Source32/Source34/Source37 teacher readers; no neutral substitution.

The reports remain original source-process evidence. Independent compact
commands bind them to actual STOP ticks; the unchanged parking checker then
replays the current compatible numeric engine and checks full measured poses.
Source32 query recovery and Source34/Source37 late-DART have explicit profiles.
No old model is imported and ordinary neutral observation remains strict.
"""
import ast
from copy import deepcopy
import hashlib
from pathlib import Path

import numpy as np

from . import m2_given_contact_dataset as data, m2_given_contact_runtime as rt
from . import m2_given_contact_intent as intent, m2_given_contact_recovery as recovery
from . import m2_stop_residual_trajectory as residual

need=rt.need
EVIDENCE_PROFILE='source32_query_recovery_v4_compact_teacher_stop_v2'
SOURCE32=dict(root='/opt/wam-runtime/code/wam-act-recovery-20261106-v32',
    code_commit='d1da79b99394163d5d778f59d57e5e42748b0956',manifest_files=1684,
    code_commit_file=dict(path='/opt/wam-runtime/code/wam-act-recovery-20261106-v32/CODE_COMMIT',bytes=41,
        sha256='5b0275a29285cfe9bfcdf7dab115825a0db94bdfad95ba60923a5d56d6b630c7'),
    source_manifest=dict(path='/opt/wam-runtime/code/wam-act-recovery-20261106-v32/SOURCE_MANIFEST.json',bytes=291519,
        sha256='d46d697994cce1dd8ef9674352c87ead09f4b966dd037222c0faf25864b18bd2'),
    archive=dict(path='/workspace/wam/source_archives/wam-act-recovery-20261106-v32/source.tar.gz',bytes=6208968,
        sha256='b3f0c22703e7d23b0e1fe08ee05c594a70588708795950de3b1733acde1971c0'),
    engine_files={
        'sim/ausc_policy/m2_stop_residual_trajectory.py':dict(bytes=25258,sha256='105aac7fef8ac9a66f0739a862753f607290bc454df8f25496cfea3d2d9b5e4d'),
        'sim/ausc_policy/m2_given_contact_handoff.py':dict(bytes=18500,sha256='0ce411e8747b445701ac089942a1b5fa2af3900f432ec7ce4a3b4ab9e87091ed'),
        'sim/auscultation_pi05_v6/execution.py':dict(bytes=4987,sha256='d143e2e659d47e5d97108869a9b7d104d98213586d809a0f4f5157e3079e98d3'),
        'sim/ausc_policy/m2_given_contact_controller.py':dict(bytes=54351,sha256='a34880a1f7c0e5917cf2e7dc00b403340496836e87a801ce2d4acf18a52bd42d')})
NUMERIC_ENGINE_SHA256='cfa91cac7226b013cda6164a66908c89cf83615cd8f09e859b41ad9cc0805d94'
DART_EVIDENCE_PROFILE='source34_late_dart_compact_teacher_stop_v1'
SOURCE34=dict(root='/opt/wam-runtime/code/wam-act-recovery-20261106-v34',
    code_commit='ac2bc918f733e25f419d73bc83894be7e4d01545',manifest_files=1690,
    code_commit_file=dict(path='/opt/wam-runtime/code/wam-act-recovery-20261106-v34/CODE_COMMIT',bytes=41,
        sha256='1db081f06ea086690e30187b9a47d3fc096b6e1bddee61ac15a09cee88218e7c'),
    source_manifest=dict(path='/opt/wam-runtime/code/wam-act-recovery-20261106-v34/SOURCE_MANIFEST.json',bytes=292497,
        sha256='4473431336b7a6038c5acbfefc65a6807778e2f7048b9935c238b1c01c6cf7c9'),
    engine_files={**deepcopy(SOURCE32['engine_files']),
        'sim/ausc_policy/m2_stop_residual_trajectory.py':dict(bytes=25479,
            sha256='59dd664cf8b9769d1c07c6af94260f34b06c59999a5936887855b74c2fbf0dfd'),
        'sim/ausc_policy/m2_given_contact_recovery.py':dict(bytes=35465,
            sha256='72f8e611fa3a71d921733d733566f038a830a7e5822089ce286cd298ce323926')})
DART37_EVIDENCE_PROFILE='source37_late_dart_compact_teacher_stop_v1'
SOURCE37=dict(root='/opt/wam-runtime/code/wam-act-recovery-20261106-v37',
    code_commit='f9c38dbfb3f36aea80fd785c7317f0fabb24fdcc',manifest_files=1696,
    code_commit_file=dict(path='/opt/wam-runtime/code/wam-act-recovery-20261106-v37/CODE_COMMIT',bytes=41,
        sha256='690d6813d31eb88ff38c84bed487005a6db0e8b919a3a651d934bb4b1ee40840'),
    source_manifest=dict(path='/opt/wam-runtime/code/wam-act-recovery-20261106-v37/SOURCE_MANIFEST.json',bytes=293546,
        sha256='fb99640af5c9c48bc55836ca02200abbddd9d3b05ac797e8401268bc0f541544'),
    engine_files={**deepcopy(SOURCE34['engine_files']),
        'sim/ausc_policy/m2_given_contact_recovery.py':dict(bytes=36320,
            sha256='cb952ce3138e4aeeb0081dab9303d0bdd132fc6abe9c1672b17407878a6a6840')})
# Static data inputs retained by the original nominal config, never runtime
# modules or aliases for Source37 code. Keep the source descriptor unchanged.
SOURCE37_HISTORICAL_JSON_ARTIFACTS={
    SOURCE34['root']+'/sim/auscultation_vla_v2/calibration/veo6_tared_snapshot.json':dict(
        path=SOURCE34['root']+'/sim/auscultation_vla_v2/calibration/veo6_tared_snapshot.json',bytes=3374,
        sha256='ce6bcda3bef2282e07be0f44686dc007756ede57cdfcd732d35e28c1d469d054'),
    SOURCE34['root']+'/sim/camera_framing_20260923/recommended_camera.json':dict(
        path=SOURCE34['root']+'/sim/camera_framing_20260923/recommended_camera.json',bytes=8126,
        sha256='aeac00eb5a963520e869c60efff52a300484bd1c4a49232067a669c525ad07f0')}


def _portable_ast_dump(tree):
    """Ignore only empty PEP 695 fields absent on Python 3.11.

    Nonempty type parameters and every other AST field remain meaningful.
    Work on a copy so the source AST and wrapper proof stay unmodified.
    """
    tree=deepcopy(tree)
    for node in ast.walk(tree):
        if type(node) in (ast.ClassDef,ast.FunctionDef,ast.AsyncFunctionDef) and 'type_params' in node._fields:
            value=getattr(node,'type_params',None)
            if type(value) is list and not value:
                node._fields=tuple(field for field in node._fields if field!='type_params')
    return ast.dump(tree,include_attributes=False)


def numeric_engine_fingerprint():
    """Entire Source32 engine AST, normalizing only the source-check extraction.

    Normalize only the extracted private kernel's name/expected-source argument
    back to Source32 syntax. Numeric statements, constants and ordering remain
    in the fingerprint. This never executes archived Python.
    """
    module=ast.parse(Path(residual.__file__).read_text())
    nodes=[]
    wrapper=ast.parse('def verify_report(report, commands):\n    \"Check evidence without trusting its event digest or reported counters.\"\n    return _verify_report(report, commands, expected_sources=_source_hashes())').body[0]
    class OriginalSourceCheck(ast.NodeTransformer):
        def visit_Name(self,node):
            if node.id=='expected_sources':
                return ast.Call(func=ast.Name(id='_source_hashes',ctx=ast.Load()),args=[],keywords=[])
            return node
    for node in module.body:
        if isinstance(node,ast.FunctionDef) and node.name=='verify_report':
            need(_portable_ast_dump(node)==_portable_ast_dump(wrapper),
                'Default current-source STOP report wrapper changed')
            continue
        if isinstance(node,ast.FunctionDef) and node.name=='_verify_report':
            node.name='verify_report';node.args.kwonlyargs=[];node.args.kw_defaults=[]
            OriginalSourceCheck().visit(node)
        nodes.append(node)
    return hashlib.sha256(_portable_ast_dump(ast.Module(body=nodes,type_ignores=[])).encode()).hexdigest()


def validate_profile(evidence_profile,recorded_source):
    need(evidence_profile==EVIDENCE_PROFILE and recorded_source==SOURCE32,
        'Explicit approved Source32 compact teacher evidence profile/descriptor required')


def report_sources():
    return {str(Path(SOURCE32['root'])/name):proof['sha256'] for name,proof in SOURCE32['engine_files'].items()
        if not name.endswith('m2_given_contact_controller.py')}


def validate_source32_header(freeze,*,evidence_profile,recorded_source):
    """Bounded metadata only; full archived bytes remain the original gate.

    The ordinary dataset source-freeze gate authenticates all archived source
    files during export. Inference reads this pinned manifest and its code
    metadata, never captures, a simulator or a historical model factory.
    """
    validate_profile(evidence_profile,recorded_source)
    need(isinstance(freeze,dict) and freeze.get('schema')==data.SOURCE_FREEZE_SCHEMA and
        freeze.get('code_commit')==SOURCE32['code_commit'] and
        freeze.get('source_manifest')==SOURCE32['source_manifest'], 'Approved Source32 source root/manifest/commit differs')
    commit=freeze.get('code_commit_file',{})
    need(commit==SOURCE32['code_commit_file'], 'Recorded Source32 CODE_COMMIT proof differs')
    path=rt.bound_path(commit)
    need(path.read_text().strip()==SOURCE32['code_commit'],'Recorded Source32 CODE_COMMIT bytes differ')
    rows=rt.read_bound(SOURCE32['source_manifest'])
    _validate_source32_inventory(freeze,rows)
    # Same immutable original modules for all other numeric helpers/controller.
    root=Path(__file__).resolve().parents[2]
    for name,proof in SOURCE32['engine_files'].items():
        if name.endswith('m2_stop_residual_trajectory.py'):continue
        source=root/name
        need(source.stat().st_size==proof['bytes'] and data.sha256(source)==proof['sha256'],
            'Current numeric helper/controller is incompatible with approved Source32')
    need(numeric_engine_fingerprint()==NUMERIC_ENGINE_SHA256,'Current STOP numeric engine differs from approved Source32')
    return freeze


def _validate_source32_inventory(freeze,rows):
    """Pure metadata layer; callers must authenticate the pinned manifest."""
    need(freeze.get('code_commit_file')==SOURCE32['code_commit_file'], 'Recorded Source32 CODE_COMMIT proof differs')
    need(isinstance(rows,list) and len(rows)==SOURCE32['manifest_files'],'Approved Source32 full manifest count differs')
    names=[r.get('file') for r in rows]
    need(all(isinstance(n,str) and n and not Path(n).is_absolute() and '..' not in Path(n).parts for n in names) and
        len(set(names))==len(names),'Approved Source32 source manifest paths differ')
    by_name={r['file']:r for r in rows}
    files=freeze.get('files',{})
    need(isinstance(files,dict) and bool(files),'Source32 archived runtime source inventory missing')
    for name,proof in SOURCE32['engine_files'].items():
        original=str(Path(SOURCE32['root'])/name);entry=files.get(original,{})
        need(sum(Path(p).name==Path(name).name for p in files)==1 and
            all(by_name.get(name,{}).get(k)==v for k,v in proof.items()) and
            entry.get('path')==original and all(entry.get(k)==v for k,v in proof.items()),
            'Approved Source32 engine/controller source manifest/inventory differs')


def _configuration_and_sources(run,metadata,config,context):
    validate_source32_header(context['source_freeze'],evidence_profile=context['evidence_profile'],recorded_source=context['recorded_source'])
    row=context.get('source_declaration');base=context.get('evidence_base')
    need(isinstance(row,dict) and isinstance(base,str) and Path(base).is_absolute() and
        row.get('run_id')==context.get('run_id')=='act9-query-recovery-20261107-v1',
        'Bound original Source32 declaration/whole-owner evidence required')
    bound_run,bound_config,bound_freeze=data._native_evidence(row,Path(base),metadata,physical=False)
    need(bound_run==run and bound_config==config and bound_freeze==context['source_freeze'],
        'Teacher run/config/source differs from original bound whole-owner chain')
    data.shared._require_run_identity(run,row)
    binding=recovery.preflight(config)
    need(binding['settings']['schema']==recovery.QUERY_SCHEMA,'Exact query-recovery V4 scene/config required')
    need(config.get(recovery.ROUTE_KEY,{}).get('schema')==recovery.QUERY_SCHEMA and
        config.get('m2_given_contact_control',{}).get('metadata')==metadata and
        metadata.get('source_kind')=='isaac_live_physics' and metadata.get('replay') is False and
        metadata.get('split')=='train' and metadata.get('subject_id')=='p001120' and
        context.get('run_id')=='act9-query-recovery-20261107-v1' and
        run.get('executed_controller_class')==recovery.__name__+'.GivenContactRecoveryExpert' and
        run.get('learned_policy_executed') is False and run.get('actual_student_command_prefix_replayed') is True and
        all(run.get(k) is True for k in ('success','physical_validation','controller_motion_complete','compact_capture_complete')) and
        run.get('errors')==[], 'Exact actual Source32 query-recovery teacher/config/acquisition required')
    need(residual.CONFIG_KEY in config,'Explicit native STOP residual configuration required')
    residual.native_options(config,metadata=metadata)
    runtime=run.get('runtime_sources_sha256',{})
    for name,proof in SOURCE32['engine_files'].items():
        need(sum(Path(p).name==Path(name).name for p in runtime)==1 and
            runtime.get(str(Path(SOURCE32['root'])/name))==proof['sha256'],'Actual Source32 runtime engine/controller inventory differs')
    need(all(context['source_freeze']['files'].get(p,{}).get('sha256')==sha for p,sha in runtime.items()),
        'Actual Source32 runtime inventory differs from archived inventory')


def verify_source32_report(report,commands,*,evidence_profile,recorded_source):
    """Numeric report check only; capture admission also requires full sources."""
    validate_profile(evidence_profile,recorded_source)
    need(numeric_engine_fingerprint()==NUMERIC_ENGINE_SHA256,'Current STOP numeric engine differs from approved Source32')
    return residual._verify_report(report,commands,expected_sources=report_sources())


def _bits(value):
    a=np.asarray(value,dtype=np.float64)
    need(a.shape==(7,) and np.isfinite(a).all(),'Complete finite actual compact q7 evidence required')
    return a.tobytes()


def _verify_compact_parking(run,receipt,audit,metadata,reference,phase_config,*,expected_sources,kin=None):
    """Shared evidence layer; synthetic unit fixtures do not use public admission."""
    preview=receipt.get('parking_preview',{});actual=receipt.get('parking_stop_residual_adapter',{})
    proposed=preview.get('stop_residual_adapter',{});start=receipt.get('parking_start_physics_row');steps=preview.get('stop_steps')
    need(preview.get('schema')==residual.PREVIEW_SCHEMA and type(start) is int and type(steps) is int and
        3<=start<start+steps<=len(audit) and receipt.get('parking_stop_complete_physics_row')==start+steps,
        'Explicit complete teacher compact STOP interval/report required')
    commands=np.asarray([row.get('q_command') for row in audit[start:start+steps]],dtype=np.float64)
    need(commands.shape==(steps,7) and np.isfinite(commands).all(),'Independent actual compact STOP commands missing')
    residual._verify_report(proposed,commands,expected_sources=expected_sources)
    residual._verify_report(actual,commands,expected_sources=expected_sources)
    history=np.asarray([row.get('q_command') for row in audit[start-3:start]],dtype=np.float64)
    need(history.shape==(3,7) and np.isfinite(history).all() and
        all(np.asarray(value,dtype=np.float64).shape==(3,7) and
            np.asarray(value,dtype=np.float64).tobytes()==history.tobytes() for value in
            (preview.get('frozen_command_history'),proposed['initial_command_history'],actual['initial_command_history'])) and
        _bits(proposed['target_q'])==_bits(actual['target_q'])==_bits(preview.get('target_q'))==_bits(history[-1]) and
        _bits(receipt.get('parking_tail',{}).get('command_q'))==_bits(commands[-1]),
        'Actual teacher STOP history/target/constant-tail command differs bit-exactly')
    need(np.asarray(preview.get('commands'),dtype=np.float64).tobytes()==commands.tobytes() and
        proposed['events']==actual['events'] and proposed['events_sha256']==actual['events_sha256'] and
        proposed['initial_stop_input']==actual['initial_stop_input']==preview.get('stop_input_state') and
        proposed['initial_command_history']==actual['initial_command_history']==preview.get('frozen_command_history'),
        'Preview/live/independent compact STOP commands or state differ')
    ids=actual['object_ids']
    need(ids==preview.get('live_state_before',{}).get('interpolator_object_ids')==
        receipt.get('parking_tail',{}).get('object_ids') and not set(ids.values())&set(proposed['object_ids'].values()),
        'Teacher live/preview/tail objects missing, changed or aliased')
    target=actual['target_q'];setter0=None
    end=receipt.get('parking_end_physics_row')
    need(type(end) is int and start+steps<end<=len(audit),'Complete actual teacher parking tail required')
    for index,row in enumerate(audit[start:end]):
        raw=row.get('raw_intent') or {};setter=deepcopy(raw.get('setter') or {});digest=setter.pop('setter_sha256',None)
        need(raw.get('schema')==intent.AUDIT_SCHEMA and raw.get('action_semantics_id')==intent.ACTION_SEMANTICS_ID and
            digest==hashlib.sha256(intent._canonical(setter)).hexdigest() and
            setter.get('schema')==intent.INTENT_SCHEMA and setter.get('action_semantics_id')==intent.ACTION_SEMANTICS_ID and
            type(setter.get('intent_id')) is int and setter['intent_id']>=0 and
            setter.get('offgrid_stop') is True and setter.get('control_owner')=='L2_STOP' and
            setter.get('execution_role')=='given_l2_park' and setter.get('physics_row_start')==start and
            type(setter.get('issued_at_s')) in (int,float) and abs(setter['issued_at_s']-start/240.)<=1e-8 and
            setter.get('context')==[metadata['episode_id'],metadata['target_id'],0,'APPROACH'] and
            setter.get('reference_sha256')==reference.sha256==metadata['reference_sha256'] and setter.get('reference_revision')==0 and
            raw.get('native_tick_in_cell')==index and _bits(setter.get('target_q'))==_bits(target) and
            _bits(raw.get('live_target_q'))==_bits(target)==_bits(raw.get('live_target_after_step_q')),
            'Actual compact STOP setter/target/context/reference/tick evidence differs')
        if setter0 is None:setter0=deepcopy(raw['setter'])
        need(raw['setter']==setter0 and row.get('physics_row')==start+index and
            type(row.get('time_s')) in (int,float) and abs(row['time_s']-(start+index+1)/240.)<=1e-8 and
            row.get('actual_control_owner')=='L2_STOP' and _bits(row.get('q_command'))==_bits(commands[min(index,steps-1)]),
            'Actual compact STOP owner/command/row/clock differs')
        for side,pose in [('controller_before','parking_pose_before'),('controller_after','parking_pose_after')]:
            labels=row.get(side,{})
            need(labels.get('actual_control_owner')=='L2_STOP' and labels.get('raw_intent')==raw and
                labels.get('parking_observed_pose')==row.get(pose) and row.get(pose) is not None,
                'Actual compact controller STOP owner/intent/full-pose binding differs')
    data._interior_parking_checks(run,receipt,audit,metadata,reference,phase_config,kin=kin,_native_stop_residual=True)
    return dict(adapter_id=residual.ADAPTER_ID,ticks=actual['ticks'],
        events_sha256=actual['events_sha256'],preview_live_events_equal=True,independent_compact_commands_equal=True,
        original_parking_gates_passed=True)


def verify_teacher_parking(run,receipt,audit,metadata,reference,phase_config,*,config,teacher_context,kin=None):
    if teacher_context.get('evidence_profile')==DART37_EVIDENCE_PROFILE:
        _source37_configuration_and_sources(run,metadata,config,teacher_context)
        proof=_verify_compact_parking(run,receipt,audit,metadata,reference,phase_config,
            expected_sources=dart37_report_sources(),kin=kin)
        return dict(proof,evidence_profile=DART37_EVIDENCE_PROFILE)
    if teacher_context.get('evidence_profile')==DART_EVIDENCE_PROFILE:
        _source34_configuration_and_sources(run,metadata,config,teacher_context)
        proof=_verify_compact_parking(run,receipt,audit,metadata,reference,phase_config,
            expected_sources=dart_report_sources(),kin=kin)
        return dict(proof,evidence_profile=DART_EVIDENCE_PROFILE)
    _configuration_and_sources(run,metadata,config,teacher_context)
    proof=_verify_compact_parking(run,receipt,audit,metadata,reference,phase_config,expected_sources=report_sources(),kin=kin)
    return dict(proof,evidence_profile=EVIDENCE_PROFILE)


def verify_teacher_physical(run,audit,metadata,*,config,teacher_context):
    """Fixed teacher route through every original physical/contact gate."""
    if teacher_context.get('evidence_profile')==DART37_EVIDENCE_PROFILE:
        return verify_source37_physical(run,audit,metadata,config=config,teacher_context=teacher_context)
    if teacher_context.get('evidence_profile')==DART_EVIDENCE_PROFILE:
        return verify_source34_physical(run,audit,metadata,config=config,teacher_context=teacher_context)
    _configuration_and_sources(run,metadata,config,teacher_context)
    return data._physical_checks(run,audit,metadata,_teacher_stop_context=dict(teacher_context,config=config))


def validate_dart_profile(evidence_profile,recorded_source):
    need(evidence_profile==DART_EVIDENCE_PROFILE and recorded_source==SOURCE34,
        'Explicit approved Source34 late-DART compact teacher profile/descriptor required')


def validate_dart37_profile(evidence_profile,recorded_source):
    need(evidence_profile==DART37_EVIDENCE_PROFILE and recorded_source==SOURCE37,
        'Explicit approved Source37 late-DART compact teacher profile/descriptor required')


def dart_report_sources():
    return _dart_report_sources(SOURCE34)


def dart37_report_sources():
    return _dart_report_sources(SOURCE37)


def _dart_report_sources(source):
    names=('sim/ausc_policy/m2_stop_residual_trajectory.py','sim/ausc_policy/m2_given_contact_handoff.py',
        'sim/auscultation_pi05_v6/execution.py')
    return {str(Path(source['root'])/name):source['engine_files'][name]['sha256'] for name in names}


def _validate_source34_inventory(freeze,rows):
    return _validate_dart_inventory(freeze,rows,SOURCE34)


def _validate_source37_inventory(freeze,rows):
    return _validate_dart_inventory(freeze,rows,SOURCE37)


def _manifest_inventory(rows,source):
    need(isinstance(rows,list) and
        len(rows)==source['manifest_files'],'Original DART source commit/full manifest count differs')
    names=[r.get('file') for r in rows]
    need(all(isinstance(n,str) and n and not Path(n).is_absolute() and '..' not in Path(n).parts for n in names) and
        len(set(names))==len(names),'Original DART source full manifest paths differ')
    return {r['file']:r for r in rows}


def _historical_source34_inventory():
    """Authenticate old data metadata without admitting its runtime helpers."""
    need(rt.bound_path(SOURCE34['code_commit_file']).read_text().strip()==SOURCE34['code_commit'],
        'Historical Source34 CODE_COMMIT bytes differ')
    return _manifest_inventory(rt.read_bound(SOURCE34['source_manifest']),SOURCE34)


def _validate_dart_inventory(freeze,rows,source):
    need(freeze.get('code_commit_file')==source['code_commit_file'],'Original DART CODE_COMMIT proof differs')
    inventory=_manifest_inventory(rows,source);files=freeze.get('files');historical=None
    need(isinstance(files,dict) and bool(files),'Original DART source runtime module inventory missing')
    for original,entry in files.items():
        path=Path(original)
        need(path.is_absolute() and entry.get('path')==original,'Recorded DART source module origin/path changed')
        if original.startswith(source['root']+'/'):
            proof=inventory.get(path.relative_to(source['root']).as_posix(),{})
            need(all(entry.get(k)==proof.get(k) for k in ('bytes','sha256')) and bool(proof),
                'Recorded DART source runtime module differs from full manifest')
        allowed=SOURCE37_HISTORICAL_JSON_ARTIFACTS.get(original) if source==SOURCE37 else None
        if allowed is not None:
            need(path.suffix=='.json' and all(entry.get(k)==v for k,v in allowed.items()),
                'Exact Source37 historical JSON path/bytes/hash required')
            if historical is None:historical=_historical_source34_inventory()
            relative=path.relative_to(SOURCE34['root']).as_posix()
            need(all(historical.get(relative,{}).get(k)==inventory.get(relative,{}).get(k)==allowed[k]
                for k in ('bytes','sha256')),'Historical JSON differs from original Source34 or Source37 full manifest')
        else:
            # No other foreign JSON or runtime module may change source root.
            need('/sim/' not in original or original.startswith(source['root']+'/sim/'),
                'Recorded DART repository module has a foreign source origin')
    for name,proof in source['engine_files'].items():
        original=str(Path(source['root'])/name)
        need(sum(Path(p).name==Path(name).name for p in files)==1 and
            all(inventory.get(name,{}).get(k)==v and files.get(original,{}).get(k)==v for k,v in proof.items()),
            'Original DART source engine/controller module graph differs')


def validate_source34_header(freeze,*,evidence_profile,recorded_source):
    validate_dart_profile(evidence_profile,recorded_source)
    return _validate_dart_header(freeze,SOURCE34,_validate_source34_inventory)


def validate_source37_header(freeze,*,evidence_profile,recorded_source):
    validate_dart37_profile(evidence_profile,recorded_source)
    return _validate_dart_header(freeze,SOURCE37,_validate_source37_inventory)


def _validate_dart_header(freeze,source,inventory_check):
    need(isinstance(freeze,dict) and freeze.get('schema')==data.SOURCE_FREEZE_SCHEMA and
        freeze.get('code_commit')==source['code_commit'] and freeze.get('source_manifest')==source['source_manifest'],
        'Original DART source freeze/root/manifest/commit differs')
    need(freeze.get('code_commit_file')==source['code_commit_file'],'Original DART CODE_COMMIT proof differs')
    need(rt.bound_path(source['code_commit_file']).read_text().strip()==source['code_commit'],
        'Original DART source CODE_COMMIT bytes differ')
    inventory_check(freeze,rt.read_bound(source['source_manifest']))
    root=Path(__file__).resolve().parents[2]
    for name,proof in source['engine_files'].items():
        if name.endswith('m2_stop_residual_trajectory.py'):continue
        path=root/name
        need(path.stat().st_size==proof['bytes'] and data.sha256(path)==proof['sha256'],
            'Current helper/controller differs from original '+source['root'])
    need(numeric_engine_fingerprint()==NUMERIC_ENGINE_SHA256,'Current STOP numeric engine differs from approved DART source')
    return freeze


def _source34_configuration_and_sources(run,metadata,config,context,*,positive=True):
    return _dart_configuration_and_sources(run,metadata,config,context,positive=positive,source=SOURCE34,
        header_check=validate_source34_header,owner_check=_source34_owner)


def _source37_configuration_and_sources(run,metadata,config,context,*,positive=True):
    return _dart_configuration_and_sources(run,metadata,config,context,positive=positive,source=SOURCE37,
        header_check=validate_source37_header,owner_check=_source37_owner)


def _dart_configuration_and_sources(run,metadata,config,context,*,positive,source,header_check,owner_check):
    from .m2_act_recovery_scope import TRAIN16_IDS
    need(metadata.get('split')=='train' and metadata.get('subject_id') in TRAIN16_IDS,
        'Only explicit original TRAIN16 identity may open DART teacher evidence')
    header_check(context['source_freeze'],evidence_profile=context['evidence_profile'],recorded_source=context['recorded_source'])
    row=context.get('source_declaration');base=context.get('evidence_base')
    need(isinstance(row,dict) and isinstance(base,str) and Path(base).is_absolute() and
        isinstance(context.get('run_id'),str) and row.get('run_id')==context['run_id'],
        'Bound original DART source declaration/whole-owner evidence required')
    bound=data._native_evidence(row,Path(base),metadata,physical=False)
    need(bound==(run,config,context['source_freeze']),'DART run/config/source differs from original whole-owner chain')
    owner_check(row,Path(base))
    data.shared._require_run_identity(run,row)
    binding=recovery.preflight(config)
    need(binding['settings']['schema']==recovery.LATE_SCHEMA and metadata.get('split')=='train' and
        metadata.get('subject_id') in TRAIN16_IDS and config['m2_given_contact_control']['metadata']==metadata and
        metadata.get('source_kind')=='isaac_live_physics' and metadata.get('replay') is False and
        run.get('executed_controller_class')==recovery.__name__+'.GivenContactRecoveryExpert' and
        run.get('learned_policy') is False and run.get('learned_policy_executed') is not True and
        run.get('actual_student_command_prefix_replayed') is not True,
        'Exact actual TRAIN late-DART teacher/config required; no query replay')
    need(residual.CONFIG_KEY in config,'Explicit original native STOP8 configuration required')
    residual.native_options(config,metadata=metadata)
    runtime=run.get('runtime_sources_sha256',{})
    for name,proof in source['engine_files'].items():
        need(sum(Path(p).name==Path(name).name for p in runtime)==1 and
            runtime.get(str(Path(source['root'])/name))==proof['sha256'],
            'Actual DART source engine/controller runtime module graph differs')
    need(all(context['source_freeze']['files'].get(p,{}).get('sha256')==sha for p,sha in runtime.items()),
        'Actual DART runtime inventory differs from full archived source')
    if positive:
        need(all(run.get(k) is True for k in ('success','physical_validation','controller_motion_complete','compact_capture_complete')) and
            run.get('errors')==[],'Complete actual full physical DART teacher required')
    return binding


def _source34_owner(row,base):
    return _dart_owner(row,base,SOURCE34)


def _source37_owner(row,base):
    return _dart_owner(row,base,SOURCE37)


def _dart_owner(row,base,source):
    """Authenticate actual prelaunch/terminal original guard and child graph."""
    owner,path=data._bound(row['owner_terminal'],base);folder=path.parent
    initial=rt.read_bound(rt.reference(folder/'run.json'));batch=owner.get('config',{})
    need(path==Path(batch.get('base',''))/'runs/ausc_policy'/row['run_id']/'result.json' and
        owner.get('run_id')==row['run_id'] and owner.get('code_commit')==source['code_commit'] and
        owner.get('completed') is True and owner.get('gpu_released') is True,
        'Actual original DART source owner/root/completion differs')
    common=('run_id','code_commit','config','config_sha256','command','resource_guard','started_utc')
    need(all(initial.get(k)==owner.get(k) for k in common) and initial.get('completed') is False,
        'Original prelaunch run/config differs from terminal owner')
    command=owner.get('command');output=folder/'output'
    need(isinstance(command,list) and len(command)==9 and command[:6]==[
        '/opt/wam-runtime/envs/isaacsim60/bin/python','-u','-B','-m','sim.ausc_policy.isaac_batch','--config'] and
        command[7:]==['--output',str(output)] and rt.read_bound(rt.reference(command[6]))==batch and
        rt.reference(command[6])['sha256']==owner.get('config_sha256'),
        'Original DART source owner executable/module/config/output differs')
    need(owner.get('resource_guard')==dict(path=source['root']+'/sim/ausc_policy/process_guard.py',
        sha256=rt.GUARD_SHA256,gpu_exit_grace_s=15.0),'Original DART source guard/path/grace differs')
    stage=owner.get('stage',{})
    need(rt.read_bound(rt.reference(folder/'lifecycle/isaac_batch.process.json'))==stage and stage.get('command')==command and
        stage.get('passed') is True and stage.get('exit_code')==0 and stage.get('process_group_drained') is True and
        all(stage.get(k)==[] for k in ('errors','signals','deferred_signals')),
        'Actual original DART source guard stage did not exit cleanly and drain')
    child,child_path=data._bound(row['child_result'],base)
    launch=rt.read_bound(rt.reference(output/(row['job_id']+'.launch.json')))
    need(child_path==output/(row['job_id']+'.result.json') and all(launch.get(k)==child.get(k) for k in
        ('id','config_sha256','pid','identity','output','started_utc')),
        'Actual original DART source child launch/result identity differs')
    summary=rt.read_bound(rt.reference(output/'COMPLETE.json'));jobs=batch.get('jobs',[]);results=summary.get('jobs',[])
    need(batch.get('experiment')=='isaac_batch' and type(batch.get('parallelism')) is int and 1<=batch['parallelism']<=4 and
        'resource_profile' not in batch and summary.get('completed') is True and summary.get('parallelism')==batch['parallelism'] and
        summary.get('attempted')==len(jobs)==len(results) and
        sorted(r.get('id') for r in results)==sorted(j.get('id') for j in jobs) and
        len({r.get('id') for r in results})==len(results) and
        [r for r in results if r.get('id')==row['job_id']]==[child] and
        summary.get('physical_successes')==sum(r.get('physical_pass') is True for r in results) and
        all(r.get('worker_complete') is True and r.get('exit_code')==0 for r in results),
        'Actual original DART source whole-owner worker graph incomplete or changed')
    return dict(original_guard_child_graph_passed=True,prelaunch=rt.reference(folder/'run.json'),
        stage=rt.reference(folder/'lifecycle/isaac_batch.process.json'),child_launch=rt.reference(output/(row['job_id']+'.launch.json')),
        batch_complete=rt.reference(output/'COMPLETE.json'))


def verify_source34_report(report,commands,*,evidence_profile,recorded_source):
    validate_dart_profile(evidence_profile,recorded_source)
    need(numeric_engine_fingerprint()==NUMERIC_ENGINE_SHA256,'Current STOP numeric engine differs from original Source34')
    return residual._verify_report(report,commands,expected_sources=dart_report_sources())


def verify_source37_report(report,commands,*,evidence_profile,recorded_source):
    validate_dart37_profile(evidence_profile,recorded_source)
    need(numeric_engine_fingerprint()==NUMERIC_ENGINE_SHA256,'Current STOP numeric engine differs from original Source37')
    return residual._verify_report(report,commands,expected_sources=dart37_report_sources())


def verify_source34_physical(run,audit,metadata,*,config,teacher_context):
    _source34_configuration_and_sources(run,metadata,config,teacher_context)
    return data._physical_checks(run,audit,metadata,_teacher_stop_context=dict(teacher_context,config=config))


def verify_source37_physical(run,audit,metadata,*,config,teacher_context):
    _source37_configuration_and_sources(run,metadata,config,teacher_context)
    return data._physical_checks(run,audit,metadata,_teacher_stop_context=dict(teacher_context,config=config))


def decode_source34_dart_capture(row,base,context,run,config,freeze,*,teacher_context):
    return _decode_dart_capture(row,base,context,run,config,freeze,teacher_context,validate_source34_header)


def decode_source37_dart_capture(row,base,context,run,config,freeze,*,teacher_context):
    return _decode_dart_capture(row,base,context,run,config,freeze,teacher_context,validate_source37_header)


def _decode_dart_capture(row,base,context,run,config,freeze,teacher_context,header_check):
    """Explicit DART route through the unchanged complete physical decoder."""
    from .m2_act_recovery_scope import TRAIN16_IDS
    need(row.get('split')==context[2].get('split')=='train' and
        row.get('subject_id')==context[2].get('subject_id') in TRAIN16_IDS,
        'Only explicit original TRAIN16 identity may open DART capture')
    header_check(freeze,evidence_profile=teacher_context.get('evidence_profile'),
        recorded_source=teacher_context.get('recorded_source'))
    need(teacher_context.get('source_freeze')==freeze and isinstance(row.get('capture_audit'),dict),
        'Bound original DART source freeze/actual audit transport required')
    return data._decode_capture(row,base,context,run,config,freeze,allow_truncated=True,
        _teacher_stop_context=teacher_context,_audit_transport=row['capture_audit'])
