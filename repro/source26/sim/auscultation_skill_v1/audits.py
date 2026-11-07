"""Unchanged mechanical/posture equations with truthful variable-block binding."""
import hashlib
import json
from pathlib import Path

from .trace_binding import validate_control_trace


def mechanical_audit(path):
    from sim.auscultation_expert_layout_v1.audit_episode import audit_sequence
    from sim.auscultation_expert_layout_v1.controller import verify_snapshot
    path=Path(path)
    artifacts={name:(path/name).read_bytes() for name in
               ('expert_trace.jsonl','control_commands.json','outcome.json')}
    rows=[json.loads(line) for line in artifacts['expert_trace.jsonl'].splitlines()]
    outcome=json.loads(artifacts['outcome.json'])
    report=audit_sequence(rows,outcome)
    report['mechanical_accepted']=report['accepted']
    try:
        report['control_trace_binding']=validate_control_trace(json.loads(artifacts['control_commands.json']),rows)
        snapshot=path/Path(outcome['input_snapshot']).parent
        if hashlib.sha256((snapshot/'manifest.json').read_bytes()).hexdigest()!=outcome['input_snapshot_manifest_sha256']:
            raise ValueError('snapshot manifest mismatch')
        manifest=verify_snapshot(snapshot)
        if outcome['collision_regeneration_sha256']!=manifest['files']['source/collision_regeneration']['sha256']:
            raise ValueError('geometry source binding mismatch')
    except (OSError,KeyError,TypeError,ValueError) as error:
        report['accepted']=False; report['reasons'].append('trace_or_source_binding:'+str(error))
    report['artifact_sha256']={name:hashlib.sha256(data).hexdigest() for name,data in artifacts.items()}
    for name,data in artifacts.items():
        if (path/name).read_bytes()!=data:
            report['accepted']=False; report['reasons'].append('artifact changed during audit:'+name)
    report['control_binder']='sim.auscultation_skill_v1.trace_binding.validate_control_trace'
    (path/'independent_audit.json').write_text(json.dumps(report,indent=2,allow_nan=False)+'\n')
    return report


def audit_saved(output):
    from sim.auscultation_expert_layout_v1.audit_low_elbow import audit_directory
    output=Path(output); results={}
    for key,function in (('mechanics',mechanical_audit),('posture',audit_directory)):
        try: results[key]=function(output)
        except Exception as error: results[key]={'accepted':False,'reasons':[str(error)]}
    report=dict(accepted=all(r.get('accepted') is True for r in results.values()),audits=results,
                scope='simulation mechanics and low-elbow posture; variable blocks bound to actual samples')
    (output/'verification_summary.json').write_text(json.dumps(report,indent=2,allow_nan=False)+'\n')
    return report
