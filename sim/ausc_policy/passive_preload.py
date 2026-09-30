"""Opt-in single explicit 1.5 N preload for a disposable Fresh diagnostic stage.

The legacy force drive's negative rest target already adds k*1.5 mm = 1.5 N.
Fresh independently applies -1.5 N each step. Only this duplicate rest-target
term is removed; nominal preload, explicit effort, Hall and all gates stay put.
No Isaac import is needed for CPU USD authoring and readback checks.
"""
import hashlib
import json
import math
from pathlib import Path

import numpy as np

JOINT_PATH = '/World/Stethoscope/PassiveCompression'
TARGET_ATTRIBUTE = 'drive:linear:physics:targetPosition'
TARGET_ATTRIBUTE_PATH = JOINT_PATH+'.'+TARGET_ATTRIBUTE
MODE = 'single_explicit_v1'
_FIELDS = dict(drive_type='Type', stiffness_N_m='Stiffness', damping_N_s_m='Damping',
    target_position_m='TargetPosition', target_velocity_m_s='TargetVelocity', max_force_N='MaxForce')


def file_sha256(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def passive_preload_settings(fresh_config):
    """Absence is the untouched legacy path; explicit input is strictly typed."""
    if fresh_config is None:
        return None
    if not isinstance(fresh_config, dict):
        raise ValueError('passive_preload requires a Fresh configuration')
    if 'passive_preload_model' not in fresh_config:
        return None
    value = fresh_config['passive_preload_model']
    if (type(value) is not dict or set(value) != {'mode'}
            or type(value['mode']) is not str or value['mode'] != MODE):
        raise ValueError('passive_preload_model requires only mode: single_explicit_v1')
    nominal = fresh_config.get('passive_preload_N')
    if type(nominal) not in (int, float) or not math.isfinite(nominal) or nominal != 1.5:
        raise ValueError('passive_preload nominal must remain the JSON number 1.5 N')
    return dict(mode=MODE, nominal_preload_N=1.5)


def protected_stage_sha256(stage):
    """Hash every prim property except the one declared target-position attribute."""
    def value_fingerprint(value):
        if type(value).__module__ == 'pxr.Vt':
            array = np.asarray(value)
            if array.dtype.kind in 'biufc':
                return dict(dtype=array.dtype.str, shape=array.shape,
                    sha256=hashlib.sha256(array.tobytes()).hexdigest())
        return str(value)
    rows = [('stage', {k: str(v) for k, v in stage.GetPseudoRoot().GetAllMetadata().items()})]
    for prim in stage.TraverseAll():
        attrs = {a.GetName(): dict(value=value_fingerprint(a.Get()),
            metadata={k: str(v) for k, v in a.GetAllMetadata().items()},
            samples=[(t, value_fingerprint(a.Get(t))) for t in a.GetTimeSamples()])
            for a in prim.GetAttributes() if str(a.GetPath()) != TARGET_ATTRIBUTE_PATH}
        rels = {r.GetName(): dict(targets=[str(t) for t in r.GetTargets()],
            metadata={k: str(v) for k, v in r.GetAllMetadata().items()}) for r in prim.GetRelationships()}
        rows.append((str(prim.GetPath()), {k: str(v) for k, v in prim.GetAllMetadata().items()}, attrs, rels))
    return hashlib.sha256(json.dumps(rows, sort_keys=True).encode()).hexdigest()


def _validate_drive_values(values, target):
    if type(values) is not dict or set(values) != set(_FIELDS) or values['drive_type'] != 'force':
        raise ValueError('Expected existing passive linear force drive')
    for key in set(_FIELDS)-{'drive_type'}:
        if type(values[key]) not in (int, float) or not math.isfinite(values[key]):
            raise ValueError('Invalid passive drive '+key)
    # These native USD attributes are Float. Compare their exact Float encoding,
    # not a newly introduced physics/control tolerance.
    for key, expected in [('stiffness_N_m', 1000.), ('damping_N_s_m', 3.2),
                          ('target_position_m', target), ('target_velocity_m_s', 0.)]:
        if values[key] != float(np.float32(expected)):
            raise ValueError('Unexpected passive drive '+key)
    if values['max_force_N'] <= 0:
        raise ValueError('Invalid passive drive max_force_N')


def _read_drive(stage, target):
    from pxr import UsdPhysics
    prim = stage.GetPrimAtPath(JOINT_PATH)
    if (not prim or not prim.IsA(UsdPhysics.PrismaticJoint)
            or not prim.HasAPI(UsdPhysics.DriveAPI, 'linear')):
        raise ValueError('Missing existing passive prismatic linear drive')
    drive = UsdPhysics.DriveAPI(prim, 'linear')
    values = {}
    for key, getter in _FIELDS.items():
        attr = getattr(drive, 'Get'+getter+'Attr')()
        if attr.GetNumTimeSamples():
            raise ValueError('Passive drive must have static attributes')
        values[key] = attr.Get()
    _validate_drive_values(values, target)
    return drive, values


def apply_single_explicit_preload(stage, fresh_config):
    """Set exactly one existing USD attribute, only under the explicit new mode."""
    settings = passive_preload_settings(fresh_config)
    if settings is None:
        return None
    from pxr import UsdGeom
    if (not stage.GetRootLayer().anonymous
            or len(stage.GetLayerStack(includeSessionLayers=False)) != 1
            or UsdGeom.GetStageMetersPerUnit(stage) != 1.):
        raise ValueError('Passive preload authoring requires a disposable flattened metric stage')
    drive, before = _read_drive(stage, -.0015)
    protected_before = protected_stage_sha256(stage)
    target = drive.GetTargetPositionAttr()
    if not target.HasAuthoredValueOpinion():
        raise ValueError('Existing authored passive target_position required')
    target.Set(0.)
    _, after = _read_drive(stage, 0.)
    protected_after = protected_stage_sha256(stage)
    if protected_after != protected_before:
        target.Set(before['target_position_m'])
        raise ValueError('Passive preload authoring changed other stage attributes')
    return dict(schema=MODE, mode=MODE, nominal_preload_N=1.5, explicit_effort_N=-1.5,
        joint_path=JOINT_PATH, changed_attributes=[TARGET_ATTRIBUTE_PATH], before=before, after=after,
        implicit_rest_target_preload_before_N=-before['stiffness_N_m']*before['target_position_m'],
        implicit_rest_target_preload_after_N=0.,
        protected_before_sha256=protected_before, protected_after_sha256=protected_after,
        other_stage_attributes_unchanged=True, source_sha256=file_sha256(__file__),
        authoring_api='existing UsdPhysics.DriveAPI(linear).GetTargetPositionAttr().Set(0.0)',
        authoring_scope='disposable run USD only; remove duplicate implicit term, retain nominal explicit preload',
        applied_effort_measured=False, physical_validation=False, training_eligible=False)


def verify_preload_readback(stage, fresh_config, receipt, *, robot=None, slider=None):
    """Read USD drive values, optionally through the established native DOF binding.

    This is not an applied-effort sensor. Existing per-step action records retain
    the actual -1.5 N requested/applied-effort evidence independently.
    """
    settings = passive_preload_settings(fresh_config)
    if settings is None:
        return None
    try:
        if (type(receipt) is not dict or receipt.get('schema') != MODE or receipt.get('mode') != MODE
                or receipt.get('joint_path') != JOINT_PATH
                or receipt.get('changed_attributes') != [TARGET_ATTRIBUTE_PATH]
                or receipt.get('nominal_preload_N') != 1.5 or receipt.get('explicit_effort_N') != -1.5
                or receipt.get('other_stage_attributes_unchanged') is not True
                or receipt.get('source_sha256') != file_sha256(__file__)
                or receipt.get('protected_before_sha256') != receipt.get('protected_after_sha256')):
            raise ValueError('mismatched fields')
        checksum = receipt.get('protected_before_sha256')
        if type(checksum) is not str or len(checksum) != 64 or any(c not in '0123456789abcdef' for c in checksum):
            raise ValueError('missing protected-stage hash')
        _validate_drive_values(receipt['before'], -.0015)
        _validate_drive_values(receipt['after'], 0.)
        if {**receipt['before'], 'target_position_m': 0.} != receipt['after']:
            raise ValueError('extra drive edit')
    except (KeyError, TypeError, ValueError) as error:
        raise ValueError('Invalid passive preload authoring receipt: '+str(error)) from error
    native = {}
    if robot is not None:
        # Same read-only name/index/path mapping used by gravity_feedforward.
        paths = np.asarray(robot._articulation_view._dof_paths, dtype=object)
        mapped = robot.get_dof_index('PassiveCompression')
        integer = lambda value: isinstance(value, (int, np.integer)) and not isinstance(value, (bool, np.bool_))
        if (not integer(slider) or not integer(mapped) or slider < 0 or mapped != slider
                or paths.ndim != 2 or paths.shape[0] != 1 or slider >= paths.shape[1]
                or paths[0, slider] != JOINT_PATH):
            raise ValueError('Native passive slider binding differs from authored USD')
        native = dict(native_dof_index=int(slider), native_dof_path=paths[0, slider])
    elif slider is not None:
        raise ValueError('Native passive slider binding requires its robot')
    _, values = _read_drive(stage, 0.)
    if values != receipt['after']:
        raise ValueError('Passive drive readback differs from authoring receipt')
    return dict(mode=MODE, nominal_preload_N=1.5, explicit_effort_N=-1.5, drive=values,
        scope='saved_or_preinitialization_usd' if robot is None else 'live_usd_after_native_initialization',
        read_only_usd=True, applied_effort_measured=False, physical_validation=False,
        training_eligible=False, **native)
