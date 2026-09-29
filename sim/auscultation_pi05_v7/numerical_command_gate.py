"""Separate, fail-closed binary64 representation audit for a jerk comparison.

Nominal jerk stays 8. No command, Ruckig state, physical gate or checkpoint is
changed. A raw rejection may pass this comparison only with an exact constant
cubic certificate and a four-position half-ULP error budget.
"""
from fractions import Fraction
import hashlib
import json
import math
from pathlib import Path
import numpy as np
from sim.auscultation_skill_v1.executor import command_derivatives, DT

VERSION='constant_cubic_half_ulp_jerk_comparison_v1'
CERTIFICATE_SCHEMA='wam_constant_jerk_command_certificate_v1'
NOMINAL_JERK=8.
ORIGINAL_SLACK=1e-8
COEFFICIENTS=(-1,3,-3,1)


def source_sha256():return hashlib.sha256(Path(__file__).read_bytes()).hexdigest()


def numerical_gate_contract():
    return dict(schema='wam_numerical_command_gate_contract_v1',version=VERSION,source_sha256=source_sha256(),
        nominal_jerk_rad_s3=NOMINAL_JERK,original_raw_slack=ORIGINAL_SLACK,dt_s=DT,
        position_representation='IEEE754 binary64',budget='sum(abs([-1,3,-3,1])*0.5*ulp(q))/float64(dt**3)',
        numerical_acceptance='raw velocity and acceleration pass; same calculation and constant cubic; exact polynomial jerk <= 8; each output within half ULP; exact stored-q jerk <= 8 + representation budget',
        unsupported_boundary='reject raw failures across calculation or cubic boundaries; no retry or fallback',
        action_semantics='absolute_joint_command_v1',original_ruckig_max_jerk_unchanged=True,
        physical_safety_gates_unchanged=True,raw_values_and_failures_retained=True)


def contract_sha256():
    return hashlib.sha256(json.dumps(numerical_gate_contract(),sort_keys=True,separators=(',',':'),allow_nan=False).encode()).hexdigest()


def _f(value):
    if isinstance(value,(bool,np.bool_)) or not np.isfinite(value):raise ValueError('Nonfinite or boolean numeric value')
    return Fraction.from_float(float(value))


def _section_value(section,key):
    value=section.get(key)
    if not isinstance(value,list) or len(value)!=1:raise ValueError('Certificate must contain one constant cubic section')
    return _f(value[0])


def _poly(p,v,a,j,t):return p+v*t+a*t*t/2+j*t*t*t/6


def _stencil(values,denominator):return sum((c*x for c,x in zip(COEFFICIENTS,values)),Fraction())/denominator


def _validate_samples(certificate,positions):
    if not isinstance(certificate,dict) or certificate.get('schema')!=CERTIFICATE_SCHEMA:raise ValueError('Missing constant cubic certificate')
    samples=certificate.get('samples')
    if not isinstance(samples,list) or len(samples)!=4:raise ValueError('Four certificate samples required')
    rows=[s.get('physics_row') for s in samples];calculations=[s.get('calculation_id') for s in samples]
    if any(type(x) is not int or x<0 for x in rows+calculations):raise ValueError('Invalid certificate row/calculation identity')
    if rows!=list(range(rows[0],rows[0]+4)) or len(set(calculations))!=1:raise ValueError('Certificate crosses row or calculation boundary')
    times=[float(_f(s.get('trajectory_time_s'))) for s in samples]
    if any(b!=a+DT for a,b in zip(times,times[1:])):raise ValueError('Certificate clock differs from retained Ruckig clock')
    recorded=np.asarray([s.get('command') for s in samples],float)
    if recorded.shape!=(4,7) or not np.array_equal(recorded,positions):raise ValueError('Certificate command/history mismatch')
    return times


def _proof(axis,positions,times,section,denominator,budget):
    if not isinstance(section,dict):raise ValueError('Missing affected-axis cubic section')
    p,v,a,j=(_section_value(section,key) for key in ('p','v','a','j'))
    if abs(j)>Fraction(8):raise ValueError('Certificate analytic jerk exceeds nominal 8')
    start=_f(section.get('start_s'));end=_f(section.get('end_s'))
    ts=[_f(t) for t in times]
    if not start<ts[0]<ts[-1]<end:raise ValueError('Certificate crosses a cubic boundary')
    polynomial=[_poly(p,v,a,j,t-start) for t in ts]
    actual=[_f(x) for x in positions[:,axis]]
    half_ulps=[_f(math.ulp(float(x)))/2 for x in positions[:,axis]]
    errors=[q-e for q,e in zip(actual,polynomial)]
    if any(abs(e)>u for e,u in zip(errors,half_ulps)):raise ValueError('Output is not within half ULP of exact cubic')
    polynomial_jerk=_stencil(polynomial,denominator)
    if abs(polynomial_jerk)>Fraction(8):raise ValueError('Exact polynomial sampled jerk exceeds nominal 8')
    sampled_jerk=_stencil(actual,denominator)
    if abs(sampled_jerk)>Fraction(8)+budget:raise ValueError('Exact stored-q jerk exceeds bounded representation budget')
    return dict(analytic_jerk_rad_s3=float(j),exact_polynomial_sampled_jerk_rad_s3=float(polynomial_jerk),
        exact_stored_q_sampled_jerk_rad_s3=float(sampled_jerk),position_errors_within_half_ulp=True,
        position_errors_in_ulp=[float(e/(2*u)) for e,u in zip(errors,half_ulps)],
        exact_jerk_budget_fraction=[str(budget.numerator),str(budget.denominator)])


def compare_command(history,command,*,velocity,acceleration,certificate=None):
    """Pure comparison. Malformed inputs raise; missing proof never overrides."""
    if np.asarray(history).dtype.kind=='b' or np.asarray(command).dtype.kind=='b':raise ValueError('Boolean positions are not commands')
    history=np.asarray(history,float);command=np.asarray(command,float)
    raw=command_derivatives(history,command)
    if not np.isfinite([velocity,acceleration]).all() or min(velocity,acceleration)<=0:raise ValueError('Invalid original comparison bounds')
    bounds=dict(velocity=float(velocity),acceleration=float(acceleration),jerk=NOMINAL_JERK)
    passed=all(raw[k]<=bounds[k]+ORIGINAL_SLACK for k in bounds)
    result=dict(schema='wam_numerical_command_comparison_v1',version=VERSION,passed=passed,raw_passed=passed,
        intervened=False,raw_peaks=raw,original_bounds=bounds,original_slack=ORIGINAL_SLACK,
        reason='original comparison passed' if passed else 'original comparison rejected',proofs={})
    if passed:return result
    if raw['velocity']>bounds['velocity']+ORIGINAL_SLACK or raw['acceleration']>bounds['acceleration']+ORIGINAL_SLACK:
        result['reason']='velocity or acceleration comparison cannot be overridden';return result
    positions=np.vstack([history,command]);denominator=_f(DT**3)
    budgets=[sum((abs(c)*_f(math.ulp(float(q)))/2 for c,q in zip(COEFFICIENTS,positions[:,axis])),Fraction())/denominator for axis in range(7)]
    result['jerk_representation_budget_rad_s3']=[float(x) for x in budgets]
    signed=np.diff(positions,n=3,axis=0)[0]/DT**3
    axes=np.flatnonzero(abs(signed)>NOMINAL_JERK+ORIGINAL_SLACK)
    try:
        times=_validate_samples(certificate,positions)
        for axis in axes:
            result['proofs'][str(axis)]=_proof(int(axis),positions,times,certificate.get('axis_sections',{}).get(str(axis)),denominator,budgets[axis])
    except (ValueError,TypeError,KeyError,OverflowError) as error:
        result['reason']=str(error);return result
    result.update(passed=True,intervened=True,reason='raw jerk rejection explained by certified half-ULP position representation')
    return result


class CertificateTracker:
    """Bounded four-command provenance; does not drive or modify a trajectory."""
    def __init__(self):
        self.samples=[];self.calculation_id=0;self.trajectory=None

    def observe(self,trajectory,command,physics_row,subphase):
        command=np.asarray(command,float)
        if (trajectory is None or not subphase.startswith('learned_')
                or not np.array_equal(command,np.asarray(trajectory.output.new_position,float))):
            self.samples=[];self.trajectory=None;return
        if self.trajectory is not trajectory:
            self.samples=[];self.calculation_id+=1;self.trajectory=trajectory
        if trajectory.output.new_calculation:self.calculation_id+=1
        self.samples.append(dict(physics_row=int(physics_row),calculation_id=self.calculation_id,
            trajectory_time_s=float(trajectory.output.time),command=command.tolist()))
        self.samples=self.samples[-4:]

    def certificate(self):
        if self.trajectory is None or len(self.samples)!=4:return None
        times=[r['trajectory_time_s'] for r in self.samples];sections={}
        profiles=self.trajectory.output.trajectory.profiles
        if len(profiles)!=1:return None
        for axis,profile in enumerate(profiles[0]):
            for index,end in enumerate(profile.t_sum):
                start=0. if index==0 else profile.t_sum[index-1]
                if start<times[0]<times[-1]<end:
                    sections[str(axis)]=dict(p=[profile.p[index]],v=[profile.v[index]],a=[profile.a[index]],
                        j=[profile.j[index]],start_s=float(start),end_s=float(end));break
        return dict(schema=CERTIFICATE_SCHEMA,samples=[dict(r) for r in self.samples],axis_sections=sections)
