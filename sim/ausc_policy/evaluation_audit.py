"""Interpret frozen demonstration audits for an explicitly ineligible policy test.

The original reports and their force/geometry/trace/posture failures are retained.
Only the named dataset-eligibility metadata failures are separated, and only
after the controller reports actual completion. This never qualifies a dataset.
"""
import copy


def interpret(original, outcome):
    if (outcome.get('expert_demonstration') is not False
            or outcome.get('training_dataset_eligible') is not False
            or outcome.get('training_accepted') is not False
            or outcome.get('m0_passed') is not False):
        raise ValueError('Expected explicit diagnostic-only policy outcome')
    reports = copy.deepcopy(original['audits'])
    if set(reports) != {'mechanics','posture'}:
        raise ValueError('Both original mechanical and posture reports required')
    excluded = {'mechanics':[], 'posture':[]}
    if outcome.get('success') is True:
        completed = {p['target']+':controller_rejected' for p in outcome.get('points',[])
                     if p.get('success') is True and p.get('training_accepted') is False}
        eligible_reasons = {'mechanics':completed, 'posture':{'controller_did_not_accept'}}
    else:
        eligible_reasons = {'mechanics':set(), 'posture':set()}
    for key, report in reports.items():
        reasons = report.get('reasons')
        if not isinstance(reasons,list) or any(not isinstance(r,str) for r in reasons):
            raise ValueError('Malformed original audit reasons')
        excluded[key] = [r for r in reasons if r in eligible_reasons[key]]
        report['reasons'] = [r for r in reasons if r not in eligible_reasons[key]]
        # An unexplained original rejection never becomes a pass.
        report['evaluation_accepted'] = (not report['reasons'] and
            (report.get('accepted') is True or bool(excluded[key])))
    return dict(accepted=all(r['evaluation_accepted'] for r in reports.values()),
        original_demonstration_audit_accepted=original['accepted'],evaluation_audits=reports,
        separated_dataset_eligibility_reasons=excluded,training_dataset_eligible=False,
        rule='Only exact eligibility metadata reasons separated for successfully completed policy diagnostics; all physical/source/trace reasons retained')
