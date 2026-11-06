import pytest
from sim.ausc_policy import native_measurements as measure


def events(n=720):
    return [dict(physics_row=i,time_s=(i+1)/240,controller_after=dict(phase='HOLD'),
        measurements=dict(diaphragm_N=0.,retainer_N=4.,hall_N=3.98,compression_m=.002,
            forbidden_N=0.,frame_N=0.,lip_N=0.,contact_rim_N=0.,carriage_other_N=0.)) for i in range(n)]


def test_force_summary_preserves_physical_and_axial_channels_and_native_duration():
    got=measure.summarize(events())
    assert got['hold_seconds']==3.
    assert got['longest_hold_force_band_seconds']==3.
    assert got['hold_contact_force_N']==dict(minimum=4.,maximum=4.,mean=4.)
    assert got['scalar_peaks']['hall_N']==3.98
    assert got['scalar_peaks']['forbidden_N']==0.
    assert 'passed' not in got  # Reporting is not a replacement qualifier.
    assert len(got['plot_bins_20hz'])==60


def test_plot_bins_keep_native_extrema_and_do_not_hide_an_outlier():
    data=events();data[360]['measurements']['retainer_N']=5.
    got=measure.summarize(data)
    assert got['longest_hold_force_band_seconds']==1.5
    assert got['hold_contact_force_N']['maximum']==5.
    assert got['plot_bins_20hz'][30]['contact_max_N']==5.


def test_missing_reported_force_is_none_and_never_an_invented_zero():
    data=events(13)
    for row in data:row['measurements'].pop('diaphragm_N')
    got=measure.summarize(data)
    assert got['hold_contact_force_N'] is None
    assert got['plot_bins_20hz'][-1]['native_ticks']==1
    assert got['scalar_peaks']['diaphragm_N'] is None


def test_reporting_failure_cannot_override_an_original_physical_verdict(monkeypatch):
    monkeypatch.setattr(measure,'summarize',lambda events:(_ for _ in ()).throw(ValueError('report only')))
    checks=dict(unchanged_physical_gate_passed=True)
    measure.add_report(checks,events())
    assert checks['unchanged_physical_gate_passed'] is True
    assert checks['measured_contact'] is None
    assert 'report only' in checks['measurement_reporting_error']
