"""Reporting of already decoded native measurements; never a qualifier."""
import math

HZ=240
CHANNELS=('diaphragm_N','retainer_N','forbidden_N','frame_N','lip_N',
    'contact_rim_N','carriage_other_N','hall_N','compression_m')


def finite(value):
    return isinstance(value,(int,float)) and not isinstance(value,bool) and math.isfinite(value)


def values(rows,key):
    found=[r.get('measurements',{}).get(key) for r in rows]
    return found if found and all(finite(v) for v in found) else None


def contact(row):
    m=row.get('measurements',{});a=m.get('diaphragm_N');b=m.get('retainer_N')
    return a+b if finite(a) and finite(b) else None


def stats(values):
    if not values or not all(finite(v) for v in values):return None
    return dict(minimum=min(values),maximum=max(values),mean=sum(values)/len(values))


def summarize(events):
    hold=[e for e in events if e.get('controller_after',{}).get('phase')=='HOLD']
    count=longest=0
    for event in events:
        force=contact(event)
        good=(event.get('controller_after',{}).get('phase')=='HOLD' and finite(force) and 3.5<=force<=4.5)
        count=count+1 if good else 0;longest=max(longest,count)
    bins=[]
    for start in range(0,len(events),12):
        rows=events[start:start+12];forces=stats([contact(e) for e in rows]);hall=stats(values(rows,'hall_N'))
        phases={e.get('controller_after',{}).get('phase') for e in rows}
        bins.append(dict(time_start_s=rows[0]['time_s']-1/HZ,time_end_s=rows[-1]['time_s'],
            native_ticks=len(rows),phase=next(iter(phases)) if len(phases)==1 else 'TRANSITION',
            contact_min_N=None if forces is None else forces['minimum'],
            contact_max_N=None if forces is None else forces['maximum'],
            contact_mean_N=None if forces is None else forces['mean'],
            hall_min_N=None if hall is None else hall['minimum'],hall_max_N=None if hall is None else hall['maximum'],
            hall_mean_N=None if hall is None else hall['mean']))
    return dict(schema='ausc_native_measurements_v1',native_ticks=len(events),native_hz=HZ,
        simulation_seconds=len(events)/HZ,hold_ticks=len(hold),hold_seconds=len(hold)/HZ,
        hold_contact_force_N=stats([contact(e) for e in hold]),longest_hold_force_band_seconds=longest/HZ,
        scalar_peaks={k:(max(v) if (v:=values(events,k)) is not None else None) for k in CHANNELS},
        plot_bins_20hz=bins,interpretation='Reported diaphragm+retainer contact proxy and axial Hall kept separate. Native extrema retained in plot bins. Force-only duration is descriptive; original independent pose/force/release/home qualification remains authoritative.')


def add_report(checks,events):
    try:checks['measured_contact']=summarize(events)
    except Exception as error:
        checks['measured_contact']=None
        checks['measurement_reporting_error']=type(error).__name__+': '+str(error)
