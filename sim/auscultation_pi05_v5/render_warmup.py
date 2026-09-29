"""Bounded camera stabilization before initializing the RGB-D body reference.

Stage loading API: https://docs.omniverse.nvidia.com/kit/docs/omni.usd/1.12.2/omni.usd/omni.usd.UsdContext.html
No physics step or relaxed body-tracker threshold is permitted here.
"""
import hashlib
import json
import time
import numpy as np
from PIL import Image, ImageDraw


def stable_pixels(previous, current, mask):
    if (previous.shape != current.shape or current.dtype != np.uint8
            or current.ndim != 3 or current.shape[-1] != 3
            or mask.shape != current.shape[:2] or not mask.any()):
        raise ValueError('Invalid warmup camera samples')
    difference = np.abs(current.astype(float) - previous.astype(float))[mask]
    mean, p99 = float(np.mean(difference)), float(np.quantile(difference, .99))
    # Observed static RTX temporal noise is approximately 0.3--0.4 levels.
    # This gate stays well below the unchanged tracker limits (15/40 levels).
    return mean <= 1. and p99 <= 4., dict(mean_absolute_rgb_change=mean, p99_rgb_change=p99)


def warm_up_camera(sensor):
    import omni.usd
    context = omni.usd.get_context()
    start = time.monotonic()
    state = sensor.env.state()
    clock, q = float(sensor.env.world.current_time), state['q'].copy()
    previous = mask = None
    stable = 0
    records = []
    try:
        for index in range(400):
            if time.monotonic() - start > 40:
                raise ValueError('Camera warmup exceeded 40 seconds')
            sensor.rep.orchestrator.step(rt_subframes=2, delta_time=0., pause_timeline=False)
            if (abs(float(sensor.env.world.current_time)-clock)>1e-8
                    or np.max(abs(sensor.env.state()['q']-q))>1e-8):
                raise ValueError('Camera warmup advanced physics')
            rgb=np.asarray(sensor.annotators[0].get_data())[...,:3].copy()
            if mask is None:
                support=Image.new('1',(rgb.shape[1],rgb.shape[0]))
                ImageDraw.Draw(support).polygon([tuple(v) for v in sensor.polygon],fill=1)
                mask=np.asarray(support,dtype=bool)
            loading=context.get_stage_loading_status()
            busy=bool(context.get_stage_streaming_status())
            loaded=int(loading[1])>=int(loading[2]) and not busy
            ok,metric=(False,{}) if previous is None else stable_pixels(previous,rgb,mask)
            stable=stable+1 if ok and loaded else 0
            records.append(dict(index=index,loading=list(loading),streaming=busy,
                consecutive_stable=stable,rgb_sha256=hashlib.sha256(rgb.tobytes()).hexdigest(),**metric))
            previous=rgb
            if index>=20 and time.monotonic()-start>=5. and stable>=10:
                preparation=sensor.out.parent/'preparation'
                preparation.mkdir(exist_ok=True)
                np.savez_compressed(preparation/'render_warmup_last.npz',rgb=rgb)
                return
        raise ValueError('Camera did not stabilize within bounded warmup')
    finally:
        (sensor.out/'render_warmup.json').write_text(json.dumps(dict(
            scope='pre-observation render preparation; not a robot action or tracking recovery',
            physics_time_s=clock,physics_advanced=False,wall_seconds=time.monotonic()-start,
            frames=records),allow_nan=False)+'\n')
