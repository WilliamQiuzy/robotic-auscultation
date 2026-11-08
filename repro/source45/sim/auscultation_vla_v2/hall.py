"""Hall sensor-domain model and shared real/sim ADC decoding.

The simulator consumes slider displacement, never contact-force ground truth.
Only ADC packets cross the observation boundary. The returned N value is the
existing calibration's spring-load estimate, not measured patient pressure.
"""
from collections import deque
from dataclasses import dataclass
from functools import lru_cache
import hashlib
import json
import math
from pathlib import Path

import numpy as np

from auscultation.mechanical_sensor.force_feedback.force_sensor import ForceCalibration


@dataclass(frozen=True)
class HallPacket:
    raw: int
    sample_time_s: float  # acquisition time mapped to the consumer monotonic clock
    received_time_s: float
    calibration_sha256: str
    source: str  # real_adc or simulated_adc; never used as a policy feature


@dataclass(frozen=True)
class HallObservation:
    hall_voltage_v: float
    spring_force_estimate_n: float
    displacement_estimate_m: float
    status: str
    sample_time_s: float
    calibration_sha256: str


class HallCalibration:
    def __init__(self, raw_json):
        self.sha256 = hashlib.sha256(raw_json).hexdigest()
        self._host = ForceCalibration.from_bytes(raw_json)
        cfg = json.loads(raw_json)
        # calibrated:true alone does NOT prove measurement against a force gauge.
        basis = cfg.get('calibration_basis', {})
        self.bench_verified = basis.get('bench_verified') is True and basis.get('not_bench_pressed') is not True
        self.calibrated_flag = self._host.calibrated
        self.supply_v = self._host.supply_v
        self.adc_full_scale = self._host.adc_full_scale
        order = np.argsort(self._host.displacements)
        self._compression_m = np.asarray(self._host.displacements)[order] / 1000.
        self._volts = np.asarray(self._host.volts)[order]
        if self._compression_m[0] != 0 or np.any(np.diff(self._compression_m) <= 0):
            raise ValueError('Inverse Hall mapping needs strictly increasing displacement from zero')
        self.travel_m = float(self._compression_m[-1])

    @classmethod
    def from_path(cls, path):
        return cls(Path(path).read_bytes())

    def voltage_at_displacement(self, compression_m):
        if not np.isfinite(compression_m) or not 0 <= compression_m <= self.travel_m:
            raise ValueError('Slider outside calibrated mechanical travel')
        return float(np.interp(compression_m, self._compression_m, self._volts))

    def decode(self, packet, *, now_s, max_age_s=.05):
        if not isinstance(packet, HallPacket):
            raise ValueError('Missing Hall ADC packet')
        if not np.isfinite([now_s, max_age_s, packet.sample_time_s, packet.received_time_s]).all() or max_age_s <= 0:
            raise ValueError('Invalid Hall clock or freshness budget')
        if not 0 <= packet.sample_time_s <= packet.received_time_s <= now_s + 1e-9:
            raise ValueError('Invalid Hall timestamp order')
        if now_s - packet.sample_time_s > max_age_s + 1e-9:
            raise ValueError('Hall sample is stale')
        if packet.source not in {'real_adc', 'simulated_adc'}:
            raise ValueError('Hall must originate from an ADC packet')
        if packet.calibration_sha256 != self.sha256:
            raise ValueError('Hall calibration identity mismatch')
        if type(packet.raw) is not int or not 0 < packet.raw < self.adc_full_scale:
            raise ValueError('Invalid or saturated Hall ADC')
        volts = self._host.raw_to_volts(packet.raw)
        # Small engineering margin for noise at tare/travel endpoints; not a
        # clinical limit or evidence of sensor accuracy outside calibration.
        if not min(self._volts) - .005 <= volts <= max(self._volts) + .005:
            raise ValueError('Hall voltage outside calibration domain')
        force = self._host.volts_to_force(volts)
        displacement = max(0., self._host.volts_to_displacement(volts) / 1000.)
        return HallObservation(volts, force, displacement, self._host.classify(force),
                               packet.sample_time_s, self.sha256)


@lru_cache(maxsize=1)
def default_calibration():
    """Pinned source snapshot; never hot-reload calibration mid-episode."""
    return HallCalibration.from_path(Path(__file__).with_name('calibration') / 'veo6_tared_snapshot.json')


class HallSimulator:
    """Forward sensor model; all perturbations are explicit, unmeasured settings.

    RP2040: quantize to 12 bits, scale each read to u16, average eight reads as
    in the repository's MicroPython firmware. Analog input is held constant
    during those eight subreads. No ADC INL, magnetic hysteresis or thermal
    dynamics have been measured/modelled. Delay/dropout retain acquisition time.
    """
    def __init__(self, calibration, *, sample_period_s=.01, noise_std_v=0.,
                 voltage_bias_v=0., drift_v_s=0., latency_s=0., curve_warp=0., seed=0):
        values = [sample_period_s, noise_std_v, voltage_bias_v, drift_v_s, latency_s, curve_warp]
        if (not np.isfinite(values).all() or sample_period_s <= 0 or noise_std_v < 0
                or latency_s < 0 or abs(curve_warp) > .5 or type(seed) is not int or seed < 0):
            raise ValueError('Invalid Hall simulator configuration')
        if calibration.adc_full_scale != 65535:
            raise ValueError('This simulator implements RP2040 read_u16 packets only')
        self.calibration = calibration
        self.period = sample_period_s
        self.noise_std_v = noise_std_v
        self.bias_v = voltage_bias_v
        self.drift_v_s = drift_v_s
        self.latency = latency_s
        self.curve_warp = curve_warp
        self._rng = np.random.default_rng(seed)
        self._last_call = -math.inf
        self._next_sample = -math.inf
        self._start = None
        self._pending = deque()
        self._last_packet = None

    def _clock(self, now):
        if not np.isfinite(now) or now < 0 or now < self._last_call - 1e-9:
            raise ValueError('Invalid or reversed Hall simulator time')
        self._last_call = now

    def update(self, time_s, compression_m, *, drop=False):
        self._clock(time_s)
        self.calibration.voltage_at_displacement(compression_m)  # bounds, even between samples
        if type(drop) is not bool:
            raise ValueError('Dropout flag must be explicit bool')
        if self._start is None:
            self._start = time_s
        if time_s + 1e-9 < self._next_sample:
            return
        # Missed sample periods are not backfilled with a future displacement.
        # Preserve the configured phase when the physics dt does not divide the
        # sensor period. Timestamp the actual available physics state, not an
        # invented earlier sample; skip missed periods without replaying them.
        periods = math.floor((time_s-self._start+1e-9) / self.period) + 1
        self._next_sample = self._start + periods * self.period
        if drop:
            return
        u = compression_m / self.calibration.travel_m
        warped = (u + self.curve_warp * u * (1-u)) * self.calibration.travel_m
        nominal_v = self.calibration.voltage_at_displacement(warped)
        analog = nominal_v + self.bias_v + self.drift_v_s * (time_s-self._start)
        analog = analog + self._rng.normal(0., self.noise_std_v, size=8)
        adc12 = np.rint(np.clip(analog / self.calibration.supply_v, 0., 1.) * 4095).astype(np.int64)
        u16 = (adc12 << 4) | (adc12 >> 8)
        packet = HallPacket(int(u16.sum() // 8), time_s, time_s+self.latency,
                            self.calibration.sha256, 'simulated_adc')
        self._pending.append(packet)

    def read(self, now_s):
        self._clock(now_s)
        while self._pending and self._pending[0].received_time_s <= now_s + 1e-9:
            self._last_packet = self._pending.popleft()
        return self._last_packet
