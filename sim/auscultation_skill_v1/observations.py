"""Actor allowlist and reproducible sensor conditions; no images are encoded."""
from collections import deque
import copy

import numpy as np

SCHEMA = "auscultation_scalar_observation_v1"
HISTORY_STEPS = 4
CONTROL_DT_S = .05
PHYSICS_DT_S = 1 / 240
PHYSICS_STEPS = 12
ACTION_LIMIT_M_S = .004
ACTION_INWARD_LIMIT_M_S = .002
# Required action-schema discriminator in every dataset/model/checkpoint.
# speed_scale affects inward motion; outward unloading stays at 4 mm/s.
ACTION_SCALING = "inward_scaled_outward_fixed_v1"
PHASES = ("APPROACH", "CONTACT", "HOLD", "RETRACT", "DONE", "ABORT")
SCALARS = ("gap_m", "hall_N", "compression_m", "normal_velocity_m_s",
           "previous_desired_normal_velocity_m_s", "executor_gap_m",
           "executor_velocity_m_s", "executor_acceleration_m_s2", "speed_scale")
FRAME_FEATURES = (SCALARS + tuple("q_"+str(i) for i in range(7))
                  + tuple("dq_"+str(i) for i in range(7))
                  + tuple("target_position_base_"+x for x in "xyz")
                  + tuple("normal_base_"+x for x in "xyz")
                  + tuple("target_rotation_r%dc%d" % (r,c) for c in range(2) for r in range(3))
                  + tuple("phase_"+phase for phase in PHASES))


def finite(value, shape, label):
    array = np.asarray(value, dtype=np.float64)
    if array.shape != shape or not np.isfinite(array).all():
        raise ValueError("invalid " + label)
    return array


def encode(observation):
    """Only the named physical observations enter the actor; metadata is ignored."""
    try:
        scalar = finite([observation[key] for key in SCALARS], (len(SCALARS),), "scalar observation")
        values = [scalar] + [finite(observation[key], (size,), key) for key,size in
                             (("q",7),("dq",7),("target_position_base_m",3),("normal_base",3))]
        rotation = finite(observation["target_rotation_base"], (3,3), "target rotation")
        phase = observation["phase"]
    except (KeyError, TypeError) as error:
        raise ValueError("missing actor observation: " + str(error)) from error
    if not .7 <= scalar[-1] <= 1.:
        raise ValueError("unsupported speed scale")
    if not np.isclose(np.linalg.norm(values[-1]),1.,atol=1e-6):
        raise ValueError("normal is not unit length")
    if not np.allclose(rotation.T@rotation,np.eye(3),atol=1e-6) or not np.isclose(np.linalg.det(rotation),1.,atol=1e-6):
        raise ValueError("invalid target rotation")
    if phase not in PHASES:
        raise ValueError("invalid actor phase")
    values.extend((rotation[:,:2].T.reshape(-1),np.array([phase==p for p in PHASES],float)))
    return np.concatenate(values).astype(np.float32)


class History:
    def __init__(self):
        self.reset()

    def reset(self):
        self.frames = deque(maxlen=HISTORY_STEPS)
        self.target = None

    def push(self, observation, target):
        if not isinstance(target,str) or not target:
            raise ValueError("missing target boundary")
        frame = encode(observation)
        if self.target != target:
            self.frames = deque((frame.copy() for _ in range(HISTORY_STEPS)),maxlen=HISTORY_STEPS)
            self.target = target
        else:
            self.frames.append(frame)
        return np.stack(self.frames)


class SensorCondition:
    """Apply once per 20 Hz decision to a COPY; never modify safety/physics truth.

    The complete measured state is delayed. Executor/phase values are assembled
    from the current controller separately. At a target boundary the queue is
    reset; warmup repeats the first real sample and preserves its timestamp.
    """
    def __init__(self, condition):
        self.condition = copy.deepcopy(condition)
        self.delay = condition.get("observation_delay_steps",0)
        self.sigma = float(condition.get("hall_noise_std_N",0.))
        self.clip = float(condition.get("hall_noise_clip_N",0.))
        seed = condition.get("seed",0)
        if (type(self.delay) is not int or not 0 <= self.delay <= 3
                or type(seed) is not int or not np.isfinite([self.sigma,self.clip]).all()
                or self.sigma < 0 or self.clip < 0 or (self.sigma == 0) != (self.clip == 0)):
            raise ValueError("invalid sensor condition")
        self.rng = np.random.default_rng(seed)
        self.queue = deque(maxlen=self.delay+1)
        self.target = None
        self.previous_time = None

    def apply(self, state, target):
        now = float(finite(state["time_s"],(),"sensor timestamp"))
        if self.previous_time is not None and now <= self.previous_time:
            raise ValueError("sensor timestamps must increase")
        if target != self.target:
            self.queue.clear(); self.target=target
        self.queue.append(copy.deepcopy(state))
        sample = copy.deepcopy(self.queue[0])
        sampled = float(finite(sample["time_s"],(),"delayed timestamp"))
        noise = float(np.clip(self.rng.normal(0.,self.sigma),-self.clip,self.clip)) if self.sigma else 0.
        sample["hall_N"] = float(finite(sample["hall_N"],(),"Hall measurement"))+noise
        self.previous_time = now
        return sample, {"observation_time_s":sampled, "hall_sample_time_s":sampled,
                        "hall_noise_N":noise, "observation_age_s":now-sampled,
                        "observation_delay_steps":self.delay}
