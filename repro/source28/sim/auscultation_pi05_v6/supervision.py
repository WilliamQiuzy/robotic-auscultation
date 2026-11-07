"""Contact-state interpretation of calibrated Hall travel, not a force gauge."""
import math


def contact_load_estimate(hall):
    """Preserve loaded spring estimate, distinguish unloaded spring preload.

    At free travel the calibration reports the internal 1.5 N preload; it
    cannot be interpreted as a patient load. The existing 0.1 mm engagement
    threshold is applied to measured Hall-estimated travel, never PhysX truth.
    This supervisor quantity does not replace the model's raw voltage/spring
    channels or the independent physical contact-force audit.
    """
    compression=float(hall.displacement_estimate_m);spring=float(hall.spring_force_estimate_n)
    if not math.isfinite(compression) or not math.isfinite(spring) or compression<0 or spring<0:
        raise ValueError('Invalid Hall estimate for contact supervision')
    return 0. if compression<.0001 else spring
