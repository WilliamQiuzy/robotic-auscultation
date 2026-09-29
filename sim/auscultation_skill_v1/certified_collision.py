"""Optional temporal distance certificates over the unchanged mesh policy.

For a fixed local mesh with ||x|| <= radius, moving its rigid transform from
(R0,t0) to (R1,t1) moves every point by at most
    ||t1-t0|| + radius * ||R1-R0||_F.
The triangle inequality therefore bounds the decrease of the separation of
two meshes by the sum of these displacements. The radius below encloses the
original local AABB, hence every original vertex and triangle. Certificates
use native-object transforms read after the inherited transform sync.

All original pairs still receive their AABB test on every check. Near-object
sync, contact exemptions, native FCL options, pair order and reported distance
floor are unchanged. Only a pair with a sufficient conservative lower bound
can omit a repeated distance query. Collision/uncertain distances never create
certificates. The separate static-containment installer must still wrap check;
certificates say nothing about solid containment or continuous swept motion.

Geometry is frozen. Replacing a native object invalidates its certificates
automatically; the original static-envelope installer also refreshes the
inherited bounds/corners and is supported. Other geometry, scale or local-frame
changes require a rebuilt checker, or a complete refresh of the original
native/broadphase geometry followed by invalidation. Cache invalidation alone
does not repair inherited geometry metadata or dynamic corners. The numerical
guard is an engineering cushion, not a formal error bound for native FCL.
Invalid query inputs, geometry bounds, native transforms or native distance
results raise an error instead of inheriting the parent's NaN comparison path.
Selection by an episode runner requires fresh source-bound qualification and
retention of the original static-envelope and containment checks.
"""
from dataclasses import dataclass

import fcl
import numpy as np

from sim.curobo_expert_v1.fast_collision import FastCollisionChecker


MINIMUM_NUMERIC_GUARD_M = 1e-8


def distance_lower_bound(distance, old_rotations, old_translations, rotations, translations, radii,
                         numeric_guard_m=MINIMUM_NUMERIC_GUARD_M):
    if not np.isfinite(numeric_guard_m) or numeric_guard_m < MINIMUM_NUMERIC_GUARD_M:
        raise ValueError('distance-certificate guard must be at least 1e-8 m')
    old_r, old_t, new_r, new_t, radius = map(np.asarray,
        (old_rotations, old_translations, rotations, translations, radii))
    if (not np.isfinite(distance) or distance < 0. or old_r.shape != (2, 3, 3)
            or new_r.shape != (2, 3, 3) or old_t.shape != (2, 3) or new_t.shape != (2, 3)
            or radius.shape != (2,) or np.any(radius < 0.)
            or not all(np.isfinite(x).all() for x in (old_r, old_t, new_r, new_t, radius))):
        return None
    displacement = np.linalg.norm(new_t-old_t, axis=1) + radius*np.linalg.norm(new_r-old_r, axis=(1, 2))
    lower = float(distance - displacement.sum() - numeric_guard_m)
    return lower if np.isfinite(lower) else None


def distance_lower_bounds(distances, old_rotations, old_translations, rotations, translations, radii,
                          numeric_guard_m=MINIMUM_NUMERIC_GUARD_M):
    """The same proof in a batch; NaN means a record cannot certify anything."""
    if not np.isfinite(numeric_guard_m) or numeric_guard_m < MINIMUM_NUMERIC_GUARD_M:
        raise ValueError('distance-certificate guard must be at least 1e-8 m')
    d, old_r, old_t, new_r, new_t, radius = map(np.asarray,
        (distances, old_rotations, old_translations, rotations, translations, radii))
    count = len(d) if d.ndim == 1 else -1
    if (old_r.shape != (count, 2, 3, 3) or new_r.shape != old_r.shape
            or old_t.shape != (count, 2, 3) or new_t.shape != old_t.shape or radius.shape != (count, 2)):
        raise ValueError('invalid distance certificate batch shape')
    valid = np.isfinite(d) & (d >= 0.) & np.isfinite(radius).all(axis=1) & (radius >= 0.).all(axis=1)
    valid &= np.isfinite(old_r).all(axis=(1, 2, 3)) & np.isfinite(new_r).all(axis=(1, 2, 3))
    valid &= np.isfinite(old_t).all(axis=(1, 2)) & np.isfinite(new_t).all(axis=(1, 2))
    with np.errstate(over='ignore', invalid='ignore'):
        displacement = np.linalg.norm(new_t-old_t, axis=2) + radius*np.linalg.norm(new_r-old_r, axis=(2, 3))
        lower = d-displacement.sum(axis=1)-numeric_guard_m
    return np.where(valid & np.isfinite(lower), lower, np.nan)


@dataclass(frozen=True)
class _Certificate:
    # Strong references prevent stale Python object id reuse.
    objects: tuple
    distance: float
    rotations: np.ndarray
    translations: np.ndarray
    radii: np.ndarray


class CertifiedCollisionChecker(FastCollisionChecker):
    def __init__(self, *args, numeric_guard_m=MINIMUM_NUMERIC_GUARD_M, **kwargs):
        if not np.isfinite(numeric_guard_m) or numeric_guard_m < MINIMUM_NUMERIC_GUARD_M:
            raise ValueError('distance-certificate guard must be at least 1e-8 m')
        super().__init__(*args, **kwargs)
        self.numeric_guard_m = float(numeric_guard_m)
        self._certificates = {}
        self._radii = {}
        self.last_certificate_reuses = self.last_exact_distance_calls = self.last_invalidated_certificates = 0
        self.total_certificate_reuses = self.total_exact_distance_calls = 0

    def invalidate_certificates(self):
        """Drop cached distances/radii; does not rebuild native or broadphase geometry."""
        self._certificates.clear()
        self._radii.clear()

    def _radius(self, index):
        obj = self.objects[index]
        cached = self._radii.get(index)
        if cached is None or cached[0] is not obj:
            corners = np.asarray(self.local_corners[index], float)
            radius = (float(np.linalg.norm(corners, axis=1).max())
                      if corners.ndim == 2 and corners.shape[1] == 3 and len(corners) and np.isfinite(corners).all()
                      else float('nan'))
            self._radii[index] = (obj, radius)
        else:
            radius = cached[1]
        return radius

    def _native_poses(self, indices):
        # Fetch each native object once after the inherited sync, then perform
        # the same finite/shape/rigidity checks once as a batch. No checks are
        # omitted for cached distances.
        rotations = (np.asarray([self.objects[i].getRotation() for i in indices], float)
                     if len(indices) else np.empty((0, 3, 3)))
        translations = (np.asarray([self.objects[i].getTranslation() for i in indices], float)
                        if len(indices) else np.empty((0, 3)))
        radii = np.asarray([self._radius(i) for i in indices])
        if (rotations.shape != (len(indices), 3, 3) or translations.shape != (len(indices), 3)
                or not np.isfinite(rotations).all() or not np.isfinite(translations).all()
                or not np.isfinite(radii).all() or np.any(radii < 0.)
                or np.any(np.abs(rotations.transpose(0, 2, 1)@rotations-np.eye(3)) > 1e-7)
                or np.any(np.abs(np.linalg.det(rotations)-1.) > 1e-7)):
            raise ValueError('invalid native collision transform or mesh radius')
        return rotations, translations, radii

    def check(self, q, clearance=.003, allow_patient_contact=False, compression=0., details=False):
        q = np.asarray(q, float)
        if (q.shape != (7,) or not np.isfinite(q).all() or not np.isscalar(clearance)
                or not np.isscalar(compression) or not np.isfinite([clearance, compression]).all() or clearance <= 0.):
            raise ValueError('invalid collision query')
        self._update_bounds(q, compression)
        if (self.bounds.shape != (len(self.entries), 2, 3) or not np.isfinite(self.bounds).all()
                or np.any(self.bounds[:, 0] > self.bounds[:, 1])):
            raise ValueError('invalid collision bounds')
        ia, ib = self._pair_a, self._pair_b
        gap = np.maximum(0., np.maximum(self.bounds[ia, 0]-self.bounds[ib, 1],
                                       self.bounds[ib, 0]-self.bounds[ia, 1]))
        near = np.linalg.norm(gap, axis=1) < clearance+1e-9
        near_pairs = self.pair_indices[near]
        self.last_checked_aabb_pair_count = len(self.pairs)
        self.last_near_pair_count = len(near_pairs)
        self._sync_near_objects(near_pairs)
        self.last_certificate_reuses = self.last_exact_distance_calls = self.last_invalidated_certificates = 0
        exempt = []
        for i, j in near_pairs:
            a, b = self.entries[i], self.entries[j]
            contact_face = (a['path'].endswith('/Diaphragm') or '/RetentionLip/' in a['path']
                            or '/ContactRim/' in a['path'])
            exempt.append(bool(allow_patient_contact and contact_face and b['path'].startswith('/World/Patient/')))
        active = near_pairs[~np.asarray(exempt, dtype=bool)]
        indices = np.unique(active)
        rotations, translations, radii = self._native_poses(indices)
        slots = np.full(len(self.entries), -1, dtype=int); slots[indices] = np.arange(len(indices))
        pair_slots = slots[active]
        records, record_indices = [], []
        for index, (i, j) in enumerate(active):
            key = (int(i), int(j)); record = self._certificates.get(key)
            if record is not None and (record.objects[0] is not self.objects[i] or record.objects[1] is not self.objects[j]):
                self._certificates.pop(key)
                self.last_invalidated_certificates += 1
                record = None
            if record is not None:
                records.append(record); record_indices.append(index)
        reusable = np.zeros(len(active), dtype=bool)
        if records:
            selected = pair_slots[record_indices]
            lower = distance_lower_bounds(np.array([r.distance for r in records]),
                np.array([r.rotations for r in records]), np.array([r.translations for r in records]),
                rotations[selected], translations[selected], np.array([r.radii for r in records]), self.numeric_guard_m)
            reusable[record_indices] = np.isfinite(lower) & (lower >= clearance+self.numeric_guard_m)
        worst = {'clear': True, 'minimum_distance_m': clearance, 'pair': None}
        # Active pairs retain their original relative order. The omitted ones
        # are exactly the unchanged contact exemptions evaluated above.
        for index, (i, j) in enumerate(active):
            a, b = self.entries[i], self.entries[j]
            key = (int(i), int(j)); objects = (self.objects[i], self.objects[j])
            if reusable[index]:
                self.last_certificate_reuses += 1
                self.total_certificate_reuses += 1
                continue
            result = fcl.DistanceResult()
            distance = float(fcl.distance(objects[0], objects[1], fcl.DistanceRequest(), result))
            self.last_exact_distance_calls += 1
            self.total_exact_distance_calls += 1
            if not np.isfinite(distance):
                self._certificates.pop(key, None)
                raise ValueError('nonfinite native FCL distance')
            # Use the exact native result, never a previously propagated lower
            # bound, as a new certificate's reference distance.
            # These native transforms/radii already passed all batch checks;
            # the reference displacement at this exact query is zero.
            if distance-self.numeric_guard_m >= clearance+self.numeric_guard_m:
                selected = pair_slots[index]
                snapshots = [values[selected].copy() for values in (rotations, translations, radii)]
                for value in snapshots:
                    value.setflags(write=False)
                self._certificates[key] = _Certificate(objects, distance, *snapshots)
            else:
                self._certificates.pop(key, None)
            if distance < worst['minimum_distance_m']:
                worst.update(minimum_distance_m=distance, pair=[a['path'], b['path']])
            if distance < clearance:
                worst['clear'] = False
                if not details:
                    return worst
        return worst
