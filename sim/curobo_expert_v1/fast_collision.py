"""Same triangle-mesh FCL policy with batched conservative broad-phase updates.

No pair, triangle, clearance, contact exemption, or FK model is removed. Every
original pair gets its AABB-distance test on every check. Only objects belonging
to a surviving near pair need a native FCL transform for exact mesh distance.
This improves CPU work per sample; it does not subsample the requested motion.
"""
import numpy as np
import fcl

from sim.auscultation_expert_layout_v1.collision import CollisionChecker
from sim.auscultation_expert_layout_v1.scene import ROBOT, WRIST, FIXED, MOVING


class FastCollisionChecker(CollisionChecker):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self._dynamic_indices = np.asarray(self.dynamic, dtype=int)
        self._is_dynamic = np.zeros(len(self.entries), dtype=bool)
        self._is_dynamic[self._dynamic_indices] = True
        self._dynamic_corners = np.asarray(self.local_corners)[self._dynamic_indices]
        self._body_names = tuple(dict.fromkeys(self.entries[i]["body"] for i in self.dynamic))
        body_slot = {name:i for i,name in enumerate(self._body_names)}
        self._object_body_slot = np.full(len(self.entries), -1, dtype=int)
        self._object_body_slot[self._dynamic_indices] = [body_slot[self.entries[i]["body"]] for i in self.dynamic]
        self._dynamic_body_slot = self._object_body_slot[self._dynamic_indices]
        self._pair_a, self._pair_b = self.pair_indices.T
        self.last_updated_dynamic_objects = ()
        self.last_checked_aabb_pair_count = 0
        self.last_near_pair_count = 0

    def _update_bounds(self, q, compression=0.):
        frames = {ROBOT+"/"+name:self.base@transform
                  for name,transform in self.kinematics.link_transforms(q).items()}
        adapter = np.diag([1., -1., -1., 1.]); adapter[2, 3] = .085
        frames[FIXED] = frames[WRIST]@adapter
        slider = np.eye(4); slider[2, 3] = compression
        frames[MOVING] = frames[FIXED]@slider
        self._body_transforms = np.stack([frames[name] for name in self._body_names])
        matrices = self._body_transforms[self._dynamic_body_slot]
        points = (np.einsum("bij,bkj->bki", matrices[:, :3, :3], self._dynamic_corners)
                  + matrices[:, None, :3, 3])
        self.bounds[self._dynamic_indices, 0] = points.min(axis=1)
        self.bounds[self._dynamic_indices, 1] = points.max(axis=1)

    def _sync_near_objects(self, near_pairs):
        indices = np.unique(near_pairs)
        indices = indices[self._is_dynamic[indices]]
        transforms = {}
        for body in np.unique(self._object_body_slot[indices]):
            matrix = self._body_transforms[body]
            transforms[body] = fcl.Transform(matrix[:3, :3], matrix[:3, 3])
        for index in indices:
            self.objects[index].setTransform(transforms[self._object_body_slot[index]])
        self.last_updated_dynamic_objects = tuple(map(int, indices))

    def check(self, q, clearance=.003, allow_patient_contact=False, compression=0., details=False):
        self._update_bounds(q, compression)
        ia, ib = self._pair_a, self._pair_b
        gap = np.maximum(0., np.maximum(self.bounds[ia, 0]-self.bounds[ib, 1],
                                       self.bounds[ib, 0]-self.bounds[ia, 1]))
        near = np.linalg.norm(gap, axis=1) < clearance+1e-9
        near_pairs = self.pair_indices[near]
        self.last_checked_aabb_pair_count = len(self.pairs)
        self.last_near_pair_count = len(near_pairs)
        self._sync_near_objects(near_pairs)
        worst = {"clear":True, "minimum_distance_m":clearance, "pair":None}
        # Preserve the original pair order and early-return behavior exactly.
        for i,j in near_pairs:
            a,b = self.entries[i], self.entries[j]
            contact_face = (a["path"].endswith("/Diaphragm") or "/RetentionLip/" in a["path"]
                            or "/ContactRim/" in a["path"])
            if allow_patient_contact and contact_face and b["path"].startswith("/World/Patient/"):
                continue
            result = fcl.DistanceResult()
            distance = float(fcl.distance(self.objects[i], self.objects[j], fcl.DistanceRequest(), result))
            if distance < worst["minimum_distance_m"]:
                worst.update(minimum_distance_m=distance, pair=[a["path"], b["path"]])
            if distance < clearance:
                worst["clear"] = False
                if not details:
                    return worst
        return worst
