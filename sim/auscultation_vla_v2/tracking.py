"""Fail-closed, operator-seeded tracking of a textured, locally rigid RGB-D patch.

This is a bounded local correspondence tracker, not an anatomical detector or
an articulated-body tracker. RGB and optical-Z depth must be registered to the
same calibrated, fixed camera. Times must share the consumer's monotonic clock.
No simulator object poses, target positions, segmentation IDs, or oracle masks
are accepted. ``depth_valid`` describes sensor validity, not patient membership.

Appearance, forward/backward flow, depth, and local rigidity gates reject common
occlusions instead of attaching to the new foreground surface. A visually and
geometrically indistinguishable replacement is not observable with this input;
this module alone is not a robot/patient semantic safety system. All failures
latch until an operator supplies a new seed with ``initialize``.
"""
import cv2
import numpy as np
from functools import wraps

from .observations import TargetEstimate, finite, rotation


class TrackingLost(ValueError):
    """No usable new estimate; stop consuming the old target and reinitialize."""


def _fail_closed(method):
    @wraps(method)
    def guarded(self,*args,**kwargs):
        try:return method(self,*args,**kwargs)
        except TrackingLost:raise
        except (ValueError,TypeError,OverflowError,FloatingPointError,
                np.linalg.LinAlgError,cv2.error) as error:
            self._reject('input or numerical tracking failure: '+type(error).__name__)
    return guarded


class SurfaceTracker:
    """Track one seeded surface using actual image/depth correspondences.

    ``initialize`` and ``update`` return a fresh ``TargetEstimate`` or raise
    ``TrackingLost``. Neither failure nor an unobserved frame refreshes a target
    timestamp. Revision is a planning epoch, incremented only on successful
    explicit reinitialization. The camera-facing outward normal comes from a
    robust local plane; its tangent is optical +X projected onto that plane
    (optical +Y fallback). It does not estimate anatomical in-plane orientation.

    Confidence is a bounded engineering quality score, NOT a calibrated
    probability of correctness. Defaults are conservative software-test values
    and require physical camera/motion/occlusion validation before deployment.
    """

    def __init__(self, intrinsics, camera_to_base, *, patch_radius=12,
                 search_radius=18, min_features=8, max_age_s=.15,
                 max_rgbd_skew_s=.02, max_frame_gap_s=.15,
                 min_texture_std=7., min_correlation=.8, uniqueness_margin=.08,
                 min_depth_fraction=.9, max_plane_error_m=.003,
                 max_speed_m_s=.6, max_normal_change_rad=.35):
        self._k = finite(intrinsics, (3, 3), 'intrinsics').copy()
        if (self._k[0, 0] <= 0 or self._k[1, 1] <= 0
                or not np.allclose(self._k[2], [0., 0., 1.], atol=1e-8, rtol=0)
                or abs(np.linalg.det(self._k)) < 1e-10):
            raise ValueError('invalid pinhole intrinsics')
        self._inverse_k = np.linalg.inv(self._k)
        self._transform = finite(camera_to_base, (4, 4), 'camera_to_base').copy()
        rotation(self._transform[:3, :3], 'camera_to_base')
        if not np.allclose(self._transform[3], [0., 0., 0., 1.], atol=1e-8, rtol=0):
            raise ValueError('invalid camera_to_base bottom row')
        for name, value, lower in [('patch_radius', patch_radius, 5),
                                   ('search_radius', search_radius, 4),
                                   ('min_features', min_features, 4)]:
            if type(value) is not int or value < lower:
                raise ValueError('invalid ' + name)
        if min_features > 64:
            raise ValueError('min_features exceeds feature budget')
        self.patch_radius, self.search_radius = patch_radius, search_radius
        self.min_features = min_features
        limits = dict(max_age_s=max_age_s, max_rgbd_skew_s=max_rgbd_skew_s,
                      max_frame_gap_s=max_frame_gap_s, min_texture_std=min_texture_std,
                      min_correlation=min_correlation, uniqueness_margin=uniqueness_margin,
                      min_depth_fraction=min_depth_fraction, max_plane_error_m=max_plane_error_m,
                      max_speed_m_s=max_speed_m_s, max_normal_change_rad=max_normal_change_rad)
        for name, value in limits.items():
            value = float(finite(value, (), name))
            if value < 0 or (name != 'max_rgbd_skew_s' and value == 0):
                raise ValueError('invalid ' + name)
            setattr(self, name, value)
        if not (0 < min_correlation < 1 and 0 < uniqueness_margin < 1
                and 0 < min_depth_fraction <= 1 and max_normal_change_rad < np.pi / 2):
            raise ValueError('invalid tracker quality limits')
        self._revision = -1
        self._lost = True
        self._loss_reason = 'not initialized; explicit initialization required'
        self._pixel = None

    @property
    def lost(self):
        return self._lost

    @property
    def loss_reason(self):
        return self._loss_reason

    @property
    def pixel_xy(self):
        """Last accepted pixel, for diagnostics only; unusable while ``lost``."""
        return None if self._pixel is None else self._pixel.copy()

    def _reject(self, reason):
        self._lost, self._loss_reason = True, reason
        raise TrackingLost(reason + '; explicit reinitialization required')

    def _frame(self, rgb, depth_m, depth_valid, rgb_time_s, depth_time_s, now_s):
        if (not isinstance(rgb, np.ndarray) or rgb.dtype != np.uint8
                or rgb.ndim != 3 or rgb.shape[2] != 3):
            self._reject('RGB must be uint8 HxWx3')
        try:
            depth, valid = np.asarray(depth_m), np.asarray(depth_valid)
        except (TypeError, ValueError):
            self._reject('invalid optical-Z metric depth or validity mask')
        if (depth.shape != rgb.shape[:2] or not np.issubdtype(depth.dtype, np.floating)
                or valid.shape != depth.shape or valid.dtype != bool):
            self._reject('invalid optical-Z metric depth or validity mask')
        if min(depth.shape) < 2 * self.patch_radius + 3:
            self._reject('frame too small for surface support')
        if not np.isfinite(depth[valid]).all() or np.any(depth[valid] <= 0):
            self._reject('nonpositive or nonfinite depth marked valid')
        try:
            rgb_time_s = float(finite(rgb_time_s, (), 'RGB timestamp'))
            depth_time_s = float(finite(depth_time_s, (), 'depth timestamp'))
            now_s = float(finite(now_s, (), 'consumer timestamp'))
        except (ValueError, TypeError):
            self._reject('invalid timestamp')
        if min(rgb_time_s, depth_time_s, now_s) < 0:
            self._reject('negative timestamp')
        if (max(rgb_time_s, depth_time_s) > now_s + 1e-9
                or now_s - min(rgb_time_s, depth_time_s) > self.max_age_s + 1e-9):
            self._reject('stale or future RGB-D frame')
        if abs(rgb_time_s - depth_time_s) > self.max_rgbd_skew_s + 1e-9:
            self._reject('RGB/depth timestamp skew')
        return cv2.cvtColor(rgb, cv2.COLOR_RGB2GRAY), depth, valid, rgb_time_s, depth_time_s

    def _patch(self, gray, pixel):
        radius = self.patch_radius
        x, y = pixel
        if (x < radius + 1 or y < radius + 1
                or x >= gray.shape[1] - radius - 1 or y >= gray.shape[0] - radius - 1):
            self._reject('surface patch outside image support')
        patch = cv2.getRectSubPix(gray, (2 * radius + 1,) * 2, tuple(map(float, pixel)))
        if np.std(patch) < self.min_texture_std:
            self._reject('textureless surface patch')
        return patch

    def _match(self, template, gray, around):
        """Bounded normalized matching both proposes motion and tests uniqueness."""
        radius, search = self.patch_radius, self.search_radius
        x, y = np.rint(around).astype(int)
        left, top = max(0, x - radius - search), max(0, y - radius - search)
        right = min(gray.shape[1], x + radius + search + 1)
        bottom = min(gray.shape[0], y + radius + search + 1)
        # Antialias the coarse integer-pixel proposal so subpixel image motion
        # does not appear as lost high-frequency texture. Flow and the original
        # unblurred seed appearance are checked separately after refinement.
        smooth = cv2.GaussianBlur(gray[top:bottom, left:right], (5, 5), .7)
        template = cv2.GaussianBlur(template, (5, 5), .7)
        scores = cv2.matchTemplate(smooth, template, cv2.TM_CCOEFF_NORMED)
        if not np.isfinite(scores).all():
            self._reject('nonfinite RGB matching score')
        _, peak, _, location = cv2.minMaxLoc(scores)
        px, py = location
        rivals = scores.copy()
        rivals[max(0, py - 3):py + 4, max(0, px - 3):px + 4] = -1.
        margin = peak - float(rivals.max())
        if peak < self.min_correlation or margin < self.uniqueness_margin:
            self._reject('weak or ambiguous RGB correspondence')
        pixel = np.array([left + px + radius, top + py + radius], dtype=float)
        if np.linalg.norm(pixel - around, ord=np.inf) >= search:
            self._reject('RGB correspondence reached search bound')
        return pixel, peak, margin

    def _features(self, gray, pixel):
        mask = np.zeros(gray.shape, np.uint8)
        cv2.circle(mask, tuple(np.rint(pixel).astype(int)), self.patch_radius - 2, 255, -1)
        features = cv2.goodFeaturesToTrack(gray, maxCorners=64, qualityLevel=.01,
                                          minDistance=3, mask=mask, blockSize=3)
        if features is None or len(features) < self.min_features:
            self._reject('insufficient independent RGB features')
        # Two-dimensional feature support rejects edges and thin line patterns.
        centered = features[:, 0] - np.mean(features[:, 0], axis=0)
        if np.linalg.eigvalsh(centered.T @ centered / len(features))[0] < 3.:
            self._reject('degenerate RGB feature geometry')
        return features

    def _rays(self, pixels):
        return np.column_stack((pixels, np.ones(len(pixels)))) @ self._inverse_k.T

    def _surface(self, depth, valid, pixel):
        x, y = np.rint(pixel).astype(int)
        radius = self.patch_radius
        if x < radius + 1 or y < radius + 1 or x >= depth.shape[1] - radius - 1 or y >= depth.shape[0] - radius - 1:
            self._reject('depth surface outside image support')
        local_valid = valid[y-radius:y+radius+1, x-radius:x+radius+1]
        center_valid = valid[y-1:y+2, x-1:x+2]
        if not valid[y, x] or np.mean(center_valid) < .75 or np.mean(local_valid) < self.min_depth_fraction:
            self._reject('missing local surface depth')
        yy, xx = np.mgrid[y-radius:y+radius+1, x-radius:x+radius+1]
        pixels = np.column_stack((xx[local_valid], yy[local_valid]))
        depths = depth[yy[local_valid], xx[local_valid]]
        points = self._rays(pixels) * depths[:, None]
        # Deterministic bounded RANSAC initialization avoids allowing a few
        # valid-but-wrong depths to rotate an ordinary least-squares plane.
        # The RNG is local; neither callers nor OpenCV global RNG are modified.
        generator = np.random.default_rng(0)
        inliers = np.zeros(len(points), bool)
        best_error = np.inf
        for _ in range(64):
            sample = points[generator.choice(len(points), 3, replace=False)]
            candidate_normal = np.cross(sample[1] - sample[0], sample[2] - sample[0])
            length = np.linalg.norm(candidate_normal)
            if length < 1e-7:
                continue
            candidate_normal /= length
            residual = np.abs((points - sample[0]) @ candidate_normal)
            candidate = residual <= self.max_plane_error_m
            median_error = np.median(residual)
            if (np.sum(candidate) > np.sum(inliers)
                    or (np.sum(candidate) == np.sum(inliers) and median_error < best_error)):
                inliers, best_error = candidate, median_error
        if np.mean(inliers) < .8:
            self._reject('inconsistent or occluded local depth surface')
        for _ in range(5):
            center = np.mean(points[inliers], axis=0)
            _, values, vh = np.linalg.svd(points[inliers] - center, full_matrices=False)
            if values[1] < 1e-5:
                self._reject('degenerate depth surface')
            normal = vh[-1]
            residual = np.abs((points - center) @ normal)
            median = np.median(residual)
            tolerance = min(self.max_plane_error_m, max(.0003, median * 3.5))
            candidate = residual <= tolerance
            if np.mean(candidate) < .8:
                self._reject('inconsistent or occluded local depth surface')
            if np.array_equal(inliers, candidate):
                break
            inliers = candidate
        center = np.mean(points[inliers], axis=0)
        _, _, vh = np.linalg.svd(points[inliers] - center, full_matrices=False)
        normal = vh[-1]
        if normal @ center > 0:
            normal = -normal
        ray = self._rays(pixel[None, :])[0]
        if -(normal @ ray) / np.linalg.norm(ray) < .25:
            self._reject('grazing or unobservable depth surface')
        position = ray * ((normal @ center) / (normal @ ray))
        median_center_depth = np.median(depth[y-1:y+2, x-1:x+2][center_valid])
        if abs(position[2] - median_center_depth) > self.max_plane_error_m:
            self._reject('seed depth disagrees with fitted surface')
        error = float(np.sqrt(np.mean(((points[inliers] - center) @ normal) ** 2)))
        return position, normal, float(np.mean(inliers)), error

    @staticmethod
    def _correlation(first, second):
        first = first.astype(float).ravel()
        second = second.astype(float).ravel()
        first -= first.mean()
        second -= second.mean()
        denominator = np.linalg.norm(first) * np.linalg.norm(second)
        return float(first @ second / denominator) if denominator > 1e-9 else -1.

    def _flow(self, gray, matched_pixel):
        previous = self._features(self._gray, self._pixel)
        guessed = previous + (matched_pixel - self._pixel).astype(np.float32)[None, None]
        settings = dict(winSize=(21, 21), maxLevel=2,
                        criteria=(cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 30, .01),
                        flags=cv2.OPTFLOW_USE_INITIAL_FLOW)
        current, forward_ok, _ = cv2.calcOpticalFlowPyrLK(self._gray, gray, previous, guessed, **settings)
        if current is None:
            self._reject('RGB optical flow failed')
        backward, backward_ok, _ = cv2.calcOpticalFlowPyrLK(gray, self._gray, current, previous.copy(), **settings)
        if backward is None:
            self._reject('backward RGB optical flow failed')
        fb = np.linalg.norm(backward[:, 0] - previous[:, 0], axis=1)
        accepted = forward_ok.ravel().astype(bool) & backward_ok.ravel().astype(bool) & (fb < .8)
        if np.sum(accepted) < self.min_features or np.mean(accepted) < .8:
            self._reject('inconsistent forward/backward RGB correspondences')
        before, after = previous[accepted, 0].astype(float), current[accepted, 0].astype(float)
        design = np.column_stack((before, np.ones(len(before))))
        retained = np.ones(len(before), dtype=bool)
        for _ in range(3):
            affine = np.linalg.lstsq(design[retained], after[retained], rcond=None)[0].T
            residuals = np.linalg.norm(design @ affine.T - after, axis=1)
            retained = residuals < .8
            if np.sum(retained) < self.min_features or np.mean(retained) < .8:
                self._reject('nonrigid or inconsistent local RGB flow')
        scales = np.linalg.svd(affine[:, :2], compute_uv=False)
        if np.linalg.det(affine[:, :2]) <= 0 or scales.min() < .8 or scales.max() > 1.25:
            self._reject('local appearance deformation exceeds tracker bounds')
        pixel = affine @ np.r_[self._pixel, 1.]
        if np.linalg.norm(pixel - matched_pixel) > 2.:
            self._reject('RGB flow and template correspondence disagree')
        return pixel, affine, before[retained], after[retained], float(np.median(fb[accepted]))

    def _rigidity(self, depth, valid, before, after):
        old_xy, new_xy = np.rint(before).astype(int), np.rint(after).astype(int)
        if (np.any(new_xy < 0) or np.any(new_xy[:, 0] >= depth.shape[1])
                or np.any(new_xy[:, 1] >= depth.shape[0])):
            self._reject('RGB correspondences outside depth support')
        support = self._valid[old_xy[:, 1], old_xy[:, 0]] & valid[new_xy[:, 1], new_xy[:, 0]]
        if np.sum(support) < self.min_features or np.mean(support) < .8:
            self._reject('missing depth at RGB correspondences')
        old_xy, new_xy = old_xy[support], new_xy[support]
        first = self._rays(before[support]) * self._depth[old_xy[:, 1], old_xy[:, 0], None]
        second = self._rays(after[support]) * depth[new_xy[:, 1], new_xy[:, 0], None]
        first_center, second_center = first.mean(axis=0), second.mean(axis=0)
        u, _, vh = np.linalg.svd((first - first_center).T @ (second - second_center))
        correction = np.eye(3)
        correction[2, 2] = np.linalg.det(vh.T @ u.T)
        fitted_rotation = vh.T @ correction @ u.T
        residual = np.linalg.norm((first - first_center) @ fitted_rotation.T - (second - second_center), axis=1)
        if np.quantile(residual, .9) > self.max_plane_error_m * 2:
            self._reject('inconsistent 3D surface correspondences or occlusion')

    def _estimate(self, position, normal, velocity, confidence, depth_time_s):
        tangent = np.array([1., 0., 0.]) - normal * normal[0]
        if np.linalg.norm(tangent) < 1e-6:
            tangent = np.array([0., 1., 0.]) - normal * normal[1]
        tangent /= np.linalg.norm(tangent)
        frame = np.column_stack((tangent, np.cross(normal, tangent), normal))
        rotate, translate = self._transform[:3, :3], self._transform[:3, 3]
        return TargetEstimate(rotate @ position + translate, rotate @ frame,
                              rotate @ velocity, float(confidence), depth_time_s,
                              self._revision, 'rgbd_surface_tracking')

    def _save(self, gray, depth, valid, pixel, position, normal, rgb_time_s, depth_time_s, now_s):
        self._gray, self._depth, self._valid = gray.copy(), depth.copy(), valid.copy()
        self._pixel, self._position, self._normal = pixel.copy(), position.copy(), normal.copy()
        self._rgb_time, self._depth_time = rgb_time_s, depth_time_s
        self._now = float(now_s)
        self._lost, self._loss_reason = False, None

    @_fail_closed
    def initialize(self, rgb, depth_m, depth_valid, pixel_xy, *, rgb_time_s, depth_time_s, now_s):
        """Explicit operator seed in (column, row); no automatic reacquisition."""
        self._lost = True
        gray, depth, valid, rgb_stamp, depth_stamp = self._frame(
            rgb, depth_m, depth_valid, rgb_time_s, depth_time_s, now_s)
        if self._revision >= 0 and (rgb_stamp < self._rgb_time or depth_stamp < self._depth_time
                                    or float(now_s) < self._now):
            self._reject('reversed timestamps during reinitialization')
        try:
            pixel = finite(pixel_xy, (2,), 'seed pixel').copy()
        except (ValueError, TypeError):
            self._reject('invalid operator seed pixel')
        patch = self._patch(gray, pixel)
        self._match(patch, gray, pixel)
        self._features(gray, pixel)
        position, normal, fraction, error = self._surface(depth, valid, pixel)
        self._anchor_patch = patch.copy()
        self._anchor_pixel = pixel.copy()
        self._anchor_to_current = np.eye(3)
        self._revision += 1
        confidence = .8 + .19 * fraction * (1 - .5 * min(1., error / self.max_plane_error_m))
        self._save(gray, depth, valid, pixel, position, normal, rgb_stamp, depth_stamp, now_s)
        return self._estimate(position, normal, np.zeros(3), confidence, depth_stamp)

    @_fail_closed
    def update(self, rgb, depth_m, depth_valid, *, rgb_time_s, depth_time_s, now_s):
        """Consume one newer synchronized frame; return only a newly observed target."""
        if self._lost:
            self._reject('tracker is lost or uninitialized')
        gray, depth, valid, rgb_stamp, depth_stamp = self._frame(
            rgb, depth_m, depth_valid, rgb_time_s, depth_time_s, now_s)
        if gray.shape != self._gray.shape:
            self._reject('camera image dimensions changed')
        if float(now_s) < self._now:
            self._reject('consumer clock reversed')
        delta = depth_stamp - self._depth_time
        if rgb_stamp <= self._rgb_time or delta <= 0:
            self._reject('duplicate or reversed RGB-D timestamps')
        if max(rgb_stamp - self._rgb_time, delta) > self.max_frame_gap_s + 1e-9:
            self._reject('RGB-D tracking frame gap exceeded')
        matched, correlation, _ = self._match(self._patch(self._gray, self._pixel), gray, self._pixel)
        pixel, affine, before, after, fb_error = self._flow(gray, matched)
        self._patch(gray, pixel)
        affine_h = np.vstack((affine, [0., 0., 1.]))
        cumulative = affine_h @ self._anchor_to_current
        radius = self.patch_radius
        yy, xx = np.mgrid[-radius:radius+1, -radius:radius+1]
        anchor_pixels = np.column_stack((xx.ravel() + self._anchor_pixel[0], yy.ravel() + self._anchor_pixel[1], np.ones(xx.size)))
        warped = anchor_pixels @ cumulative[:2].T
        aligned = cv2.remap(gray, warped[:, 0].reshape(xx.shape).astype(np.float32),
                            warped[:, 1].reshape(yy.shape).astype(np.float32), cv2.INTER_LINEAR)
        anchor_correlation = self._correlation(self._anchor_patch, aligned)
        if anchor_correlation < self.min_correlation:
            self._reject('seed appearance changed or occluded')
        position, normal, fraction, plane_error = self._surface(depth, valid, pixel)
        if np.linalg.norm(position - self._position) / delta > self.max_speed_m_s:
            self._reject('surface motion or foreground depth jump exceeds speed bound')
        angle = np.arccos(np.clip(normal @ self._normal, -1., 1.))
        if angle > self.max_normal_change_rad:
            self._reject('surface normal changed beyond local tracking bound')
        self._rigidity(depth, valid, before, after)
        velocity = (position - self._position) / delta
        # Accepted-gate quality score; deliberately never interpreted as probability.
        quality = fraction * min(correlation, anchor_correlation) * (1 - .25 * min(1., fb_error / .8))
        quality *= 1 - .5 * min(1., plane_error / self.max_plane_error_m)
        confidence = .8 + .19 * np.clip(quality, 0., 1.)
        self._anchor_to_current = cumulative
        self._save(gray, depth, valid, pixel, position, normal, rgb_stamp, depth_stamp, now_s)
        return self._estimate(position, normal, velocity, confidence, depth_stamp)
