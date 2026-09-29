"""Independent synthetic image/depth fixtures for the bounded local tracker."""
import cv2
import numpy as np
import pytest
from scipy.spatial.transform import Rotation

from sim.auscultation_vla_v2.tracking import SurfaceTracker, TrackingLost


HEIGHT, WIDTH = 144, 192
K = np.array([[220., 0., 96.], [0., 220., 72.], [0., 0., 1.]])
SEED = np.array([96., 72.])


def texture(seed=17):
    rng = np.random.default_rng(seed)
    raw = rng.integers(0, 256, (HEIGHT, WIDTH, 3), dtype=np.uint8)
    return cv2.GaussianBlur(raw, (3, 3), .55)


def scene(rgb=None, rotation=None, translation=None):
    """Render a textured plane with a known rigid motion; tracker sees only pixels."""
    rgb = texture() if rgb is None else rgb
    rotation = np.eye(3) if rotation is None else rotation
    translation = np.zeros(3) if translation is None else np.asarray(translation)
    center = np.array([0., 0., 1.])
    offset = center + translation - rotation @ center
    homography = K @ (rotation + np.outer(offset, [0., 0., 1.])) @ np.linalg.inv(K)
    image = cv2.warpPerspective(rgb, homography, (WIDTH, HEIGHT))
    yy, xx = np.mgrid[:HEIGHT, :WIDTH]
    rays = np.stack(((xx - K[0, 2]) / K[0, 0],
                     (yy - K[1, 2]) / K[1, 1], np.ones_like(xx)), axis=-1)
    normal = rotation @ np.array([0., 0., -1.])
    depth = ((normal @ (center + translation)) / (rays @ normal)).astype(np.float32)
    return image, depth, np.ones((HEIGHT, WIDTH), dtype=bool)


def initialize(tracker, frame=None, time_s=1.):
    return tracker.initialize(*(scene() if frame is None else frame), SEED,
                              rgb_time_s=time_s, depth_time_s=time_s, now_s=time_s)


def update(tracker, frame=None, time_s=1.05, **times):
    return tracker.update(*(scene() if frame is None else frame),
                          **dict(rgb_time_s=time_s, depth_time_s=time_s,
                                 now_s=time_s, **times))


def test_initial_geometry_uses_optical_z_and_rigid_camera_transform():
    transform = np.eye(4)
    transform[:3, :3] = Rotation.from_euler('xyz', [.3, -.2, .4]).as_matrix()
    transform[:3, 3] = [.2, -.5, .8]
    tracker = SurfaceTracker(K, transform)
    result = tracker.initialize(*scene(), [108., 76.],
                                rgb_time_s=1., depth_time_s=1.005, now_s=1.01)
    point_camera = np.array([12. / 220., 4. / 220., 1.])
    np.testing.assert_allclose(result.position_base_m,
                               transform[:3, :3] @ point_camera + transform[:3, 3], atol=1e-6)
    np.testing.assert_allclose(result.rotation_base,
                               transform[:3, :3] @ np.diag([1., -1., -1.]), atol=1e-6)
    np.testing.assert_array_equal(result.velocity_base_m_s, np.zeros(3))
    assert result.source == 'rgbd_surface_tracking'
    assert result.time_s == 1.005
    assert result.revision == 0
    assert .8 <= result.confidence <= 1.
    assert not tracker.lost


def test_rgb_translation_and_causal_velocity_not_seed_pixel_or_oracle():
    tracker = SurfaceTracker(K, np.eye(4))
    first = initialize(tracker)
    second = update(tracker, scene(translation=[.018, -.009, 0.]))
    third = update(tracker, scene(translation=[.027, -.0045, 0.]), time_s=1.1)
    np.testing.assert_allclose(second.position_base_m, [.018, -.009, 1.], atol=.001)
    np.testing.assert_allclose(second.velocity_base_m_s,
                               (second.position_base_m - first.position_base_m) / .05, atol=1e-9)
    np.testing.assert_allclose(third.velocity_base_m_s,
                               (third.position_base_m - second.position_base_m) / .05, atol=1e-9)
    np.testing.assert_allclose(tracker.pixel_xy, SEED + [5.94, -.99], atol=.25)
    assert first.revision == second.revision == third.revision == 0


def test_rotated_surface_normal_and_deterministic_full_tangent_frame():
    tracker = SurfaceTracker(K, np.eye(4))
    initialize(tracker)
    rotation = Rotation.from_euler('xyz', [.035, -.055, .025]).as_matrix()
    result = update(tracker, scene(rotation=rotation, translation=[.004, .002, .002]))
    normal = rotation @ [0., 0., -1.]
    tangent = np.array([1., 0., 0.]) - normal * normal[0]
    tangent /= np.linalg.norm(tangent)
    expected = np.column_stack((tangent, np.cross(normal, tangent), normal))
    np.testing.assert_allclose(result.position_base_m, [.004, .002, 1.002], atol=.001)
    np.testing.assert_allclose(result.rotation_base, expected, atol=2e-5)
    np.testing.assert_allclose(result.rotation_base.T @ result.rotation_base, np.eye(3), atol=1e-8)
    assert np.linalg.det(result.rotation_base) == pytest.approx(1.)


@pytest.mark.parametrize('outlier_m', [.02, .15])
def test_robust_fit_tolerates_sparse_depth_outliers(outlier_m):
    image, depth, valid = scene()
    depth[63:83:5, 87:107:5] += outlier_m
    result = initialize(SurfaceTracker(K, np.eye(4)), (image, depth, valid))
    np.testing.assert_allclose(result.position_base_m, [0., 0., 1.], atol=1e-5)
    np.testing.assert_allclose(result.rotation_base[:, 2], [0., 0., -1.], atol=1e-5)


@pytest.mark.parametrize('kind', ['missing', 'invalid_center', 'mixed_surfaces'])
def test_rejects_missing_local_depth_and_inconsistent_surface(kind):
    image, depth, valid = scene()
    if kind == 'missing':
        valid[45:99, 69:123] = False
    elif kind == 'invalid_center':
        valid[70:75, 94:99] = False
    else:
        depth[:, 96:] -= .08
    tracker = SurfaceTracker(K, np.eye(4))
    with pytest.raises(TrackingLost):
        initialize(tracker, (image, depth, valid))
    assert tracker.lost


@pytest.mark.parametrize('kind', ['blank', 'periodic'])
def test_rejects_textureless_and_ambiguous_rgb_seeds(kind):
    image, depth, valid = scene()
    if kind == 'blank':
        image[:] = 127
    else:
        yy, xx = np.mgrid[:HEIGHT, :WIDTH]
        image[:] = (((xx // 4 + yy // 4) % 2) * 200 + 20)[..., None]
    with pytest.raises(TrackingLost):
        initialize(SurfaceTracker(K, np.eye(4)), (image, depth, valid))


@pytest.mark.parametrize('kind', ['plain_robot', 'different_texture', 'same_texture_foreground'])
def test_occlusion_latches_loss_and_requires_explicit_reinitialization(kind):
    tracker = SurfaceTracker(K, np.eye(4))
    initialize(tracker)
    image, depth, valid = scene()
    if kind == 'plain_robot':
        image[48:96, 72:120] = 80
    elif kind == 'different_texture':
        image[48:96, 72:120] = texture(90)[48:96, 72:120]
    depth[48:96, 72:120] = .75
    with pytest.raises(TrackingLost):
        update(tracker, (image, depth, valid))
    assert tracker.lost and tracker.loss_reason
    with pytest.raises(TrackingLost, match='reinitializ'):
        update(tracker, time_s=1.1)
    recovered = initialize(tracker, time_s=1.15)
    assert recovered.revision == 1
    np.testing.assert_array_equal(recovered.velocity_base_m_s, np.zeros(3))


@pytest.mark.parametrize('times', [
    dict(rgb_time_s=1., depth_time_s=1., now_s=1.05),
    dict(rgb_time_s=.99, depth_time_s=.99, now_s=1.05),
    dict(rgb_time_s=1.05, depth_time_s=1.05, now_s=1.4),
    dict(rgb_time_s=1.1, depth_time_s=1.1, now_s=1.05),
    dict(rgb_time_s=1.02, depth_time_s=1.05, now_s=1.05),
    dict(rgb_time_s=float('nan'), depth_time_s=1.05, now_s=1.05),
])
def test_rejects_duplicate_reversed_stale_future_skewed_timestamps(times):
    tracker = SurfaceTracker(K, np.eye(4))
    old = initialize(tracker)
    with pytest.raises(TrackingLost):
        tracker.update(*scene(), **times)
    assert old.time_s == 1.
    assert tracker.lost
    with pytest.raises(TrackingLost):
        update(tracker, time_s=1.45)


def test_no_input_or_return_value_aliases_change_tracker_state():
    intrinsic, extrinsic = K.copy(), np.eye(4)
    tracker = SurfaceTracker(intrinsic, extrinsic)
    frame = scene()
    snapshots = [value.copy() for value in frame]
    first = initialize(tracker, frame)
    for value, snapshot in zip(frame, snapshots):
        np.testing.assert_array_equal(value, snapshot)
    intrinsic[:] = 0
    extrinsic[:] = 0
    for value in frame:
        value[:] = 0
    first.position_base_m[:] = 999
    first.rotation_base[:] = 999
    first.velocity_base_m_s[:] = 999
    tracker.pixel_xy[:] = 999
    result = update(tracker)
    np.testing.assert_allclose(result.position_base_m, [0., 0., 1.], atol=1e-6)
    np.testing.assert_allclose(result.velocity_base_m_s, 0., atol=1e-6)
    np.testing.assert_allclose(tracker.pixel_xy, SEED, atol=.01)


@pytest.mark.parametrize('kind', ['rgb_type', 'depth_type', 'mask_type', 'valid_nan', 'shape'])
def test_malformed_frames_fail_closed(kind):
    tracker = SurfaceTracker(K, np.eye(4))
    initialize(tracker)
    rgb, depth, valid = scene()
    if kind == 'rgb_type':
        rgb = rgb.astype(float)
    elif kind == 'depth_type':
        depth = depth.astype(np.uint16)
    elif kind == 'mask_type':
        valid = valid.astype(np.uint8)
    elif kind == 'valid_nan':
        depth[72, 96] = np.nan
    else:
        depth = depth[:-1]
    with pytest.raises(TrackingLost):
        update(tracker, (rgb, depth, valid))
    assert tracker.lost


@pytest.mark.parametrize('kind', ['focal', 'singular', 'reflection', 'nonrigid'])
def test_invalid_camera_calibration_rejected(kind):
    intrinsic, transform = K.copy(), np.eye(4)
    if kind == 'focal':
        intrinsic[0, 0] = 0
    elif kind == 'singular':
        intrinsic[1] = intrinsic[0]
    elif kind == 'reflection':
        transform[2, 2] = -1
    else:
        transform[3, 0] = .1
    with pytest.raises(ValueError):
        SurfaceTracker(intrinsic, transform)


def test_update_before_initialize_and_too_close_to_image_boundary_rejected():
    tracker = SurfaceTracker(K, np.eye(4))
    with pytest.raises(TrackingLost, match='initializ'):
        update(tracker)
    with pytest.raises(TrackingLost):
        tracker.initialize(*scene(), [2., 2.], rgb_time_s=1., depth_time_s=1., now_s=1.)


def test_optical_y_tangent_fallback_for_side_facing_plane():
    image, depth, valid = scene()
    yy, xx = np.mgrid[:HEIGHT, :WIDTH]
    valid = xx > 96
    depth[:] = np.nan
    depth[valid] = .3 / ((xx[valid] - 96.) / 220.)
    tracker = SurfaceTracker(K, np.eye(4))
    result = tracker.initialize(image, depth, valid, [168., 72.],
                                rgb_time_s=1., depth_time_s=1., now_s=1.)
    np.testing.assert_allclose(result.position_base_m, [.3, 0., .3 / (72. / 220.)], atol=1e-6)
    np.testing.assert_allclose(result.rotation_base,
                               np.column_stack(([0., 1., 0.], [0., 0., -1.], [-1., 0., 0.])), atol=1e-6)


def test_consumer_clock_reversal_fails_closed_even_when_sources_advance():
    tracker = SurfaceTracker(K, np.eye(4))
    tracker.initialize(*scene(), SEED, rgb_time_s=1., depth_time_s=1., now_s=1.1)
    with pytest.raises(TrackingLost):
        tracker.update(*scene(), rgb_time_s=1.05, depth_time_s=1.05, now_s=1.05)
    assert tracker.lost


def test_excessive_frame_gap_fails_even_when_new_frame_is_fresh():
    tracker = SurfaceTracker(K, np.eye(4))
    initialize(tracker)
    with pytest.raises(TrackingLost, match='gap'):
        update(tracker, time_s=1.3)
    assert tracker.lost


def test_subpixel_initial_seed_and_source_arrays_remain_unmodified():
    frame = scene()
    snapshots = [value.copy() for value in frame]
    tracker = SurfaceTracker(K, np.eye(4))
    result = tracker.initialize(*frame, [96.35, 72.15],
                                rgb_time_s=1., depth_time_s=1., now_s=1.)
    np.testing.assert_allclose(result.position_base_m, [.35 / 220., .15 / 220., 1.], atol=1e-6)
    for value, expected in zip(frame, snapshots):
        np.testing.assert_array_equal(value, expected)


def test_reinitialization_cannot_replay_an_older_source_frame():
    tracker = SurfaceTracker(K, np.eye(4))
    initialize(tracker)
    update(tracker, time_s=1.05)
    with pytest.raises(TrackingLost, match='reversed'):
        tracker.initialize(*scene(), SEED, rgb_time_s=1.02, depth_time_s=1.02, now_s=1.06)
    assert tracker.lost
    recovered = initialize(tracker, time_s=1.1)
    assert recovered.revision == 1


def test_malformed_depth_conversion_latches_failure():
    tracker=SurfaceTracker(K,np.eye(4));initialize(tracker)
    image,_,valid=scene()
    with pytest.raises(TrackingLost):
        tracker.update(image,[[1.],[1.,2.]],valid,
                       rgb_time_s=1.05,depth_time_s=1.05,now_s=1.05)
    assert tracker.lost
    with pytest.raises(TrackingLost):update(tracker,time_s=1.1)


@pytest.mark.parametrize('fault',['opencv','overflow'])
def test_numerical_backend_fault_latches_failure(fault,monkeypatch):
    import cv2
    tracker=SurfaceTracker(K,np.eye(4));initialize(tracker)
    if fault=='opencv':
        def fail(*args,**kwargs):raise cv2.error('injected flow failure')
        monkeypatch.setattr(cv2,'calcOpticalFlowPyrLK',fail)
    with pytest.raises(TrackingLost):
        if fault=='overflow':
            tracker.update(*scene(),rgb_time_s=1.05,depth_time_s=1.05,now_s=10**400)
        else:update(tracker,time_s=1.05)
    assert tracker.lost
