"""Read-only full media verification; only a new receipt is written on success."""
import argparse
import json
import math
from pathlib import Path
import shutil
import subprocess

import numpy as np

from .contract import PROJECT, TARGETS, require, sha
from sim.curobo_expert_v1.quality import validate_episode_provenance, verify_hashes
from sim.route_b_full_video_20260924.timeline import full_schedule

CAMERA = "sim/camera_framing_20260923/recommended_camera.json"
LAYOUT = {"posture": [0, 0, 1280, 720], "robotcam": [1280, 0, 640, 360],
          "metadata": [1280, 360, 640, 360]}


def read(path):
    def pairs(items):
        result = {}
        for key, value in items:
            require(key not in result, "duplicate JSON key: " + key)
            result[key] = value
        return result
    def nonfinite(value):
        raise ValueError("nonfinite JSON: " + value)
    return json.loads(Path(path).read_text(), object_pairs_hook=pairs, parse_constant=nonfinite)


def child(root, relative):
    root = Path(root).resolve()
    require(isinstance(relative, str) and relative and not Path(relative).is_absolute()
            and ".." not in Path(relative).parts, "unsafe relative source path")
    path = (root / relative).resolve()
    require(root in path.parents, "unsafe resolved source path")
    return path


def number(actual, expected, label, tolerance=1e-7):
    require(type(actual) in (float, int) and math.isfinite(actual)
            and abs(actual - expected) <= tolerance, "invalid " + label)


def integer(actual, expected, label):
    require(type(actual) is int and actual == expected, "invalid " + label)


def verify_sources(hashes, project=PROJECT):
    require(isinstance(hashes, dict) and hashes, "missing source hashes")
    paths = {}
    for relative, digest in hashes.items():
        path = child(project, relative)
        require(isinstance(digest, str) and len(digest) == 64
                and all(c in "0123456789abcdef" for c in digest), "invalid source digest")
        require(path.is_file() and sha(path) == digest, "source hash mismatch: " + relative)
        paths[relative] = path
    require(len(set(paths.values())) == len(paths), "duplicate source aliases")
    return paths


def validate_schedule(data, outcome, rows, indices, poses, body_paths):
    """Bind every output frame to complete, zero-origin, 240 Hz source evidence."""
    targets = outcome.get("requested_points")
    require(targets in (["R1", "L1", "R2"], list(TARGETS)), "invalid requested point order")
    require(outcome.get("success") is True, "episode did not complete")
    count = len(rows)
    integer(outcome.get("physics_steps"), count, "outcome physics rows")
    require(count >= 120, "insufficient final home evidence")
    times = np.asarray([row["time_s"] for row in rows], float)
    require(np.isfinite(times).all() and np.allclose(times, np.arange(1, count+1)/240,
            atol=1e-7, rtol=0), "source clock shifted or discontinuous")
    points = outcome.get("points", [])
    require([p.get("target") for p in points] == targets, "point inventory/order mismatch")
    end = 0
    number(outcome.get("phase_config", {}).get("hold_duration_s"), 3., "engineering HOLD duration")
    for point in points:
        integer(point.get("start_physics_row"), end, "point start")
        next_end = point.get("end_physics_row")
        require(type(next_end) is int and end < next_end <= count and point.get("success") is True,
                "invalid completed point interval")
        require(all(r["target"] == point["target"] for r in rows[end:next_end]), "point trace coverage mismatch")
        hold = [i for i in range(end, next_end) if rows[i]["phase"] == "HOLD"]
        require(len(hold) == 720 and hold[-1]-hold[0] == 719, "missing continuous 3 s HOLD: " + point["target"])
        end = next_end
    require(end == count, "incomplete point coverage")
    require(all(r["target"] == targets[-1] and r["phase"] == "DONE" and r["subphase"] == "home_settle"
                for r in rows[-120:]), "incomplete final home settle")
    require(data.get("point_intervals") == [{k: p[k] for k in
        ("target", "start_physics_row", "end_physics_row")} for p in points], "point interval manifest mismatch")
    require(indices.ndim == 1 and np.issubdtype(indices.dtype, np.integer)
            and np.array_equal(indices, np.arange(7, count, 8)), "replay pose row inventory mismatch")
    require(body_paths.ndim == 1 and len(body_paths) > 0 and body_paths.dtype.kind in ("U", "S")
            and len(set(map(str, body_paths))) == len(body_paths), "invalid replay body paths")
    require(poses.shape == (len(indices), len(body_paths), 7) and np.isfinite(poses).all(), "invalid replay poses")
    require(np.all(np.abs(np.linalg.norm(poses[:, :, 3:], axis=2)-1.) <= 1e-4), "nonunit replay rotation")
    integer(data.get("fps"), 15, "fps")
    number(data.get("playback_speed"), 1., "playback speed")
    require(data.get("resolution") == [1920, 720], "unexpected render resolution")
    rr, kk, tt = full_schedule(times, indices, 15)
    integer(data.get("frame_count"), len(rr), "frame count")
    integer(data.get("source_physics_rows"), count, "source physics rows")
    for key, value in {"source_duration_s": count/240, "duration_s": len(rr)/15,
                       "terminal_video_padding_s": len(rr)/15-count/240,
                       "source_physics_dt_s": 1/240, "source_pose_dt_s": 1/30,
                       "first_saved_pose_s": times[indices[0]], "last_saved_pose_s": times[indices[-1]]}.items():
        number(data.get(key), value, key)
    frames = data.get("frames")
    require(isinstance(frames, list) and len(frames) == len(rr), "incomplete frame inventory")
    shown_holds = set()
    for index, (frame, r, k, t) in enumerate(zip(frames, rr, kk, tt)):
        # Independent integer identities supplement the retained timeline helper.
        require(r == min((index+1)*16, count)-1 and k == (r+1)//8-1, "frame schedule disagreement")
        for key, value in {"frame_index": index, "measurement_row": int(r), "pose_index": int(k),
                           "pose_row": int(indices[k])}.items():
            integer(frame.get(key), value, "frame " + str(index) + " " + key)
        for key, value in {"video_time_s": index/15, "simulation_interval_end_s": t,
                           "measurement_time_s": times[r], "pose_time_s": times[indices[k]]}.items():
            number(frame.get(key), value, "frame " + str(index) + " " + key)
        for key in ("target", "phase", "subphase"):
            require(frame.get(key) == rows[r][key], "frame phase/source mismatch: " + str(index))
        if frame["phase"] == "HOLD": shown_holds.add(frame["target"])
    require(shown_holds == set(targets) and frames[-1]["subphase"] == "home_settle", "missing HOLD/final home frames")
    return {"frame_count": len(frames), "physics_rows": count, "targets": targets,
            "final_home_included": True, "all_frame_source_bindings_verified": True}


def _probe_pts(path, count, resolution, fps):
    executable = shutil.which("ffprobe")
    if executable is None:
        return None
    result = subprocess.run([executable, "-v", "error", "-select_streams", "v:0", "-show_frames",
        "-show_entries", "stream=width,height,avg_frame_rate,duration,start_time:frame=best_effort_timestamp_time",
        "-of", "json", str(path)], capture_output=True, text=True, timeout=180)
    require(result.returncode == 0, "ffprobe failed: " + result.stderr[-1000:])
    data = json.loads(result.stdout)
    require(len(data.get("streams", [])) == 1, "missing video stream")
    stream = data["streams"][0]
    require((stream["width"], stream["height"]) == tuple(resolution), "ffprobe dimensions mismatch")
    numerator, denominator = map(float, stream["avg_frame_rate"].split("/"))
    number(numerator/denominator, fps, "ffprobe fps")
    frames = data.get("frames", [])
    require(len(frames) == count, "ffprobe frame count mismatch")
    for index, frame in enumerate(frames):
        number(float(frame["best_effort_timestamp_time"]), index/fps, "presentation timestamp", 2e-6)
    number(float(stream["duration"]), count/fps, "video PTS duration", 2e-6)
    number(float(stream.get("start_time", 0)), 0., "video start PTS", 2e-6)
    return {"backend": "ffprobe", "frames": count, "duration_s": count/fps}


def decode_video(path, expected_count, resolution=(1920, 720), fps=15, frame_check=None):
    """Read every frame through EOF; PTS are mandatory via ffprobe or OpenCV."""
    import cv2
    cap = cv2.VideoCapture(str(path))
    require(cap.isOpened(), "video decoder unavailable")
    index = 0
    cv_clock_valid = True
    try:
        number(cap.get(cv2.CAP_PROP_FPS), fps, "decoded fps", 1e-6)
        while True:
            ok, frame = cap.read()
            if not ok: break
            require(index < expected_count, "decoded frame count exceeds manifest")
            require(frame.shape == (resolution[1], resolution[0], 3), "decoded frame dimensions mismatch")
            timestamp = cap.get(cv2.CAP_PROP_POS_MSEC)/1000
            cv_clock_valid = cv_clock_valid and math.isfinite(timestamp) and abs(timestamp-index/fps) < 2e-5
            if frame_check is not None: frame_check(index, frame)
            index += 1
    finally:
        cap.release()
    require(index == expected_count, "decoded frame count differs from manifest")
    probe = _probe_pts(path, index, resolution, fps)
    require(probe is not None or cv_clock_valid, "presentation timestamps unavailable or discontinuous")
    return {"decoded_frames": index, "decoded_through_eof": True, "pts_verified": True,
            "pts": probe or {"backend": "OpenCV", "frames": index, "duration_s": index/fps}}


def validate_comparison_metadata(data, manifests):
    from .video_compare import clip_length
    integer(data.get("fps"), 15, "comparison fps")
    require(len(manifests) == 2, "two panel manifests required")
    lengths = [clip_length(m) for m in manifests]
    require(data.get("panel_frames") == lengths, "comparison panel frame counts mismatch")
    count = max(lengths)
    integer(data.get("frames"), count, "comparison maximum frame count")
    integer(data.get("decoded_frames"), count, "recorded decoded frame count")
    number(data.get("duration_s"), count/15, "comparison duration")
    require(len(data.get("panel_duration_s", [])) == 2
            and len(data.get("frozen_panel_padding_s", [])) == 2, "missing panel timing")
    for index, length in enumerate(lengths):
        number(data["panel_duration_s"][index], length/15, "panel duration")
        number(data["frozen_panel_padding_s"][index], (count-length)/15, "frozen panel padding")
    return lengths


def check_panel_pixels(comparison, source, panel):
    """Check visible source image areas; H.264 is lossy, so compare with explicit tolerances."""
    import cv2
    x = panel*960
    expected = [cv2.resize(source[:720, :1280], (960, 540), interpolation=cv2.INTER_AREA)[40:],
                cv2.resize(source[:360, 1280:], (320, 180), interpolation=cv2.INTER_AREA)]
    actual = [comparison[40:540, x:x+960], comparison[540:, x:x+320]]
    errors = [np.abs(a.astype(np.int16)-b.astype(np.int16)).reshape(-1) for a, b in zip(actual, expected)]
    values = np.concatenate(errors)
    mean, percentile = float(values.mean()), float(np.percentile(values, 99))
    require(mean <= 5. and percentile <= 30., "panel source pixels differ beyond codec tolerance")
    return {"mean_absolute_error": mean, "p99_absolute_error": percentile}


def verify_render(directory, project=PROJECT):
    directory = Path(directory).resolve(); project = Path(project).resolve()
    manifest_path = directory/"review_manifest.json"
    manifest_hash = sha(manifest_path); data = read(manifest_path)
    require(data.get("schema") == "continuous_recorded_route_v1", "unsupported render manifest")
    paths = verify_sources(data.get("source_sha256"), project)
    episode = child(project, data.get("episode"))
    required = {str((episode/name).relative_to(project)) for name in
                ("outcome.json", "training_scene.usdc", "expert_trace.jsonl", "expert_replay.npz")}
    required.update((CAMERA, "sim/route_b_full_video_20260924/timeline.py",
                     "sim/curobo_expert_v1/render_review.py", "sim/curobo_expert_v1/quality.py",
                     "sim/curobo_expert_v1/provenance.py", "sim/expert_quality_20260924/quality.py"))
    require(required.issubset(paths) and bool(set(paths) & {
        "sim/upper_pair_posture_20260924/render_episode.py", "sim/route_b_full_video_20260924/render_full.py"}),
        "incomplete render source inventory")
    require(data.get("camera_config") == CAMERA and data.get("layout") == LAYOUT, "unapproved camera/layout")
    camera = read(paths[CAMERA])
    number(camera.get("downward_pitch_delta_deg"), 7., "camera pitch")
    number(camera.get("camera_height_delta_m"), .05, "camera height")
    require(data.get("source_files_unchanged") is True, "render lacks source stability receipt")
    provenance = validate_episode_provenance(episode)
    require(provenance["status"] == "passed", "episode provenance rejected: " + str(provenance["reasons"]))
    recorded = data.get("episode_provenance", {})
    require(recorded.get("status") == "passed" and recorded.get("artifact_sha256") == provenance["artifact_sha256"],
            "render episode provenance artifact mismatch")
    video = child(directory, data.get("video"))
    require(sha(video) == data.get("video_sha256"), "render video hash mismatch")
    outcome = read(episode/"outcome.json")
    require(sha(episode/"training_scene.usdc") == outcome.get("training_scene_sha256"), "scene hash mismatch")
    require(data.get("source_outcome_accepted") is (outcome.get("training_accepted") is True), "recorded acceptance mismatch")
    with (episode/"expert_trace.jsonl").open() as stream:
        rows = [{k: row[k] for k in ("time_s", "target", "phase", "subphase")}
                for row in map(json.loads, stream)]
    with np.load(episode/"expert_replay.npz", allow_pickle=False) as archive:
        schedule = validate_schedule(data, outcome, rows, archive["row_indices"], archive["poses"], archive["paths"])
    decode = decode_video(video, schedule["frame_count"])
    verify_sources(data["source_sha256"], project)
    require(not verify_hashes(episode, provenance["artifact_sha256"]), "episode changed during media verification")
    require(sha(manifest_path) == manifest_hash and sha(video) == data["video_sha256"], "render changed during verification")
    return ({"status": "passed", "manifest": str(manifest_path), "manifest_sha256": manifest_hash,
             "video_sha256": data["video_sha256"], "episode": str(episode), "schedule": schedule,
             "decode": decode, "recorded_acceptance": provenance.get("recorded_acceptance"),
             "scope": "Media fidelity and saved-source binding; numerical physics acceptance is not recomputed."}, data)


def verify_comparison(directory, project=PROJECT):
    import cv2
    directory = Path(directory).resolve(); project = Path(project).resolve()
    manifest_path = directory/"comparison_video_manifest.json"
    manifest_hash = sha(manifest_path); data = read(manifest_path)
    require(data.get("schema") == "upper_pair_side_by_side_original_time_v1", "unsupported comparison manifest")
    sources = verify_sources(data.get("source_sha256"), project)
    panels = data.get("panel_sources", [])
    require(len(panels) == 2 and [p.get("role") for p in panels] == ["baseline", "alternative"],
            "explicit baseline/alternative panel sources required")
    reports, manifests, videos = [], [], []
    for panel in panels:
        require(panel.get("manifest") in sources and panel.get("video") in sources, "panel source binding missing")
        source_manifest, source_video = sources[panel["manifest"]], sources[panel["video"]]
        require(source_manifest.name == "review_manifest.json", "invalid panel render manifest")
        report, manifest = verify_render(source_manifest.parent, project)
        require(child(source_manifest.parent, manifest["video"]) == source_video, "panel video identity mismatch")
        reports.append(report); manifests.append(manifest); videos.append(source_video)
    lengths = validate_comparison_metadata(data, manifests)
    video = child(directory, data.get("video"))
    require(sha(video) == data.get("sha256"), "comparison video hash mismatch")
    captures = [cv2.VideoCapture(str(p)) for p in videos]
    last = [None, None]; worst = [0., 0.]
    try:
        require(all(c.isOpened() for c in captures), "comparison source decoder unavailable")
        def frame_check(index, frame):
            for panel in range(2):
                if index < lengths[panel]:
                    ok, source = captures[panel].read()
                    require(ok, "comparison source lost frame")
                    last[panel] = source
                require(last[panel] is not None, "empty comparison source panel")
                check = check_panel_pixels(frame, last[panel], panel)
                worst[panel] = max(worst[panel], check["mean_absolute_error"])
        decode = decode_video(video, max(lengths), frame_check=frame_check)
    finally:
        for cap in captures: cap.release()
    verify_sources(data["source_sha256"], project)
    for report, manifest in zip(reports, manifests):
        verify_sources(manifest["source_sha256"], project)
        require(not verify_hashes(Path(report["episode"]), manifest["episode_provenance"]["artifact_sha256"]),
                "source episode changed during comparison verification")
    require(sha(manifest_path) == manifest_hash and sha(video) == data["sha256"], "comparison changed during verification")
    return {"status": "passed", "manifest": str(manifest_path), "manifest_sha256": manifest_hash,
            "video_sha256": data["sha256"], "panel_sources": panels, "source_renders": reports,
            "panel_frames": lengths, "frozen_panel_padding_s": data["frozen_panel_padding_s"],
            "exact_panel_source_index_rule": "min(output_frame_index, panel_frame_count - 1)",
            "all_panel_frames_compared": True, "worst_panel_mean_absolute_error": worst,
            "pixel_tolerance": {"mean_absolute_error_max": 5., "p99_absolute_error_max": 30.,
                                "scope": "Source image regions excluding metadata and title overlay; lossy H264."},
            "decode": decode}


def validate_receipt_path(path, project=PROJECT, episodes=()):
    path = Path(path).resolve(); project = Path(project).resolve()
    require(not path.exists(), "verification receipt must be a new file")
    if project in path.parents:
        relative = path.relative_to(project)
        require(relative.parts[:2] == ("sim", "upper_pair_posture_20260924"),
                "receipt must be outside old sealed packages")
    require(not any(Path(p).resolve() == path or Path(p).resolve() in path.parents for p in episodes),
            "receipt must be outside source episodes")
    return path


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--render", type=Path)
    parser.add_argument("--comparison", type=Path)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args(argv)
    require(args.render is not None or args.comparison is not None, "provide --render and/or --comparison")
    output = validate_receipt_path(args.output)
    result = {"schema": "upper_pair_media_verification_v1", "status": "passed", "verifier_sha256": sha(__file__)}
    if args.render is not None: result["render"] = verify_render(args.render)[0]
    if args.comparison is not None: result["comparison"] = verify_comparison(args.comparison)
    episodes = ([result["render"]["episode"]] if "render" in result else [])
    if "comparison" in result: episodes += [r["episode"] for r in result["comparison"]["source_renders"]]
    validate_receipt_path(output, episodes=episodes)
    require(sha(__file__) == result["verifier_sha256"], "verifier changed during verification")
    output.parent.mkdir(parents=True, exist_ok=True)
    with output.open("x") as stream: stream.write(json.dumps(result, indent=2, allow_nan=False) + "\n")
    print(json.dumps({"status": "passed", "receipt": str(output)}), flush=True)


if __name__ == "__main__":
    main()
