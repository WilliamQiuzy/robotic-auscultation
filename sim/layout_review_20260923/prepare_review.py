"""Verify downloaded Isaac outputs and arrange existing frames for review.

This performs no 3D rendering, scene changes, or image synthesis.
"""
import hashlib
import json
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw, ImageFont

ROOT = Path(__file__).resolve().parent
PROJECT = ROOT.parent.parent
OUTPUT = ROOT / "camera_comparison_l40s_v1"
REVIEW = ROOT / "review"


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def font(size):
    for name in ("/System/Library/Fonts/Helvetica.ttc", "DejaVuSans.ttf"):
        try:
            return ImageFont.truetype(name, size)
        except OSError:
            pass
    return ImageFont.load_default()


def pair(suffix, filename, subtitle):
    names = ["baseline", "farther_5cm_lower_2cm"]
    titles = ["Original layout | seat 78.0 cm", "Farther 5 cm | seat lower 2 cm"]
    canvas = Image.new("RGB", (1968, 676), "#f2f5f9")
    draw = ImageDraw.Draw(canvas)
    for i, (name, title) in enumerate(zip(names, titles)):
        x = 16 + i * 976
        draw.text((x + 8, 19), title, fill="#16334a", font=font(30))
        with Image.open(OUTPUT / (name + suffix)) as original:
            frame = original.convert("RGB").resize((960, 540), Image.Resampling.LANCZOS)
        canvas.paste(frame, (x, 72))
    draw.text((24, 632), subtitle, fill="#344859", font=font(23))
    canvas.save(REVIEW / filename)


def main():
    manifest = json.loads((OUTPUT / "layout_comparison.json").read_text())
    assert manifest["source_files_unchanged"]
    for relative, expected in manifest["input_sha256"].items():
        assert sha(PROJECT / relative) == expected, relative
    for relative, expected in manifest["output_sha256"].items():
        assert sha(OUTPUT / relative) == expected, relative
    metrics = {}
    for name, variant in manifest["variants"].items():
        points = variant["projected_existing_points"]
        depth = np.load(OUTPUT / (name + "_depth_m.npy"), allow_pickle=False)
        assert depth.shape == (1080, 1920)
        assert len(points) == 10 and all(p["inside_frame"] for p in points)
        residuals = []
        for p in points:
            u, v = p["uv_px"]
            sample = float(depth[int(v), int(u)])
            assert np.isfinite(sample) and sample > 0
            assert abs(sample - p["render_depth_m"]) < 1e-7
            residual = sample - p["optical_xyz_m"][2]
            assert abs(residual - p["depth_residual_m"]) < 1e-7
            residuals.append(abs(residual))
        image_stats = {}
        for camera in ("robotcam", "side"):
            with Image.open(OUTPUT / variant["images"][camera]) as image:
                pixels = np.asarray(image.convert("RGB"))
            assert pixels.shape == (1080, 1920, 3) and pixels.std() > 5
            image_stats[camera] = {"std": float(pixels.std()), "dimensions": [1920, 1080]}
        metrics[name] = {
            "inside_frame_count": len(points),
            "maximum_target_pixel_depth_residual_m": max(residuals),
            "finite_positive_depth_fraction": float(np.mean(np.isfinite(depth) & (depth > 0))),
            "images": image_stats,
        }
    REVIEW.mkdir(exist_ok=True)
    pair("_robotcam_targets.png", "robotcam_comparison.png",
         "Same RobotCam pose and optics | Existing 3D target markers | Static layout comparison")
    pair("_side.png", "side_comparison.png",
         "Same neutral arm pose in both frames | Patient and stool shifted | New motion not simulated")
    receipt = {
        "accepted": True,
        "scope": "Downloaded source/output integrity, image/depth arrays and projected targets; not motion validation.",
        "source_files_checked": len(manifest["input_sha256"]),
        "output_files_checked": len(manifest["output_sha256"]),
        "render_manifest_sha256": sha(OUTPUT / "layout_comparison.json"),
        "metrics": metrics,
        "review_image_sha256": {name: sha(REVIEW / name) for name in ("robotcam_comparison.png", "side_comparison.png")},
    }
    (REVIEW / "verification.json").write_text(json.dumps(receipt, indent=2) + "\n")
    print(json.dumps(receipt, indent=2))


if __name__ == "__main__":
    main()
