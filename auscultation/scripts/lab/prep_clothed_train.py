"""Build clothed training canvases on the 5090: each labeled full-frame -> back_square_crop(depth)
-> 518 canvas + labels mapped to 518 via full_to_det. Same crop as deploy (train==serve)."""
import os, sys, json, numpy as np, cv2
ROOT = os.path.expanduser("~/Desktop/wam-vla-medical/auscultation")
sys.path.insert(0, os.path.join(ROOT, "ausc/perception"))
from tracking.geometry import back_square_crop, Intrinsics  # noqa

LBL = os.path.expanduser("~/ausc_test/clothed_labels_a.json")
SESS = os.path.expanduser("~/ausc_test/orbbec_session2")
OUT = os.path.expanduser("~/ausc_test/clothed_train"); os.makedirs(f"{OUT}/canvas", exist_ok=True)
d = json.load(open(f"{SESS}/intrinsics.json"))
intr = Intrinsics(d["fx"], d["fy"], d["cx"], d["cy"], d["width"], d["height"], 1.0)
labels = json.load(open(LBL))
cap = cv2.VideoCapture(f"{SESS}/rgb.mp4")
ORDER = [f"{s}{i}" for i in range(1, 6) for s in ("R", "L")]
out = {"_meta": {"space": "back_square_crop 518", "domain": "clothed-orbbec"}}
n = 0
for stem, rec in labels.items():
    if stem == "_meta":
        continue
    fr = rec["frame"]
    cap.set(1, fr); ok, bgr = cap.read()
    if not ok:
        continue
    rgb = bgr[:, :, ::-1]
    depth = np.load(f"{SESS}/depth/d{fr:04d}.npy")
    im518, crop = back_square_crop(rgb, depth, intr)
    full = np.array([rec["points"][nm] for nm in ORDER])
    det = crop.full_to_det(full)
    inb = ((det[:, 0] >= 0) & (det[:, 0] < 518) & (det[:, 1] >= 0) & (det[:, 1] < 518)).all()
    cv2.imwrite(f"{OUT}/canvas/{stem}.png", im518[:, :, ::-1])
    out[stem] = {"points": {nm: [round(float(det[i, 0]), 1), round(float(det[i, 1]), 1)]
                            for i, nm in enumerate(ORDER)},
                 "crop": [crop.x0, crop.y0, crop.side], "frame": fr, "all_in_bounds": bool(inb)}
    n += 1
cap.release()
json.dump(out, open(f"{OUT}/train_clothed.json", "w"), indent=1)
oob = [k for k, v in out.items() if k != "_meta" and not v["all_in_bounds"]]
print(f"built {n} clothed canvases -> {OUT}; out-of-bounds frames: {oob}")
