"""Verify all source bindings and sequentially decode every review-video frame."""
import argparse
import json
from pathlib import Path
from .visual_judge import validate_review, decode_samples, sha256


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('manifest',type=Path)
    args=parser.parse_args()
    manifest=args.manifest.resolve(); before=sha256(manifest)
    data,items=validate_review(manifest)
    records=[]
    for item in items:
        _,sampling=decode_samples(item,data['fps'],data['resolution'],8)
        records.append(dict(clip=item['clip']['clip_id'],frame_count=item['clip']['frame_count'],
                            sha256=item['video_sha256'],decoded_full=True,
                            sampled_rgb_sha256=sampling['decoded_rgb_sha256']))
    validate_review(manifest)
    if sha256(manifest)!=before:raise ValueError('manifest changed during verification')
    report=dict(source_bindings_passed=True,manifest_sha256=before,videos=records,
                scope='file identity, source timing, dimensions, rate and complete sequential decode; no policy acceptance')
    (manifest.parent/'media_verification.json').write_text(json.dumps(report,indent=2)+'\n')
    print(json.dumps({'verified_videos':len(records),'manifest':str(manifest)}),flush=True)


if __name__=='__main__':main()
