"""Rebuild the public result table from retained evidence, without a GPU.

This reads historical public summaries. It does not rerun Isaac, qualify raw
physics or reinterpret unreported runs as success or failure.
"""
import argparse
import hashlib
import json
from pathlib import Path


def build_results(data_root):
    root = Path(data_root)
    expert = json.loads((root / 'expert_summary.json').read_text())
    act = json.loads((root / 'act_summary.json').read_text())
    cases = expert['cases']
    if len(cases) != expert['planned_denominator'] or len({c['case'] for c in cases}) != len(cases):
        raise ValueError('Expert denominator or case identities changed')
    successes = sum(c['independent_physical_pass'] and c['independent_case_qualification_pass'] for c in cases)
    if successes != expert['physical_successes']:
        raise ValueError('Expert qualification counts do not reconstruct')
    native = act['native_cases']
    if len({c['case'] for c in native}) != len(native):
        raise ValueError('Duplicate learned-policy episodes')
    source = act['training'][0]
    train_ids = source['scope']['train']['subject_ids']; val_ids = source['scope']['validation']['subject_ids']
    if set(train_ids) & set(val_ids):
        raise ValueError('Identity overlap')
    offline = []
    for row in act['training']:
        offline.append(dict(run_id=row['run_id'], completed_steps=row['steps'], selected_step=row['selected_step'],
                            train_mae_rad=row['selected']['train']['act'],
                            validation_mae_rad=row['selected']['validation']['act'],
                            elapsed_wall_s=row['elapsed_wall_s']))
    return dict(schema='ausc-public-results-v1', evidence_cutoff='2026-10-29',
                expert_qualification=dict(successes=successes, episodes=len(cases),
                                          success_rate=successes / len(cases), cohort='fixed diagnostic expert eight'),
                learned_closed_loop=dict(successes=sum(c['success'] for c in native), episodes=len(native),
                                         success_rate=sum(c['success'] for c in native) / len(native),
                                         cohort='one optimizer TRAIN scene and one selection VAL scene',
                                         independent_heldout=False, cases=native),
                offline_training=offline,
                independent_heldout=dict(status='not_reported', successes=None, episodes=None, success_rate=None),
                astra=dict(status='not_evaluated', successes=None, episodes=None, success_rate=None),
                act_consumes_rgb_depth_hall=False, comparable_leaderboard=False)


def verify_assets(site):
    root = Path(site).resolve(); manifest = json.loads((root / 'data' / 'assets_manifest.json').read_text())
    for item in manifest['files']:
        p = (root / item['path']).resolve()
        if root not in p.parents:
            raise ValueError('Asset escapes site')
        data = p.read_bytes()
        if len(data) != item['bytes'] or hashlib.sha256(data).hexdigest() != item['sha256']:
            raise ValueError('Asset checksum mismatch: ' + item['path'])
    return len(manifest['files'])


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--data', default='docs/data'); parser.add_argument('--output')
    parser.add_argument('--verify-assets', action='store_true'); args = parser.parse_args()
    value = build_results(args.data)
    if args.verify_assets:
        value['verified_assets'] = verify_assets(Path(args.data).parent)
    text = json.dumps(value, indent=2, ensure_ascii=False, allow_nan=False) + '\n'
    if args.output:
        Path(args.output).write_text(text)
    else:
        print(text, end='')


if __name__ == '__main__':
    main()
