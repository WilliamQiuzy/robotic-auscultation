"""Describe a sealed dataset without fitting, selecting or changing its splits."""
import argparse
from collections import Counter
import json
from pathlib import Path
import numpy as np
from .train_openpi import load_dataset, _atomic_json
from .training_data import normalize


def summarize(manifest):
    data = load_dataset(manifest)
    episodes = []
    for e in data.episodes:
        normalized = normalize(e['state'], data.normalization['state'])
        outside = np.abs(normalized) > 1
        phases = Counter(e['phases'].tolist())
        episodes.append(dict(
            episode_id=e['episode_id'], target=e['target'], split=e['split'],
            labeled_frames=len(e['state']), phase_labeled_frames=dict(phases),
            full_chunks=sum(data.episodes[i]['episode_id'] == e['episode_id']
                            for i, _ in data.indices[e['split']]),
            time_span_s=float(e['times'][-1] - e['times'][0]),
            hall_voltage_range_V=[float(e['state'][:, 20].min()), float(e['state'][:, 20].max())],
            spring_estimate_range_N=[float(e['state'][:, 21].min()), float(e['state'][:, 21].max())],
            state_token_saturation_fraction=float(outside.mean()),
            state_token_saturation_by_channel=outside.mean(axis=0).tolist(),
        ))
    return dict(
        schema='wam_multimodal_dataset_summary_v1', data_hash=data.data_hash,
        episodes=episodes, total_labeled_frames=sum(e['labeled_frames'] for e in episodes),
        split_chunks={k: len(v) for k, v in data.indices.items()},
        phase_labeled_frames=dict(sum((Counter(e['phase_labeled_frames']) for e in episodes), Counter())),
        normalization_fit_episode_ids=data.normalization['fit_episode_ids'],
        token_saturation_scope='The official-style discrete state prompt clips normalized state to [-1,1]. Reported fractions refer to channel values outside the training-fitted range; action labels are not clipped.',
        sample_count_scope='Adjacent full chunks overlap; chunk counts are not independent episodes. Labeled frame totals exclude phase boundaries and rejected/ineligible endpoints.',
        generalization_scope='One simulated body and one layout. Independent target plans can share transit prefixes. No moving-patient, unseen-person, real sensor-noise or hardware acceptance.',
        modality_scope='One RGB camera; aligned metric depth through online geometry; Hall ADC voltage/spring estimate; measured robot state; stage and target instruction. No audio.',
    )


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--manifest', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    if args.output.exists():
        raise ValueError('fresh summary output required')
    report = summarize(args.manifest)
    _atomic_json(args.output, report)
    print(json.dumps(report, indent=2, allow_nan=False))


if __name__ == '__main__':
    main()
