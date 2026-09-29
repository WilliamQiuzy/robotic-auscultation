"""Original-time full-ten-point recorded physics review through a private adapter.

Uses a private copy of the sealed two-view renderer with explicit owned-resource
teardown and this package's provenance validator. Sealed inputs remain unchanged.
"""
import argparse
import importlib.util
import json
import os
from pathlib import Path
import sys

from .config import PROJECT, TARGETS
from .contract import sha
from .quality import validate_episode_provenance
from .media_teardown import require_complete_teardown


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--episode', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--label', default='AUSCULTATION SIMULATION SKILL')
    args = parser.parse_args(argv)
    if not sys.platform.startswith('linux'):
        raise RuntimeError('recorded-physics rendering runs only on the authorized Linux Isaac host')
    episode, output = args.episode.resolve(), args.output.resolve()
    provenance = validate_episode_provenance(episode)
    if provenance['status'] != 'passed':
        raise ValueError('skill episode provenance rejected: '+str(provenance.get('reasons')))
    outcome = json.loads((episode/'outcome.json').read_text())
    if outcome.get('requested_points') != list(TARGETS):
        raise ValueError('final skill video requires the complete accepted ten-point sequence')
    from .provenance import implementation_sources
    implementation = {str(path.relative_to(PROJECT)): sha(path) for path in implementation_sources().values()}
    inherited = PROJECT/'sim/upper_pair_posture_20260924/render_episode.py'
    source = PROJECT/'sim/auscultation_skill_v1/private_renderer.py'
    spec = importlib.util.spec_from_file_location('wam_skill_recorded_render_adapter', source)
    renderer = importlib.util.module_from_spec(spec); spec.loader.exec_module(renderer)
    renderer.validate_episode_provenance = validate_episode_provenance
    saved_argv = sys.argv
    try:
        sys.argv = [str(Path(__file__)), '--episode', str(episode), '--output', str(output), '--label', args.label]
        import isaacsim
        from .run import graceful_simulation_shutdown
        with graceful_simulation_shutdown(isaacsim) as lifecycle:
            teardown = renderer.main()
    finally:
        sys.argv = saved_argv
    if lifecycle['apps_created'] != 1:
        raise RuntimeError('renderer did not return through exactly one initialized SimulationApp')
    require_complete_teardown(teardown, annotators=2, products=2)
    if any(sha(PROJECT/path) != digest for path, digest in implementation.items()):
        raise RuntimeError('versioned renderer implementation changed during rendering')
    manifest_path = output/'review_manifest.json'
    manifest = json.loads(manifest_path.read_text())
    manifest.update(schema='auscultation_skill_recorded_full10_review_v1',
                    renderer_implementation_sha256=implementation,
                    render_lifecycle=dict(apps_created=1, close_returned=True,
                        completion_stage='post_app_close_pre_process_exit',
                        exit_policy='controlled_os_exit_after_close'),
                    source_binding_validator='sim.auscultation_skill_v1.quality.validate_episode_provenance',
                    executed_controller_class=outcome['executed_controller_class'],
                    inherited_renderer=str(inherited.relative_to(PROJECT)),
                    executed_renderer=str(source.relative_to(PROJECT)), resource_teardown=teardown,
                    scope='Actual saved physics poses at original time, approved 7 degree camera plus whole arm view; no interpolation or controller rerun; no acoustic or clinical acceptance.')
    with manifest_path.open('w') as stream:
        stream.write(json.dumps(manifest, indent=2, allow_nan=False)+'\n')
        stream.flush(); os.fsync(stream.fileno())
    return 0


if __name__ == '__main__':
    from .runtime import exit_after_cleanup
    exit_after_cleanup(main())
