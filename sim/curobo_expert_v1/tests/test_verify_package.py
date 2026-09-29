"""Package integrity checks use temporary packages, never the delivered package."""
import contextlib
import io
import json
import platform
import socket
import sys
from unittest.mock import patch

import pytest

from sim.curobo_expert_v1 import verify_package as verifier


REQUIRED = (
    'README.md', 'BACKEND.md', 'RESULTS_ZH.md', 'comparison.html',
    'results/comparison_summary.json', 'configs/README.md',
    'configs/bridge_v2/bridge_manifest.json', 'verify_package.py',
)


@pytest.fixture
def package(tmp_path, monkeypatch):
    for name in REQUIRED:
        path = tmp_path / name
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text('fixture ' + name + '\n')
    monkeypatch.setattr(verifier, 'ROOT', tmp_path)
    return tmp_path


def run(*args, system='Darwin', hostname='test-mac'):
    output = io.StringIO()
    code = 0
    with patch.object(sys, 'argv', ['verify_package', *args]), \
            patch.object(platform, 'system', return_value=system), \
            patch.object(socket, 'gethostname', return_value=hostname), \
            contextlib.redirect_stdout(output), contextlib.redirect_stderr(io.StringIO()):
        try:
            verifier.main()
        except SystemExit as exc:
            code = exc.code
    return code, json.loads(output.getvalue()) if output.getvalue().strip() else {}


def seal():
    code, report = run('--seal', '--host-label', 'mac')
    assert code == 0 and report['passed'] is True
    return report


def read_manifest(package):
    return json.loads((package / 'package_manifest.json').read_text())


def write_manifest(package, manifest):
    (package / 'package_manifest.json').write_text(json.dumps(manifest))


def both_receipts():
    seal()
    code, report = run('--host-label', 'linux', system='Linux', hostname='test-linux')
    assert code == 0 and report['passed'] is True


def test_seal_records_observed_host(package):
    report = seal()
    assert report.get('platform_system') == 'Darwin'
    assert report.get('hostname') == 'test-mac'
    assert {item['path'] for item in read_manifest(package)['files']} == set(REQUIRED)


@pytest.mark.parametrize('change', ['tamper', 'delete'])
def test_listed_file_change_fails(package, change):
    seal()
    path = package / 'README.md'
    path.write_text('changed') if change == 'tamper' else path.unlink()
    code, report = run('--host-label', 'mac')
    assert code == 1 and report['passed'] is False


def test_unlisted_result_fails(package):
    seal()
    (package / 'results/unbound.json').write_text('{"passed": true}')
    code, report = run('--host-label', 'mac')
    assert code == 1 and report['passed'] is False
    assert any(item['path'] == 'results/unbound.json' for item in report['problems'])


@pytest.mark.parametrize('mutation', ['empty', 'schema', 'duplicate', 'missing_required',
                                     'excluded', 'absolute', 'parent_path', 'size_bool', 'sha256'])
def test_invalid_manifest_fails(package, mutation):
    seal()
    manifest = read_manifest(package)
    if mutation == 'empty':
        manifest['files'] = []
    elif mutation == 'schema':
        manifest['schema'] = 'unrelated_schema'
    elif mutation == 'duplicate':
        manifest['files'].append(manifest['files'][0])
    elif mutation == 'missing_required':
        manifest['files'] = [x for x in manifest['files'] if x['path'] != 'README.md']
        (package / 'README.md').unlink()
    elif mutation == 'excluded':
        path = package / 'results/graph_path_gate_probe_v1/probe.py'
        path.parent.mkdir(parents=True)
        path.write_text('must stay local')
        manifest['files'].append({'path': str(path.relative_to(package)),
                                  'bytes': path.stat().st_size, 'sha256': verifier.sha256(path)})
    elif mutation == 'absolute':
        manifest['files'][0]['path'] = str(package / manifest['files'][0]['path'])
    elif mutation == 'parent_path':
        manifest['files'][0]['path'] = 'configs/../' + manifest['files'][0]['path']
    elif mutation == 'size_bool':
        manifest['files'][0]['bytes'] = True
    else:
        manifest['files'][0]['sha256'] = 'not a digest'
    write_manifest(package, manifest)
    code, report = run('--host-label', 'mac')
    assert code == 1 and report['passed'] is False


def test_seal_requires_delivery_files(package):
    (package / 'comparison.html').unlink()
    code, report = run('--seal', '--host-label', 'mac')
    assert code == 1 and report['passed'] is False


def test_only_declared_optional_and_temporary_files_are_excluded(package):
    excluded = (
        'results/graph_path_gate_probe_v1/probe.py',
        'results/render/disposable_render.usdc', '__pycache__/module.pyc',
        'module.pyc', '.pytest_cache/v/cache/nodeids', '.DS_Store',
        '.__AppleDouble/sidecar', '.AppleDouble/sidecar', 'results/._outcome.json',
    )
    included = ('results/graph_path_gate_probe_v1_backup/probe.py',
                'results/archive/package_manifest.json', 'results/render/scene.usdc')
    for name in excluded + included:
        path = package / name
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text('fixture')
    seal()
    listed = {item['path'] for item in read_manifest(package)['files']}
    assert not listed.intersection(excluded)
    assert set(included).issubset(listed)
    for name in excluded:
        (package / name).unlink()
    code, report = run('--host-label', 'mac')
    assert code == 0 and report['passed'] is True


@pytest.mark.parametrize(('system', 'label'), [('Darwin', 'linux'), ('Linux', 'mac'),
                                             ('Windows', 'mac')])
def test_requested_host_must_match_platform(package, system, label):
    code, _ = run('--seal', '--host-label', label, system=system)
    assert code != 0
    assert not (package / ('package_verification_' + label + '.json')).exists()
    assert not (package / 'package_manifest.json').exists()


def test_reseal_invalidates_other_host_receipt(package):
    both_receipts()
    (package / 'README.md').write_text('new package')
    seal()
    assert not (package / 'package_verification_linux.json').exists()


def test_pair_verification_is_read_only(package):
    both_receipts()
    before = {p.name: p.read_bytes() for p in package.glob('package*.json')}
    code, report = run('--verify-pair', '--host-label', 'mac')
    assert code == 0 and report['passed'] is True
    assert before == {p.name: p.read_bytes() for p in package.glob('package*.json')}


def test_pair_requires_other_host_receipt(package):
    seal()
    code, report = run('--verify-pair', '--host-label', 'mac')
    assert code == 1 and report.get('passed') is False


@pytest.mark.parametrize(('field', 'value'), [
    ('passed', False), ('manifest_sha256', '0' * 64), ('files_checked', 1),
    ('bytes_checked', 1), ('platform_system', 'Darwin'), ('hostname', ''),
    ('host_label', 'mac'), ('problems', [{'reason': 'failed'}]),
    ('verification_mode', 'pair'), ('schema', 'legacy_receipt'),
    ('files_checked', True),
])
def test_pair_rejects_invalid_linux_receipt(package, field, value):
    both_receipts()
    path = package / 'package_verification_linux.json'
    receipt = json.loads(path.read_text())
    receipt[field] = value
    path.write_text(json.dumps(receipt))
    code, report = run('--verify-pair', '--host-label', 'mac')
    assert code == 1 and report.get('passed') is False


def test_pair_rejects_stale_mac_receipt_without_rewriting_it(package):
    both_receipts()
    receipt_path = package / 'package_verification_mac.json'
    old_mac = receipt_path.read_bytes()
    (package / 'README.md').write_text('new package')
    seal()
    run('--host-label', 'linux', system='Linux', hostname='test-linux')
    receipt_path.write_bytes(old_mac)
    code, report = run('--verify-pair', '--host-label', 'mac')
    assert code == 1 and report.get('passed') is False
    assert receipt_path.read_bytes() == old_mac


def test_pair_checks_current_file_contents(package):
    both_receipts()
    (package / 'README.md').write_text('tampered after receipts')
    code, report = run('--verify-pair', '--host-label', 'mac')
    assert code == 1 and report.get('passed') is False


@pytest.mark.parametrize('content', ['{', '[]', '{"schema":"wam.expert_comparison.package.v1","files":{}}'])
def test_invalid_json_or_container_clears_old_success(package, content):
    seal()
    (package / 'package_manifest.json').write_text(content)
    code, report = run('--host-label', 'mac')
    assert code == 1 and report['passed'] is False
    saved = json.loads((package / 'package_verification_mac.json').read_text())
    assert saved['passed'] is False


def test_core_symlink_is_rejected(package, tmp_path):
    seal()
    (package / 'linked.txt').symlink_to(package / 'README.md')
    code, report = run('--host-label', 'mac')
    assert code == 1 and report['passed'] is False


def test_failed_reseal_still_invalidates_old_linux_receipt(package):
    both_receipts()
    (package / 'comparison.html').unlink()
    code, report = run('--seal', '--host-label', 'mac')
    assert code == 1 and report['passed'] is False
    assert not (package / 'package_verification_linux.json').exists()
