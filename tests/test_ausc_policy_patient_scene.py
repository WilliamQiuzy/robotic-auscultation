import hashlib
import json
from pathlib import Path
import pytest
from sim.ausc_policy import geometry_epoch as g


def specification():
    root=Path(__file__).resolve().parents[1]
    folder=root/'results/ausc_policy/m2_seated_pose_pilot_v1'
    if not folder.exists():pytest.skip('Local numeric pilot artifact unavailable')
    manifest=json.loads((folder/'asset_manifest.json').read_text())
    return dict(directory=str(folder),
        manifest_sha256=hashlib.sha256((folder/'asset_manifest.json').read_bytes()).hexdigest(),
        summary_sha256=hashlib.sha256((folder/'summary.json').read_bytes()).hexdigest(),
        file=next(r['file'] for r in manifest if r['numeric_candidate_screen_passed']),
        identity_split=str(root/'configs/ausc_policy/m2_identity_split_20260930.json'),
        anchor_inventory=str(root/'results/ausc_policy/asset_inventory/inventory.json'),
        geometry_id=g.GEOMETRY_ID,geometry_source_sha256=g.GEOMETRY_SOURCE_SHA256,freshly_generated=True)


def test_actual_candidate_keeps_ten_surface_anchors_and_rejects_failed_candidate():
    from sim.ausc_policy.patient_scene import load_candidate
    spec=specification();arrays,receipt=load_candidate(spec)
    assert len(receipt['targets'])==10 and arrays['vertices'].shape==(6890,3)
    assert receipt['training_eligible'] is False
    manifest=json.loads((Path(spec['directory'])/'asset_manifest.json').read_text())
    spec['file']=next(r['file'] for r in manifest if not r['numeric_candidate_screen_passed'])
    with pytest.raises(ValueError,match='gate'):load_candidate(spec)


def test_patient_sources_cannot_be_relabelled_to_old_geometry_or_changed_manifest():
    from sim.ausc_policy.patient_scene import load_candidate
    spec=specification();spec['geometry_id']='old1262'
    with pytest.raises(ValueError,match='geometry'):load_candidate(spec)
    spec=specification();spec['manifest_sha256']='0'*64
    with pytest.raises(ValueError,match='hash'):load_candidate(spec)


def test_load_uses_verified_bytes_even_if_filename_changes(tmp_path,monkeypatch):
    import shutil
    from sim.ausc_policy import patient_scene as p
    spec=specification();source=Path(spec['directory'])
    for name in ('asset_manifest.json','summary.json',spec['file']):shutil.copyfile(source/name,tmp_path/name)
    spec['directory']=str(tmp_path);read=p._read
    def replace_after_verifying(path,digest):
        data=read(path,digest)
        if Path(path).suffix=='.npz':Path(path).write_bytes(b'unverified replacement')
        return data
    monkeypatch.setattr(p,'_read',replace_after_verifying)
    arrays,_=p.load_candidate(spec)
    assert arrays['vertices'].shape==(6890,3)
