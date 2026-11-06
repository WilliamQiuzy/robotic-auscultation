"""Install only native SDK namespace/plugin paths; never import USD or Kit.

Targets the verified Isaac Sim 6.0.0.1 cp312 SDK. A conflicting existing .pth
or changed SDK metadata fails before publication. Native CPU probes remain
separate from installation and must use the declared native LD_LIBRARY_PATH.
"""
import argparse
import hashlib
import json
from pathlib import Path

USD = 'isaacsim/extscache/omni.usd.libs-1.0.3+6312fa25.lx64.r.cp312'
PHYSX = 'isaacsim/extscache/omni.usd.schema.physx-110.1.11+110.1.1.lx64.r.cp312.u7f4'
CORE = 'isaacsim/extscache/omni.usd.core-1.8.3+6312fa25.lx64.r'
METADATA = {
    CORE+'/bin/usd/omniKit/resources/plugInfo.json': '8906004a147f9f7da5b15d62837457055dbfba8921a194ba32f94e6377a33c60',
    PHYSX+'/plugins/OmniUsdPhysicsDeformableSchema/resources/plugInfo.json': '1852fe2603a2de8e18506435bd672331d3e711f196a24e220e199cba2deff783',
    PHYSX+'/plugins/PhysxSchema/resources/plugInfo.json': 'e274f45c131722ff7909a200e11e35dab014c39cfd522c015e177b3895f3a2c0',
    PHYSX+'/plugins/PhysxSchemaAddition/resources/plugInfo.json': '1b28b438fa272077d486ec5784f366818f855523bf5f365a70bdf7c081048ce9',
}


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def render_pth(site):
    site=Path(site)
    roots=[str(site/USD), str(site/PHYSX)]
    plugins=[str((site/relative).parent) for relative in METADATA]
    return ('# WAM CPU-only native SDK namespace/plugin paths; no pxr, Kit or native library imports.\n'
            'import sys; sys.path[:0] = '+repr(roots)+'\n'
            'import os; os.environ["PXR_PLUGINPATH_NAME"] = os.pathsep.join('+repr(plugins)+
            ' + ([os.environ["PXR_PLUGINPATH_NAME"]] if os.environ.get("PXR_PLUGINPATH_NAME") else []))\n')


def publish(path, content):
    path=Path(path)
    if path.exists():
        if path.read_text()!=content:
            raise ValueError('Existing different bootstrap is preserved: '+str(path))
        return False
    with path.open('x') as stream:
        stream.write(content)
    return True


def install(site):
    site=Path(site).resolve(strict=True)
    for relative,expected in METADATA.items():
        if sha(site/relative)!=expected:
            raise ValueError('Trusted SDK metadata changed: '+relative)
    for relative in (USD+'/pxr/Usd/__init__.py', PHYSX+'/pxr/PhysxSchema/__init__.py'):
        if not (site/relative).is_file():
            raise ValueError('Trusted SDK Python module missing: '+relative)
    target=site/'wam_sdk_cpu_schema_paths.pth'
    created=publish(target,render_pth(site))
    return dict(schema='wam_sdk_cpu_schema_paths_install_v1',path=str(target),bytes=target.stat().st_size,
                sha256=sha(target),created=created,usd_or_kit_imported_by_installer=False,
                scope='Installation only; fresh-process import and metadata-roundtrip proof is separate.')


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--site-packages',required=True,type=Path)
    args=parser.parse_args()
    print(json.dumps(install(args.site_packages),indent=2))


if __name__=='__main__': main()
