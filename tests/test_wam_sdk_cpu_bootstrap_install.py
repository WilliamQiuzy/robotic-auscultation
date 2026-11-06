from pathlib import Path
import subprocess
import sys
import json

import pytest


def module():
    from scripts.experiments.wam_runpod_20261026 import install_sdk_cpu_bootstrap
    return install_sdk_cpu_bootstrap


def test_recipe_reproduces_exact_verified_native_pth():
    m=module()
    root=Path(__file__).resolve().parents[1]
    preserved=root/'reports/ausc_policy/recovery_bootstrap_20261105/wam_sdk_cpu_schema_paths.pth'
    site=Path('/opt/wam-runtime/envs/isaacsim60/lib/python3.12/site-packages')
    assert m.render_pth(site).encode()==preserved.read_bytes()


def test_pth_execution_does_not_import_usd_or_kit_and_preserves_plugin_paths():
    m=module()
    code=m.render_pth(Path('/trusted/sdk/site'))
    code+='import sys,json,os; print(json.dumps({"loaded":[n for n in sys.modules if n == "pxr" or n.startswith("pxr.") or n == "omni" or n.startswith("omni.") or n == "carb" or n == "isaacsim"], "plugins":os.environ["PXR_PLUGINPATH_NAME"]}))\n'
    result=subprocess.run([sys.executable,'-S','-c',code],env={'PXR_PLUGINPATH_NAME':'/existing/plugins'},
                          text=True,capture_output=True,check=True)
    value=json.loads(result.stdout)
    assert value['loaded']==[]
    assert value['plugins'].endswith('/existing/plugins')


def test_existing_different_pth_is_never_overwritten(tmp_path):
    m=module();target=tmp_path/'bootstrap.pth';target.write_text('original\n')
    with pytest.raises(ValueError,match='different'):
        m.publish(target,'changed\n')
    assert target.read_text()=='original\n'
    assert m.publish(target,'original\n') is False
