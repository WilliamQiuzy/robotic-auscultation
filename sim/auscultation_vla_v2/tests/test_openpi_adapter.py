"""Seven action channels must not truncate the 32 observed state channels."""
import importlib
import numpy as np
import pytest


def api():
    try:
        module=importlib.import_module('sim.auscultation_vla_v2.openpi_adapter')
    except ModuleNotFoundError:
        module=None
    assert module is not None, 'official model task adapter is missing'
    return module


def pretrained():
    return {('action_in_proj','kernel'):np.arange(32*8).reshape(32,8),
        ('action_in_proj','bias'):np.zeros(8),
        ('action_out_proj','kernel'):np.arange(8*32).reshape(8,32),
        ('action_out_proj','bias'):np.arange(32),
        ('PaliGemma','llm','weight'):np.ones((3,5))}


def test_only_three_action_projection_shapes_change_and_source_is_not_mutated():
    m=api();p=pretrained();out=m.adapt_projection_params(p)
    assert out[('action_in_proj','kernel')].shape==(7,8)
    assert out[('action_out_proj','kernel')].shape==(8,7)
    assert out[('action_out_proj','bias')].shape==(7,)
    assert p[('action_in_proj','kernel')].shape==(32,8)
    assert out[('PaliGemma','llm','weight')] is p[('PaliGemma','llm','weight')]
    assert np.array_equal(out[('action_in_proj','kernel')],p[('action_in_proj','kernel')][:7])


def test_incompatible_checkpoint_fails_before_partial_adaptation():
    m=api();p=pretrained();p[('action_out_proj','bias')]=np.zeros(9)
    with pytest.raises(ValueError,match='projection'):m.adapt_projection_params(p)


def test_only_lora_and_task_projections_are_trainable():
    m=api()
    for path in [('action_in_proj','bias'),('action_out_proj','kernel'),
                 ('PaliGemma','llm','layers','attn','q_einsum','lora_a')]:
        assert m.trainable_path(path)
    for path in [('PaliGemma','llm','kernel'),('PaliGemma','img','embedding'),('time_mlp_in','kernel')]:
        assert not m.trainable_path(path)


# Recorded in results/readiness_v1/probe_gradient.json, confirmed against the
# pinned official lora.py + gemma.py; these are parameter paths, not toy names.
OFFICIAL_LORA_PATHS = (
    'PaliGemma/llm/layers/attn/attn_vec_einsum/lora_a',
    'PaliGemma/llm/layers/attn/attn_vec_einsum/lora_b',
    'PaliGemma/llm/layers/attn/attn_vec_einsum_1/lora_a',
    'PaliGemma/llm/layers/attn/attn_vec_einsum_1/lora_b',
    'PaliGemma/llm/layers/attn/kv_einsum/lora_a',
    'PaliGemma/llm/layers/attn/kv_einsum/lora_b',
    'PaliGemma/llm/layers/attn/kv_einsum_1/lora_a',
    'PaliGemma/llm/layers/attn/kv_einsum_1/lora_b',
    'PaliGemma/llm/layers/attn/q_einsum/lora_a',
    'PaliGemma/llm/layers/attn/q_einsum/lora_b',
    'PaliGemma/llm/layers/attn/q_einsum_1/lora_a',
    'PaliGemma/llm/layers/attn/q_einsum_1/lora_b',
    'PaliGemma/llm/layers/mlp/gating_einsum_lora_a',
    'PaliGemma/llm/layers/mlp/gating_einsum_lora_b',
    'PaliGemma/llm/layers/mlp/linear_lora_a',
    'PaliGemma/llm/layers/mlp/linear_lora_b',
    'PaliGemma/llm/layers/mlp_1/gating_einsum_lora_a',
    'PaliGemma/llm/layers/mlp_1/gating_einsum_lora_b',
    'PaliGemma/llm/layers/mlp_1/linear_lora_a',
    'PaliGemma/llm/layers/mlp_1/linear_lora_b',
)


@pytest.mark.parametrize('path', OFFICIAL_LORA_PATHS)
def test_every_observed_official_attention_and_mlp_lora_factor_is_trainable(path):
    assert api().trainable_path(tuple(path.split('/')))


@pytest.mark.parametrize('path', [
    'PaliGemma/llm/layers/mlp/linear',
    'PaliGemma/llm/layers/mlp/gating_einsum',
    'PaliGemma/llm/layers/mlp/linear_lora_a_backup',
    'PaliGemma/llm/layers/mlp/lora_statistics',
    'PaliGemma/llm/layers/attn/q_einsum/lora_a_backup',
    'PaliGemma/llm/lora_diagnostics/kernel',
    'PaliGemma/img/lora_a',
    'other/linear_lora_a',
    'action_in_proj/unknown_parameter',
])
def test_lora_like_names_and_unknown_projection_parameters_are_not_trainable(path):
    assert not api().trainable_path(tuple(path.split('/')))


def test_pi05_prompt_preserves_all_32_state_features_and_rejects_7_state():
    m=api();state=np.linspace(-1,1,32,dtype=np.float32)
    text=m.state_prompt('Contact and hold.',state)
    numbers=text.split('State: ')[1].split(';')[0].split()
    assert len(numbers)==32 and numbers[0]=='0' and numbers[-1]=='255'
    altered=state.copy();altered[20]=.5
    assert m.state_prompt('Contact and hold.',altered)!=text
    with pytest.raises(ValueError,match='32'):m.state_prompt('Contact.',state[:7])


def test_token_overflow_is_an_error_not_silent_state_truncation():
    m=api()
    class Tokens:
        def encode(self,text,add_bos=True):return list(range(210))
    with pytest.raises(ValueError,match='token'):
        m.tokenize_state(Tokens(),'Contact.',np.zeros(32),max_len=200)


def test_rgb_preprocessing_keeps_one_camera_and_zero_missing_slots():
    m=api();rgb=np.full((80,120,3),100,np.uint8)
    images,masks=m.prepare_images(rgb)
    assert all(x.shape==(224,224,3) for x in images.values())
    assert masks=={'base_0_rgb':True,'left_wrist_0_rgb':False,'right_wrist_0_rgb':False}
    assert np.isfinite(images['base_0_rgb']).all()
    assert not images['left_wrist_0_rgb'].any()
