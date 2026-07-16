import gc

import pytest
import torch
from safetensors.torch import save_file
from transformers import (
    AutoConfig,
    Gemma4Config,
    Gemma4ForConditionalGeneration,
    Gemma4TextConfig,
    Gemma4VisionConfig,
)

from mergekit.architecture import arch_info_for_config
from mergekit.common import ModelReference
from mergekit.config import InputModelDefinition, MergeConfiguration
from mergekit.io import LazyTensorLoader
from mergekit.io.tasks import LoaderCache
from mergekit.merge import run_merge
from mergekit.options import MergeOptions


EMBEDDING_NAME = "model.language_model.embed_tokens.weight"
LM_HEAD_NAME = "lm_head.weight"
INCORRECT_LM_HEAD_NAME = "model.language_model.lm_head.weight"


def _tiny_gemma4_config(tied: bool) -> Gemma4Config:
    text_config = Gemma4TextConfig(
        num_hidden_layers=1,
        vocab_size=4,
        tie_word_embeddings=tied,
    )
    vision_config = Gemma4VisionConfig(num_hidden_layers=1)
    config = Gemma4Config(
        text_config=text_config,
        vision_config=vision_config,
        audio_config=None,
        tie_word_embeddings=tied,
    )
    config.architectures = ["Gemma4ForConditionalGeneration"]
    return config


def _tiny_loadable_gemma4_config() -> Gemma4Config:
    text_config = Gemma4TextConfig(
        vocab_size=16,
        hidden_size=16,
        intermediate_size=32,
        num_hidden_layers=1,
        num_attention_heads=1,
        num_key_value_heads=1,
        head_dim=16,
        global_head_dim=16,
        layer_types=["full_attention"],
        hidden_size_per_layer_input=0,
        vocab_size_per_layer_input=16,
        tie_word_embeddings=True,
    )
    vision_config = Gemma4VisionConfig(
        hidden_size=16,
        intermediate_size=32,
        num_hidden_layers=1,
        num_attention_heads=1,
        num_key_value_heads=1,
        head_dim=16,
        global_head_dim=16,
        position_embedding_size=4,
        patch_size=2,
        standardize=True,
    )
    config = Gemma4Config(
        text_config=text_config,
        vision_config=vision_config,
        audio_config=None,
        tie_word_embeddings=True,
    )
    config.architectures = ["Gemma4ForConditionalGeneration"]
    return config


def _make_checkpoint(path, offset: int, tied: bool, physical_head: bool):
    path.mkdir()
    config = _tiny_gemma4_config(tied=tied)
    config.save_pretrained(path)

    architecture = arch_info_for_config(config)
    tensors = {}
    for index, weight in enumerate(architecture.all_weights(config)):
        if weight.name == LM_HEAD_NAME or weight.optional:
            continue
        shape = (2, 2)
        if weight.name == EMBEDDING_NAME:
            shape = (4, 2)
        elif weight.name.endswith("layer_scalar"):
            shape = (1,)
        tensors[weight.name] = torch.full(
            shape,
            offset + index,
            dtype=torch.bfloat16,
        )

    embedding = torch.arange(8, dtype=torch.bfloat16).reshape(4, 2) + offset
    tensors[EMBEDDING_NAME] = embedding
    if physical_head:
        tensors[LM_HEAD_NAME] = embedding.clone() if tied else embedding + 100

    save_file(tensors, path / "model.safetensors")
    return str(path), tensors


def _run_linear_merge(
    output_path,
    model_a,
    model_b,
    base_model=None,
    dtype="bfloat16",
):
    config = MergeConfiguration(
        merge_method="linear",
        base_model=base_model,
        models=[
            InputModelDefinition(model=model_a, parameters={"weight": 0.25}),
            InputModelDefinition(model=model_b, parameters={"weight": 0.75}),
        ],
        dtype=dtype,
    )
    run_merge(
        config,
        out_path=str(output_path),
        options=MergeOptions(
            copy_tokenizer=False,
            write_model_card=False,
            quiet=True,
        ),
    )
    LoaderCache().loaders.clear()
    gc.collect()


def _load_output(path):
    loader = LazyTensorLoader.from_disk(str(path))
    names = set(loader.index.tensor_paths)
    tensors = {name: loader.get_tensor(name) for name in names}
    del loader
    gc.collect()
    return names, tensors


def test_gemma4_two_tied_models_follow_vanilla_optional_head_layout(tmp_path):
    model_a, tensors_a = _make_checkpoint(
        tmp_path / "model_a",
        offset=0,
        tied=True,
        physical_head=False,
    )
    model_b, tensors_b = _make_checkpoint(
        tmp_path / "model_b",
        offset=10,
        tied=True,
        physical_head=False,
    )
    output_path = tmp_path / "output"

    _run_linear_merge(output_path, model_a, model_b, base_model=model_a)
    names, output = _load_output(output_path)
    output_config = AutoConfig.from_pretrained(output_path)
    expected_embedding = (
        0.25 * tensors_a[EMBEDDING_NAME] + 0.75 * tensors_b[EMBEDDING_NAME]
    ).to(torch.bfloat16)

    assert EMBEDDING_NAME in names
    assert LM_HEAD_NAME not in names
    assert INCORRECT_LM_HEAD_NAME not in names
    assert torch.equal(output[EMBEDDING_NAME], expected_embedding)
    assert output_config.tie_word_embeddings is True
    assert output_config.text_config.tie_word_embeddings is True


def test_gemma4_tied_output_loads_with_naturally_shared_lm_head(tmp_path):
    models = []
    for index, value in enumerate((0.01, 0.03)):
        model_path = tmp_path / f"model_{index}"
        model = Gemma4ForConditionalGeneration(_tiny_loadable_gemma4_config())
        with torch.no_grad():
            for parameter in model.parameters():
                parameter.fill_(value)
            model.model.vision_tower.std_bias.fill_(value)
            model.model.vision_tower.std_scale.fill_(value)
        model.save_pretrained(model_path, safe_serialization=True)
        del model
        gc.collect()

        source_loader = LazyTensorLoader.from_disk(str(model_path))
        source_names = set(source_loader.index.tensor_paths)
        del source_loader
        assert LM_HEAD_NAME not in source_names
        assert EMBEDDING_NAME in source_names
        models.append(ModelReference.parse(str(model_path)))

    output_path = tmp_path / "output"
    _run_linear_merge(
        output_path,
        models[0],
        models[1],
        base_model=models[0],
        dtype="float32",
    )
    output_names, _ = _load_output(output_path)
    assert LM_HEAD_NAME not in output_names

    merged = Gemma4ForConditionalGeneration.from_pretrained(output_path)
    assert merged.config.tie_word_embeddings is True
    assert merged.config.text_config.tie_word_embeddings is True
    assert (
        merged.lm_head.weight.data_ptr()
        == merged.model.language_model.embed_tokens.weight.data_ptr()
    )
    actual = merged.model.language_model.embed_tokens.weight.flatten()[0].item()
    assert actual == pytest.approx(0.025, abs=1e-6)
    del merged
    gc.collect()


def test_gemma4_mixed_physical_and_tied_heads_merge_at_checkpoint_root(tmp_path):
    model_a, tensors_a = _make_checkpoint(
        tmp_path / "model_a",
        offset=0,
        tied=True,
        physical_head=True,
    )
    model_b, tensors_b = _make_checkpoint(
        tmp_path / "model_b",
        offset=10,
        tied=True,
        physical_head=False,
    )
    output_path = tmp_path / "output"

    _run_linear_merge(output_path, model_a, model_b, base_model=model_a)
    names, output = _load_output(output_path)
    expected = (0.25 * tensors_a[LM_HEAD_NAME] + 0.75 * tensors_b[EMBEDDING_NAME]).to(
        torch.bfloat16
    )

    assert LM_HEAD_NAME in names
    assert INCORRECT_LM_HEAD_NAME not in names
    assert torch.equal(output[LM_HEAD_NAME], expected)
    assert torch.equal(output[LM_HEAD_NAME], output[EMBEDDING_NAME])
    assert output[LM_HEAD_NAME].dtype == torch.bfloat16


def test_gemma4_rejects_false_tying_flag_when_physical_head_is_missing(tmp_path):
    model_a, _ = _make_checkpoint(
        tmp_path / "model_a",
        offset=0,
        tied=False,
        physical_head=False,
    )
    model_b, _ = _make_checkpoint(
        tmp_path / "model_b",
        offset=10,
        tied=False,
        physical_head=False,
    )

    with pytest.raises(RuntimeError, match="checkpoint is inconsistent"):
        _run_linear_merge(
            tmp_path / "output",
            model_a,
            model_b,
            base_model=model_a,
        )


def test_gemma4_untied_models_with_physical_heads_remain_supported(tmp_path):
    model_a, tensors_a = _make_checkpoint(
        tmp_path / "model_a",
        offset=0,
        tied=False,
        physical_head=True,
    )
    model_b, tensors_b = _make_checkpoint(
        tmp_path / "model_b",
        offset=10,
        tied=False,
        physical_head=True,
    )
    output_path = tmp_path / "output"

    _run_linear_merge(output_path, model_a, model_b, base_model=model_a)
    names, output = _load_output(output_path)
    output_config = AutoConfig.from_pretrained(output_path)
    expected_head = (
        0.25 * tensors_a[LM_HEAD_NAME] + 0.75 * tensors_b[LM_HEAD_NAME]
    ).to(torch.bfloat16)

    assert LM_HEAD_NAME in names
    assert torch.equal(output[LM_HEAD_NAME], expected_head)
    assert not torch.equal(output[LM_HEAD_NAME], output[EMBEDDING_NAME])
    assert output_config.tie_word_embeddings is False
    assert output_config.text_config.tie_word_embeddings is False


def test_referenced_models_preserve_declared_order():
    config = MergeConfiguration(
        merge_method="linear",
        models=[
            InputModelDefinition(model=ModelReference.parse("owner/model-b")),
            InputModelDefinition(model=ModelReference.parse("owner/model-a")),
            InputModelDefinition(model=ModelReference.parse("owner/model-b")),
        ],
    )

    assert [str(model) for model in config.referenced_models()] == [
        "owner/model-b",
        "owner/model-a",
    ]
