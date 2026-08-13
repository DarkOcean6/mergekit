from typing import Dict

import pytest
import torch
from transformers import PretrainedConfig

from mergekit.architecture import ModelArchitecture, WeightInfo
from mergekit.common import ModelReference
from mergekit.config import ConfigReader, InputModelDefinition, MergeConfiguration
from mergekit.graph import Executor, Task
from mergekit.io.tasks import LoaderCache, LoadTensor
from mergekit.merge_methods.base import MergeMethod, MergeTensorInput
from mergekit.options import MergeOptions
from mergekit.plan import MergePlanner


class _FakeIndex:
    def __init__(self, names):
        self.tensor_paths = {name: "model.safetensors" for name in names}


class _FakeLoader:
    def __init__(self, tensors):
        self.index = _FakeIndex(tensors)
        self.tensors = tensors

    def get_tensor(self, name, **_kwargs):
        return self.tensors[name]

    def flush(self):
        pass


class _StrictMergeTask(Task[torch.Tensor]):
    gather_tensors: MergeTensorInput

    def arguments(self) -> Dict[str, Task]:
        return {"tensors": self.gather_tensors}

    def execute(self, tensors):
        if len(tensors) < 2:
            raise RuntimeError("merge method requires at least one donor tensor")
        return torch.stack(list(tensors.values())).sum(dim=0)


class _StrictMergeMethod(MergeMethod):
    def name(self) -> str:
        return "strict_test_merge"

    def make_task(self, *, tensors, **_kwargs) -> Task:
        return _StrictMergeTask(gather_tensors=tensors)


@pytest.fixture
def model_refs():
    return {
        "base": ModelReference.parse("base"),
        "donor_a": ModelReference.parse("donor_a"),
        "donor_b": ModelReference.parse("donor_b"),
    }


@pytest.fixture(autouse=True)
def disable_checkpoint_conversion(monkeypatch):
    monkeypatch.setattr(
        LoadTensor,
        "_load_converted_tensor",
        lambda _self, _loader: None,
    )


def _plan_optional_tensor(model_refs, tensors_by_model, tensor_name):
    base = model_refs["base"]
    donors = [model_refs["donor_a"], model_refs["donor_b"]]
    loaders = {
        model: _FakeLoader(tensors_by_model.get(model, {})) for model in [*donors, base]
    }
    LoaderCache().loaders = loaders

    config = MergeConfiguration(
        merge_method="linear",
        base_model=base,
        models=[InputModelDefinition(model=model) for model in donors],
    )
    planner = MergePlanner(
        config=config,
        arch_info=ModelArchitecture(modules={}, architectures=[], model_type="test"),
        options=MergeOptions(),
        out_model_config=PretrainedConfig(),
    )
    planner._method = _StrictMergeMethod()
    weight = WeightInfo(name=tensor_name, optional=True)
    planner.plan_tensor(
        weight=weight,
        weights_in=[weight, weight, weight],
        models=[*donors, base],
        cfg_reader=ConfigReader(config=config, t=0),
    )

    assert len(planner._tensors) == 1
    return planner._tensors[0][1]


def _execute(task):
    return next(Executor([task]).run(quiet=True))[1]


def test_optional_mtp_merges_when_all_models_have_tensor(model_refs):
    name = "mtp.pre_fc_norm_embedding.weight"
    tensors = {
        model_refs["base"]: {name: torch.tensor([10.0])},
        model_refs["donor_a"]: {name: torch.tensor([1.0])},
        model_refs["donor_b"]: {name: torch.tensor([2.0])},
    }

    result = _execute(_plan_optional_tensor(model_refs, tensors, name))

    torch.testing.assert_close(result, torch.tensor([13.0]))


def test_optional_mtp_ignores_missing_donor(model_refs):
    name = "mtp.layers.0.mlp.experts.gate_up_proj"
    tensors = {
        model_refs["base"]: {name: torch.tensor([10.0])},
        model_refs["donor_a"]: {name: torch.tensor([1.0])},
    }

    result = _execute(_plan_optional_tensor(model_refs, tensors, name))

    torch.testing.assert_close(result, torch.tensor([11.0]))


def test_optional_mtp_copies_base_when_all_donors_lack_tensor(model_refs):
    name = "mtp.norm.weight"
    base_tensor = torch.tensor([10.0])
    tensors = {model_refs["base"]: {name: base_tensor}}

    result = _execute(_plan_optional_tensor(model_refs, tensors, name))

    torch.testing.assert_close(result, base_tensor)


@pytest.mark.parametrize(
    "name",
    [
        "model.language_model.layers.0.input_layernorm.weight",
        "model.language_model.layers.0.mlp.experts.down_proj",
        "model.language_model.layers.0.mlp.gate.weight",
        "transformer.h.0.attn.c_attn.weight",
    ],
)
def test_missing_required_language_model_tensor_still_raises(model_refs, name):
    LoaderCache().loaders = {model_refs["donor_a"]: _FakeLoader({})}

    with pytest.raises(RuntimeError, match="required but not present"):
        LoadTensor(
            model=model_refs["donor_a"],
            tensor=name,
            optional=False,
        ).execute()
