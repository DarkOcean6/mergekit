import torch
from safetensors.torch import load_file, save_file

from mergekit.common import ModelReference
from mergekit.io.tasks import LoadTensor, LoaderCache, SaveTensor, TensorWriterTask


EMBEDDING_NAME = "model.language_model.embed_tokens.weight"
LM_HEAD_NAME = "lm_head.weight"
INCORRECT_LM_HEAD_NAME = "model.language_model.lm_head.weight"


def _materialize_lm_head(tmp_path, physical_head=None):
    source_dir = tmp_path / "source"
    output_dir = tmp_path / "output"
    source_dir.mkdir()

    embedding = torch.arange(12, dtype=torch.bfloat16).reshape(3, 4)
    source_tensors = {EMBEDDING_NAME: embedding}
    if physical_head is not None:
        source_tensors[LM_HEAD_NAME] = physical_head
    save_file(source_tensors, source_dir / "model.safetensors")

    LoaderCache().loaders.clear()
    model = ModelReference.parse(str(source_dir))
    load_task = LoadTensor(
        model=model,
        tensor=LM_HEAD_NAME,
        tied_names=(EMBEDDING_NAME,),
    )
    loaded = load_task.execute()

    writer_task = TensorWriterTask(
        out_path=str(output_dir),
        max_shard_size=1_000_000,
    )
    save_task = SaveTensor(
        tensor_name=LM_HEAD_NAME,
        tensor_task=load_task,
        writer_task=writer_task,
        clone=False,
    )
    writer = writer_task.execute()
    save_task.execute(writer=writer, tensor=loaded)
    writer.finalize()

    return source_tensors, load_file(output_dir / "model.safetensors")


def test_gemma4_missing_lm_head_materializes_own_tied_embedding(tmp_path):
    source, output = _materialize_lm_head(tmp_path)

    assert set(output) == {LM_HEAD_NAME}
    assert INCORRECT_LM_HEAD_NAME not in output
    assert torch.equal(output[LM_HEAD_NAME], source[EMBEDDING_NAME])
    assert output[LM_HEAD_NAME].dtype == source[EMBEDDING_NAME].dtype


def test_gemma4_physical_root_lm_head_is_preferred(tmp_path):
    physical_head = torch.arange(12, dtype=torch.bfloat16).reshape(3, 4) + 100
    source, output = _materialize_lm_head(
        tmp_path,
        physical_head=physical_head,
    )

    assert set(output) == {LM_HEAD_NAME}
    assert INCORRECT_LM_HEAD_NAME not in output
    assert torch.equal(output[LM_HEAD_NAME], source[LM_HEAD_NAME])
    assert not torch.equal(output[LM_HEAD_NAME], source[EMBEDDING_NAME])
    assert output[LM_HEAD_NAME].dtype == source[LM_HEAD_NAME].dtype
