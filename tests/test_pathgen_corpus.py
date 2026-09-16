"""Guards for the PathGen corpus path: LMDB format, grouped split, epoch cap.

The PathGen run reuses the QUILT training stack, so what has to hold is that the
LMDB it builds is byte-compatible with the loader, that a slide cannot straddle
train and val, and that capping the epoch keeps checkpoints inside the six-hour
preemptable window without distorting the schedule.
"""

import importlib.util
import io
import sys
from pathlib import Path

import numpy as np
import pytest
import torch
from PIL import Image

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from lumen.training.data import QuiltLMDBDataset, split_indices_grouped

lmdb = pytest.importorskip("lmdb")


def _load_builder():
    """Import cli/build_pathgen_lmdb.py without importing heavy CLI siblings."""
    path = Path(__file__).resolve().parents[1] / "cli" / "build_pathgen_lmdb.py"
    spec = importlib.util.spec_from_file_location("build_pathgen_lmdb", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


class _StubTokenizer:
    """Stands in for BioMedBERT: fixed-width ids plus the fields it emits."""

    def __call__(self, text, padding=None, truncation=None, max_length=None):
        ids = [101] + [ord(c) % 1000 for c in text[: max_length - 2]] + [102]
        pad = max_length - len(ids)
        return {
            "input_ids": ids + [0] * pad,
            "token_type_ids": [0] * max_length,
            "attention_mask": [1] * len(ids) + [0] * pad,
        }


def _write_patch(path: Path, size: int = 672) -> None:
    rng = np.random.default_rng(0)
    array = rng.integers(0, 255, size=(size, size, 3), dtype=np.uint8)
    path.parent.mkdir(parents=True, exist_ok=True)
    Image.fromarray(array).save(path, "JPEG", quality=95)


def test_builder_records_load_through_the_training_loader(tmp_path):
    builder = _load_builder()
    root = tmp_path / "corpus"
    rel = "patches/TCGA-XX-0001/TCGA-XX-0001_100_200.jpg"
    _write_patch(root / rel)

    builder._TOK = _StubTokenizer()
    builder._OPT = {"root": root, "size": 224, "quality": 95, "max_text_len": 128}
    blob, group, err = builder._encode_one((rel, "an example caption"))
    assert err is None
    assert group == "TCGA-XX-0001"

    path = tmp_path / "corpus.lmdb"
    env = lmdb.open(str(path), map_size=1 << 26)
    with env.begin(write=True) as txn:
        txn.put(b"0", blob)
        txn.put(b"__len__", builder.pickle.dumps(1))
        txn.put(b"__keys__", builder.pickle.dumps([b"0"]))
        txn.put(b"__groups__", builder.pickle.dumps([group]))
    env.close()

    dataset = QuiltLMDBDataset(path, image_transform=lambda image: image)
    image, tokens = dataset[0]
    assert len(dataset) == 1
    assert dataset.token_length == 128
    assert dataset.groups == ["TCGA-XX-0001"]
    assert sorted(tokens) == ["attention_mask", "input_ids", "token_type_ids"]
    assert all(value.shape == (128,) for value in tokens.values())
    # Stored at the model's input size, so the training transform's Resize is a
    # no-op rather than a second resample.
    assert image.size == (224, 224)


def test_stored_resize_matches_the_training_transform(tmp_path):
    """The build-time PIL resize must equal torchvision's, pixel for pixel."""
    torchvision = pytest.importorskip("torchvision")
    from torchvision.transforms import InterpolationMode, Resize

    builder = _load_builder()
    source = tmp_path / "patch.jpg"
    _write_patch(source)
    with Image.open(source) as handle:
        original = handle.convert("RGB")

    ours = original.resize((224, 224), Image.BICUBIC)
    theirs = Resize(224, interpolation=InterpolationMode.BICUBIC)(original)
    assert theirs.size == (224, 224)
    np.testing.assert_array_equal(np.asarray(ours), np.asarray(theirs))
    # And the transform leaves an already-sized image untouched.
    np.testing.assert_array_equal(
        np.asarray(Resize(224, interpolation=InterpolationMode.BICUBIC)(ours)),
        np.asarray(ours))


def test_unreadable_patch_is_reported_not_raised(tmp_path):
    builder = _load_builder()
    builder._TOK = _StubTokenizer()
    builder._OPT = {"root": tmp_path, "size": 224, "quality": 95, "max_text_len": 128}
    blob, group, err = builder._encode_one(("patches/WSI-A/missing.jpg", "caption"))
    assert blob is None
    assert group == "WSI-A"
    assert "No such file" in err or "FileNotFoundError" in err


def test_grouped_split_never_puts_a_slide_on_both_sides():
    groups = [f"wsi{i // 100}" for i in range(1000)]  # 10 slides, 100 patches each
    train_idx, val_idx, n_val_groups = split_indices_grouped(groups, 0.2, seed=0)

    assert sorted(train_idx + val_idx) == list(range(1000))
    assert n_val_groups >= 1
    train_groups = {groups[i] for i in train_idx}
    val_groups = {groups[i] for i in val_idx}
    assert not (train_groups & val_groups)
    # Whole slides, so the held-out rows land on a group boundary.
    assert len(val_idx) == 100 * n_val_groups


def test_grouped_split_is_deterministic_across_calls():
    groups = [f"wsi{i // 7}" for i in range(700)]
    first = split_indices_grouped(groups, 0.1, seed=3)
    second = split_indices_grouped(groups, 0.1, seed=3)
    other = split_indices_grouped(groups, 0.1, seed=4)
    assert first == second
    assert first[1] != other[1]


class _Head(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.logit_scale = torch.nn.Parameter(torch.tensor(1.3))

    def forward(self, image, text):
        return image @ text.T * self.logit_scale.exp().clamp(1.0, 20.0)


class _ToyModel(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.image = torch.nn.Linear(4, 3)
        self.text = torch.nn.Linear(5, 3)
        self.contrastive_head = _Head()

    def forward(self, images, tokens):
        image = self.image(images)
        text = self.text(tokens["features"])
        logits = self.contrastive_head(
            torch.nn.functional.normalize(image, dim=-1),
            torch.nn.functional.normalize(text, dim=-1))
        return {"logits": logits, "img_proj": image, "txt_proj": text}


def _toy_loader(n_batches: int, batch_size: int = 4):
    generator = torch.Generator().manual_seed(5)
    batches = [(torch.randn(batch_size, 4, generator=generator),
                {"features": torch.randn(batch_size, 5, generator=generator)})
               for _ in range(n_batches)]

    class _Loader:
        """Stands in for a DataLoader: len() plus a re-iterable sequence."""

        def __len__(self):
            return len(batches)

        def __iter__(self):
            return iter(batches)

    return _Loader()


def _capped_config(tmp_path, **overrides):
    from lumen.training.config import TrainConfig

    cfg = TrainConfig()
    cfg.epochs = 2
    cfg.steps_per_epoch = 2
    cfg.grad_accum = 1
    cfg.batch_size = 4
    cfg.amp = False
    cfg.min_epochs = 2
    cfg.optim.warmup_steps = 1
    cfg.out_dir = str(tmp_path / "run")
    for name, value in overrides.items():
        setattr(cfg, name, value)
    return cfg


@pytest.mark.parametrize("gradient_cache, grad_accum", [(False, 1), (True, 2)])
def test_epoch_cap_stops_the_epoch_and_sizes_the_schedule(
        tmp_path, gradient_cache, grad_accum):
    """A capped epoch must take exactly N optimizer steps, then checkpoint.

    Without this, one PathGen pass (~1,555 steps, ~6.8h) outruns the six-hour
    preemptable wall and no window ever reaches the end-of-epoch checkpoint.
    """
    from lumen.training.engine import train

    cfg = _capped_config(tmp_path, gradient_cache=gradient_cache,
                         grad_accum=grad_accum)
    loader = _toy_loader(n_batches=10 * grad_accum)
    result = train(cfg, loader, _toy_loader(n_batches=2 * grad_accum),
                   _ToyModel(), torch.device("cpu"))

    assert len(result["history"]) == cfg.epochs
    checkpoint = torch.load(Path(cfg.out_dir) / "checkpoint_last.pt",
                            map_location="cpu", weights_only=False)
    # 2 epochs x 2 capped steps, not the 10 steps a full pass would have taken.
    assert checkpoint["extra"]["global_step"] == cfg.epochs * cfg.steps_per_epoch


def test_uncapped_epoch_still_walks_the_whole_loader(tmp_path):
    from lumen.training.engine import train

    cfg = _capped_config(tmp_path, steps_per_epoch=0)
    train(cfg, _toy_loader(n_batches=6), _toy_loader(n_batches=2),
          _ToyModel(), torch.device("cpu"))

    checkpoint = torch.load(Path(cfg.out_dir) / "checkpoint_last.pt",
                            map_location="cpu", weights_only=False)
    assert checkpoint["extra"]["global_step"] == 2 * 6


def test_flat_split_is_used_when_the_lmdb_carries_no_groups(tmp_path):
    """QUILT has no group metadata and must keep its original flat split."""
    builder = _load_builder()
    root = tmp_path / "corpus"
    rel = "patches/WSI-A/p.jpg"
    _write_patch(root / rel, size=224)
    builder._TOK = _StubTokenizer()
    builder._OPT = {"root": root, "size": 224, "quality": 95, "max_text_len": 128}
    blob, _, _ = builder._encode_one((rel, "caption"))

    path = tmp_path / "nogroups.lmdb"
    env = lmdb.open(str(path), map_size=1 << 26)
    with env.begin(write=True) as txn:
        txn.put(b"0", blob)
        txn.put(b"__len__", builder.pickle.dumps(1))
        txn.put(b"__keys__", builder.pickle.dumps([b"0"]))
    env.close()

    assert QuiltLMDBDataset(path, image_transform=lambda image: image).groups is None
