"""Dependency compatibility tests.

Verifies that every third-party library used across the src/ codebase can be
imported and that the specific features relied upon work correctly with the
installed versions.  Each test class covers one library as declared in
requirements.txt.
"""

import importlib
import sys
from pathlib import Path
from packaging.version import Version

import numpy as np
import pytest
import torch


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _get_version(pkg_name: str) -> Version:
    """Return the installed version of *pkg_name* as a packaging Version."""
    mod = importlib.import_module(pkg_name)
    return Version(mod.__version__)


# ===================================================================
# 1. PyTorch  (torch, torchvision, torchaudio)
# ===================================================================


class TestTorch:
    """Verify torch >= 2.1.0 features used in src/."""

    def test_version_in_range(self):
        v = _get_version("torch")
        assert v >= Version("2.1.0")

    def test_tensor_creation(self):
        t = torch.randn(2, 3)
        assert t.shape == (2, 3)

    def test_from_numpy_roundtrip(self):
        arr = np.random.rand(4, 5).astype(np.float32)
        t = torch.from_numpy(arr)
        assert t.shape == (4, 5)
        np.testing.assert_allclose(t.numpy(), arr)

    def test_nn_module_forward(self):
        """Linear + LayerNorm + Dropout pipeline used in ST-GCN."""
        layer = torch.nn.Sequential(
            torch.nn.Linear(10, 16),
            torch.nn.LayerNorm(16),
            torch.nn.Dropout(0.1),
        )
        out = layer(torch.randn(2, 10))
        assert out.shape == (2, 16)

    def test_conv2d(self):
        """Conv2d used in ST-GCN blocks for spatial and temporal convolutions."""
        conv = torch.nn.Conv2d(3, 64, kernel_size=1)
        x = torch.randn(2, 3, 16, 33)
        out = conv(x)
        assert out.shape == (2, 64, 16, 33)

    def test_batchnorm2d(self):
        """BatchNorm2d used in ST-GCN blocks."""
        bn = torch.nn.BatchNorm2d(64)
        x = torch.randn(2, 64, 16, 33)
        out = bn(x)
        assert out.shape == (2, 64, 16, 33)

    def test_cross_entropy(self):
        """Cross entropy used in prototypical training."""
        criterion = torch.nn.CrossEntropyLoss()
        logits = torch.randn(4, 10)
        targets = torch.randint(0, 10, (4,))
        loss = criterion(logits, targets)
        assert loss.dim() == 0 and loss.item() > 0

    def test_softmax(self):
        probs = torch.nn.functional.softmax(torch.randn(2, 5), dim=1)
        sums = probs.sum(dim=1)
        torch.testing.assert_close(sums, torch.ones(2), atol=1e-5, rtol=0)

    def test_topk(self):
        t = torch.tensor([0.1, 0.9, 0.5, 0.3, 0.7])
        vals, idx = t.topk(3)
        assert idx[0].item() == 1

    def test_adamw_optimizer(self):
        model = torch.nn.Linear(4, 2)
        opt = torch.optim.AdamW(model.parameters(), lr=1e-3, weight_decay=1e-4)
        loss = model(torch.randn(1, 4)).sum()
        loss.backward()
        opt.step()

    def test_onecycle_lr(self):
        model = torch.nn.Linear(4, 2)
        opt = torch.optim.AdamW(model.parameters(), lr=1e-3)
        sched = torch.optim.lr_scheduler.OneCycleLR(opt, max_lr=1e-3, total_steps=10)
        for _ in range(5):
            sched.step()
        assert opt.param_groups[0]["lr"] > 0

    def test_cosine_annealing_lr(self):
        model = torch.nn.Linear(4, 2)
        opt = torch.optim.AdamW(model.parameters(), lr=1e-3)
        sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=10)
        for _ in range(5):
            sched.step()
        assert opt.param_groups[0]["lr"] > 0

    def test_sequential_lr(self):
        model = torch.nn.Linear(4, 2)
        opt = torch.optim.AdamW(model.parameters(), lr=1e-3)
        warmup = torch.optim.lr_scheduler.LinearLR(opt, start_factor=0.1, total_iters=5)
        cosine = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=5)
        sched = torch.optim.lr_scheduler.SequentialLR(opt, schedulers=[warmup, cosine], milestones=[5])
        for _ in range(8):
            sched.step()

    def test_grad_scaler(self):
        """GradScaler used for mixed-precision training."""
        from torch.cuda.amp import GradScaler
        scaler = GradScaler(enabled=False)
        assert scaler is not None

    def test_autocast_context(self):
        """torch.amp.autocast used in train.py."""
        with torch.amp.autocast(device_type="cpu", enabled=False):
            out = torch.randn(2, 2) @ torch.randn(2, 2)
        assert out.shape == (2, 2)

    def test_clip_grad_norm(self):
        model = torch.nn.Linear(4, 2)
        loss = model(torch.randn(1, 4)).sum()
        loss.backward()
        norm = torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        assert norm >= 0

    def test_weighted_random_sampler(self):
        from torch.utils.data import WeightedRandomSampler
        weights = [1.0, 2.0, 0.5, 1.5]
        sampler = WeightedRandomSampler(weights, num_samples=4, replacement=True)
        indices = list(sampler)
        assert len(indices) == 4

    def test_dataloader(self):
        from torch.utils.data import DataLoader, TensorDataset
        ds = TensorDataset(torch.randn(8, 3), torch.randint(0, 2, (8,)))
        loader = DataLoader(ds, batch_size=4, shuffle=True)
        batch = next(iter(loader))
        assert batch[0].shape == (4, 3)

    def test_save_and_load(self, tmp_path):
        model = torch.nn.Linear(4, 2)
        path = tmp_path / "model.pt"
        torch.save({"state_dict": model.state_dict()}, str(path))
        ckpt = torch.load(str(path), map_location="cpu", weights_only=False)
        assert "state_dict" in ckpt

    def test_onnx_export(self, tmp_path):
        """torch.onnx.export used in export_onnx.py."""
        model = torch.nn.Linear(4, 2)
        model.eval()
        dummy = torch.randn(1, 4)
        path = tmp_path / "test.onnx"
        torch.onnx.export(
            model, dummy, str(path),
            opset_version=17,
            input_names=["input"],
            output_names=["output"],
            dynamic_axes={"input": {0: "batch"}, "output": {0: "batch"}},
        )
        assert path.exists()

    def test_xavier_init(self):
        linear = torch.nn.Linear(8, 4)
        torch.nn.init.xavier_uniform_(linear.weight)
        torch.nn.init.zeros_(linear.bias)
        assert linear.bias.sum().item() == 0.0

    def test_parameter(self):
        """nn.Parameter used for learnable positional encoding."""
        pe = torch.nn.Parameter(torch.randn(1, 16, 32) * 0.02)
        assert pe.requires_grad


class TestTorchvision:
    """Verify torchvision is importable (used as dependency)."""

    def test_version_in_range(self):
        v = _get_version("torchvision")
        assert v >= Version("0.16.0")


class TestTorchaudio:
    """Verify torchaudio is importable (listed in requirements)."""

    def test_import(self):
        import torchaudio
        v = Version(torchaudio.__version__)
        assert v >= Version("2.1.0")


# ===================================================================
# 2. OpenCV
# ===================================================================


class TestOpenCV:
    """Verify opencv-python features used across src/data and src/inference."""

    def test_version_in_range(self):
        import cv2
        v = Version(cv2.__version__)
        assert v >= Version("4.8.0")

    def test_video_capture_properties(self):
        import cv2
        assert hasattr(cv2, "VideoCapture")
        assert hasattr(cv2, "CAP_PROP_FPS")
        assert hasattr(cv2, "CAP_PROP_FRAME_WIDTH")

    def test_cvt_color_constants(self):
        import cv2
        assert hasattr(cv2, "COLOR_BGR2RGB")
        assert hasattr(cv2, "COLOR_BGR2GRAY")

    def test_resize(self):
        import cv2
        img = np.random.randint(0, 255, (480, 640, 3), dtype=np.uint8)
        resized = cv2.resize(img, (224, 224))
        assert resized.shape == (224, 224, 3)

    def test_imwrite_and_imread(self, tmp_path):
        import cv2
        img = np.random.randint(0, 255, (100, 100, 3), dtype=np.uint8)
        path = str(tmp_path / "test.jpg")
        cv2.imwrite(path, img)
        loaded = cv2.imread(path)
        assert loaded is not None and loaded.shape == (100, 100, 3)

    def test_rectangle_and_puttext(self):
        import cv2
        img = np.zeros((100, 200, 3), dtype=np.uint8)
        cv2.rectangle(img, (10, 10), (50, 50), (255, 0, 0), -1)
        cv2.putText(img, "test", (10, 80), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1, cv2.LINE_AA)
        assert img.max() > 0

    def test_add_weighted(self):
        import cv2
        a = np.full((50, 50, 3), 200, dtype=np.uint8)
        b = np.full((50, 50, 3), 100, dtype=np.uint8)
        blended = cv2.addWeighted(a, 0.7, b, 0.3, 0)
        assert blended.shape == (50, 50, 3)

    def test_cvt_color(self):
        import cv2
        bgr = np.random.randint(0, 255, (50, 50, 3), dtype=np.uint8)
        rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
        assert rgb.shape == bgr.shape
        # Red and blue channels should be swapped
        np.testing.assert_array_equal(bgr[:, :, 0], rgb[:, :, 2])


# ===================================================================
# 3. MediaPipe
# ===================================================================


class TestMediaPipe:
    """Verify mediapipe import paths used in preprocess.py and live_demo.py."""

    def test_version_in_range(self):
        import mediapipe as mp
        v = Version(mp.__version__)
        assert v >= Version("0.10.7")
        assert v <= Version("0.10.14")

    def test_holistic_import(self):
        """_import_mediapipe_holistic fallback chain from preprocess.py."""
        from src.data.preprocess import _import_mediapipe_holistic
        holistic_mod = _import_mediapipe_holistic()
        assert hasattr(holistic_mod, "Holistic")

    def test_drawing_import(self):
        from src.data.preprocess import _import_mediapipe_drawing
        holistic_mod, drawing_mod, styles_mod = _import_mediapipe_drawing()
        # At least holistic should be available
        assert holistic_mod is not None

    def test_holistic_instantiation(self):
        from src.data.preprocess import _import_mediapipe_holistic
        mp_holistic = _import_mediapipe_holistic()
        h = mp_holistic.Holistic(
            static_image_mode=True,
            model_complexity=0,
            min_detection_confidence=0.5,
            min_tracking_confidence=0.5,
        )
        h.close()


# ===================================================================
# 4. NumPy
# ===================================================================


class TestNumPy:
    """Verify numpy features used heavily across all src/ modules."""

    def test_version_in_range(self):
        v = Version(np.__version__)
        assert v >= Version("1.24.0")

    def test_ndarray_creation(self):
        a = np.zeros((10, 543, 3), dtype=np.float32)
        assert a.shape == (10, 543, 3)

    def test_random_operations(self):
        arr = np.random.rand(30, 543, 3).astype(np.float32)
        assert arr.shape == (30, 543, 3)
        # random.beta used in augmentation sampling
        lam = np.random.beta(0.2, 0.2)
        assert 0.0 <= lam <= 1.0

    def test_linalg_norm(self):
        """Used in normalize_keypoints and preprocess.py."""
        v = np.array([[3.0, 4.0, 0.0]])
        norm = np.linalg.norm(v, axis=-1)
        np.testing.assert_allclose(norm, [5.0])

    def test_save_and_load(self, tmp_path):
        arr = np.random.rand(20, 543, 3).astype(np.float32)
        path = str(tmp_path / "kps.npy")
        np.save(path, arr)
        loaded = np.load(path)
        np.testing.assert_array_equal(arr, loaded)

    def test_stack_and_concatenate(self):
        frames = [np.random.rand(543, 3).astype(np.float32) for _ in range(10)]
        stacked = np.stack(frames, axis=0)
        assert stacked.shape == (10, 543, 3)
        extra = np.zeros((5, 543, 3), dtype=np.float32)
        cat = np.concatenate([stacked, extra], axis=0)
        assert cat.shape == (15, 543, 3)

    def test_linspace_and_round(self):
        """Uniform frame sampling used in TemporalCrop and datasets."""
        indices = np.linspace(0, 99, 32, dtype=np.float64)
        indices = np.round(indices).astype(np.int64)
        assert len(indices) == 32
        assert indices[0] == 0
        assert indices[-1] == 99

    def test_tile(self):
        """Frame padding via tile used in dataset _pad_or_crop."""
        frame = np.random.rand(1, 543, 3).astype(np.float32)
        tiled = np.tile(frame, (5, 1, 1))
        assert tiled.shape == (5, 543, 3)

    def test_bincount(self):
        """Used in get_dataloader for class weights."""
        labels = np.array([0, 1, 1, 2, 0, 0])
        counts = np.bincount(labels)
        np.testing.assert_array_equal(counts, [3, 2, 1])

    def test_fill_diagonal(self):
        """Used in find_hard_negatives."""
        m = np.ones((3, 3))
        np.fill_diagonal(m, 0)
        assert m[0, 0] == 0 and m[1, 1] == 0

    def test_deg2rad_cos_sin(self):
        """Used in KeypointRotation."""
        rad = np.deg2rad(90)
        np.testing.assert_allclose(np.cos(rad), 0.0, atol=1e-7)
        np.testing.assert_allclose(np.sin(rad), 1.0, atol=1e-7)

    def test_median(self):
        """Used for shoulder width scale in normalize_keypoints."""
        vals = np.array([1.0, 2.0, 3.0, 4.0, 5.0])
        assert np.median(vals) == 3.0

    def test_reshape_flatten(self):
        """(T, K, C) -> (T, K*C) flatten used in datasets."""
        arr = np.random.rand(32, 543, 6).astype(np.float32)
        flat = arr.reshape(32, -1)
        assert flat.shape == (32, 543 * 6)


# ===================================================================
# 5. Pandas
# ===================================================================


class TestPandas:
    """Verify pandas features used in dataset.py, preprocess.py, evaluate.py."""

    def test_version_in_range(self):
        import pandas as pd
        v = Version(pd.__version__)
        assert v >= Version("2.0.0")

    def test_dataframe_creation_and_csv(self, tmp_path):
        import pandas as pd
        rows = [
            {"video_id": "v001", "label_idx": 0, "gloss": "book"},
            {"video_id": "v002", "label_idx": 1, "gloss": "drink"},
        ]
        df = pd.DataFrame(rows)
        csv_path = tmp_path / "split.csv"
        df.to_csv(csv_path, index=False)
        loaded = pd.read_csv(csv_path)
        assert len(loaded) == 2
        assert list(loaded.columns) == ["video_id", "label_idx", "gloss"]

    def test_iterrows(self):
        import pandas as pd
        df = pd.DataFrame({"a": [1, 2], "b": [3, 4]})
        rows = list(df.iterrows())
        assert len(rows) == 2

    def test_nunique(self):
        import pandas as pd
        df = pd.DataFrame({"label_idx": [0, 0, 1, 1, 2]})
        assert df["label_idx"].nunique() == 3

    def test_reset_index(self):
        import pandas as pd
        df = pd.DataFrame({"x": [10, 20, 30]}, index=[5, 6, 7])
        df2 = df.reset_index(drop=True)
        assert list(df2.index) == [0, 1, 2]

    def test_boolean_filter(self):
        import pandas as pd
        df = pd.DataFrame({"split": ["train", "val", "test", "train"]})
        train_df = df[df["split"] == "train"]
        assert len(train_df) == 2

    def test_apply(self):
        """df.apply used in WLASLKeypointDataset to filter valid .npy files."""
        import pandas as pd
        df = pd.DataFrame({"video_id": ["a", "b", "c"]})
        mask = df["video_id"].apply(lambda x: x in ("a", "c"))
        filtered = df[mask]
        assert len(filtered) == 2


# ===================================================================
# 6. PyYAML
# ===================================================================


class TestPyYAML:
    """Verify yaml.safe_load and yaml.dump used in config.py."""

    def test_version_in_range(self):
        import yaml
        v = Version(yaml.__version__)
        assert v >= Version("6.0.1")

    def test_safe_load(self):
        import yaml
        data = yaml.safe_load("approach: stgcn_proto\nwlasl_variant: 100\n")
        assert data["approach"] == "stgcn_proto"
        assert data["wlasl_variant"] == 100

    def test_dump(self):
        import yaml
        data = {"approach": "stgcn_proto", "T": 32, "fp16": True}
        text = yaml.dump(data, default_flow_style=False, sort_keys=False)
        reloaded = yaml.safe_load(text)
        assert reloaded == data

    def test_roundtrip(self, tmp_path):
        import yaml
        data = {"lr": 0.001, "epochs": 100, "use_motion": True}
        path = tmp_path / "cfg.yaml"
        with open(path, "w") as f:
            yaml.dump(data, f)
        with open(path) as f:
            loaded = yaml.safe_load(f)
        assert loaded == data


# ===================================================================
# 7. scikit-learn
# ===================================================================


class TestScikitLearn:
    """Verify sklearn.metrics.confusion_matrix used in train.py and evaluate.py."""

    def test_version_in_range(self):
        import sklearn
        v = Version(sklearn.__version__)
        assert v >= Version("1.3.0")

    def test_confusion_matrix(self):
        from sklearn.metrics import confusion_matrix
        y_true = [0, 1, 2, 0, 1, 2]
        y_pred = [0, 1, 1, 0, 2, 2]
        cm = confusion_matrix(y_true, y_pred, labels=[0, 1, 2])
        assert cm.shape == (3, 3)
        assert cm[0, 0] == 2  # class 0 correct
        assert cm[1, 2] == 1  # class 1 predicted as 2

    def test_confusion_matrix_with_labels(self):
        """Explicit labels kwarg used in validate() and compute_metrics()."""
        from sklearn.metrics import confusion_matrix
        y_true = [0, 0]
        y_pred = [0, 0]
        cm = confusion_matrix(y_true, y_pred, labels=list(range(5)))
        assert cm.shape == (5, 5)


# ===================================================================
# 8. Matplotlib
# ===================================================================


class TestMatplotlib:
    """Verify matplotlib features used in evaluate.py plot_confusion_matrix."""

    def test_version_in_range(self):
        import matplotlib
        v = Version(matplotlib.__version__)
        assert v >= Version("3.7.0")

    def test_agg_backend(self):
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        fig, ax = plt.subplots(figsize=(4, 4))
        assert fig is not None and ax is not None
        plt.close(fig)

    def test_savefig(self, tmp_path):
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        fig, ax = plt.subplots()
        ax.plot([1, 2, 3], [1, 4, 9])
        path = tmp_path / "plot.png"
        fig.savefig(str(path), dpi=150, bbox_inches="tight")
        plt.close(fig)
        assert path.exists()


# ===================================================================
# 9. Seaborn
# ===================================================================


class TestSeaborn:
    """Verify seaborn.heatmap used in plot_confusion_matrix."""

    def test_version_in_range(self):
        import seaborn
        v = Version(seaborn.__version__)
        assert v >= Version("0.12.0")

    def test_heatmap(self, tmp_path):
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        import seaborn as sns
        data = np.random.rand(5, 5)
        fig, ax = plt.subplots()
        sns.heatmap(data, annot=True, fmt=".2f", cmap="Blues", ax=ax, vmin=0, vmax=1)
        path = tmp_path / "heatmap.png"
        fig.savefig(str(path))
        plt.close(fig)
        assert path.exists()


# ===================================================================
# 10. tqdm
# ===================================================================


class TestTqdm:
    """Verify tqdm progress bars used in train.py, evaluate.py, preprocess.py."""

    def test_version_in_range(self):
        import tqdm as tqdm_mod
        v = Version(tqdm_mod.__version__)
        assert v >= Version("4.66.0")

    def test_basic_iteration(self):
        from tqdm import tqdm
        items = list(tqdm(range(10), desc="test", leave=False, disable=True))
        assert items == list(range(10))


# ===================================================================
# 11. ONNX
# ===================================================================


class TestOnnx:
    """Verify onnx features used in export_onnx.py."""

    def test_version_in_range(self):
        import onnx
        v = Version(onnx.__version__)
        assert v >= Version("1.15.0")

    def test_load_and_check(self, tmp_path):
        import onnx
        model = torch.nn.Linear(4, 2)
        model.eval()
        path = tmp_path / "test.onnx"
        torch.onnx.export(model, torch.randn(1, 4), str(path), opset_version=17)
        onnx_model = onnx.load(str(path))
        onnx.checker.check_model(onnx_model)


# ===================================================================
# 12. ONNX Runtime
# ===================================================================


class TestOnnxRuntime:
    """Verify onnxruntime features used in export_onnx.py verify/benchmark."""

    def test_version_in_range(self):
        import onnxruntime
        v = Version(onnxruntime.__version__)
        assert v >= Version("1.16.0")

    def test_inference_session(self, tmp_path):
        import onnxruntime as ort
        model = torch.nn.Linear(4, 2)
        model.eval()
        path = tmp_path / "test.onnx"
        torch.onnx.export(
            model, torch.randn(1, 4), str(path),
            opset_version=17,
            input_names=["input"],
            output_names=["output"],
        )
        sess = ort.InferenceSession(str(path))
        dummy = np.random.randn(1, 4).astype(np.float32)
        result = sess.run(None, {"input": dummy})
        assert result[0].shape == (1, 2)

    def test_session_options(self):
        import onnxruntime as ort
        opts = ort.SessionOptions()
        opts.inter_op_num_threads = 1
        opts.intra_op_num_threads = 4
        opts.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL


# ===================================================================
# 13. einops
# ===================================================================


class TestEinops:
    """Verify einops (in requirements, may be used by pytorchvideo/timm deps)."""

    def test_version_in_range(self):
        import einops
        v = Version(einops.__version__)
        assert v >= Version("0.7.0")

    def test_rearrange(self):
        from einops import rearrange
        t = torch.randn(2, 3, 4)
        out = rearrange(t, "b c h -> b h c")
        assert out.shape == (2, 4, 3)


# ===================================================================
# 14. timm
# ===================================================================


class TestTimm:
    """Verify timm (in requirements, used as a transitive dep for models)."""

    def test_version_in_range(self):
        import timm
        v = Version(timm.__version__)
        assert v >= Version("0.9.10")

    def test_list_models(self):
        import timm
        models = timm.list_models("resnet*", pretrained=False)
        assert len(models) > 0


# ===================================================================
# 15. pytorchvideo
# ===================================================================


class TestPytorchVideo:
    """Verify pytorchvideo (optional video processing dependency)."""

    def test_import(self):
        import pytorchvideo
        assert hasattr(pytorchvideo, "__version__")

    def test_hub_import(self):
        import pytorchvideo.models.hub  # noqa: F401


# ===================================================================
# 16. albumentations
# ===================================================================


class TestAlbumentations:
    """Verify albumentations (used for video frame transforms in dataset.py)."""

    def test_version_in_range(self):
        import albumentations
        v = Version(albumentations.__version__)
        assert v >= Version("1.3.1")

    def test_compose_and_resize(self):
        import albumentations as A
        transform = A.Compose([A.Resize(224, 224)])
        img = np.random.randint(0, 255, (480, 640, 3), dtype=np.uint8)
        result = transform(image=img)
        assert result["image"].shape == (224, 224, 3)


# ===================================================================
# 17. wandb
# ===================================================================


class TestWandb:
    """Verify wandb is importable (used optionally in train.py)."""

    def test_version_in_range(self):
        import wandb
        v = Version(wandb.__version__)
        assert v >= Version("0.16.0")

    def test_import_attributes(self):
        import wandb
        assert hasattr(wandb, "init")
        assert hasattr(wandb, "log")
        assert hasattr(wandb, "finish")


# ===================================================================
# 18. TensorBoard
# ===================================================================


class TestTensorBoard:
    """Verify tensorboard + torch SummaryWriter used in train.py."""

    def test_version_in_range(self):
        import tensorboard
        v = Version(tensorboard.__version__)
        assert v >= Version("2.14.0")

    def test_summary_writer(self, tmp_path):
        from torch.utils.tensorboard import SummaryWriter
        writer = SummaryWriter(log_dir=str(tmp_path / "tb_logs"))
        writer.add_scalar("test/loss", 0.5, 0)
        writer.add_scalar("test/loss", 0.3, 1)
        writer.close()
        # Check that log files were created
        log_files = list((tmp_path / "tb_logs").iterdir())
        assert len(log_files) > 0


# ===================================================================
# 19. rich
# ===================================================================


class TestRich:
    """Verify rich (used for CLI output formatting)."""

    def test_version_in_range(self):
        from importlib.metadata import version as pkg_version
        v = Version(pkg_version("rich"))
        assert v >= Version("13.0.0")

    def test_console(self):
        from rich.console import Console
        console = Console(file=None, force_terminal=False)
        assert console is not None


# ===================================================================
# 20. gdown
# ===================================================================


class TestGdown:
    """Verify gdown (used in download_wlasl.py)."""

    def test_version_in_range(self):
        import gdown
        v = Version(gdown.__version__)
        assert v >= Version("5.0.0")

    def test_download_callable(self):
        import gdown
        assert callable(gdown.download)


# ===================================================================
# 21. kaggle
# ===================================================================


class TestKaggle:
    """Verify kaggle (used in download_kaggle.py)."""

    def test_import(self):
        import kaggle
        assert hasattr(kaggle, "api")


# ===================================================================
# 22. av (PyAV)
# ===================================================================


class TestAV:
    """Verify av / PyAV (video I/O, listed in requirements)."""

    def test_version_in_range(self):
        import av
        v = Version(av.__version__)
        assert v >= Version("11.0.0")

    def test_open_memory(self):
        """Verify av can create an output container (basic functionality)."""
        import av
        import io
        buf = io.BytesIO()
        try:
            container = av.open(buf, mode="w", format="mp4")
            container.close()
        except av.error.InvalidDataError:
            pass  # OK — we just verify the API is callable


# ===================================================================
# 23. Packaging (used by this test file itself to parse versions)
# ===================================================================


class TestPackaging:
    """Verify packaging module for version comparisons."""

    def test_version_comparison(self):
        from packaging.version import Version
        assert Version("2.1.0") < Version("2.2.0")
        assert Version("1.26.4") >= Version("1.24.0")


# ===================================================================
# 24. Integration: src modules import cleanly
# ===================================================================


class TestSrcImports:
    """Verify all src modules can be imported without errors."""

    def test_import_config(self):
        from src.training.config import Config, load_config, save_config
        cfg = Config()
        assert cfg.wlasl_variant == 100

    def test_import_augment(self):
        from src.data.augment import (
            TemporalCrop, TemporalSpeedPerturb, KeypointHorizontalFlip,
            KeypointNoise, KeypointScale, KeypointRotation, KeypointTranslation,
            KeypointDropout, Compose, get_train_transforms, get_val_transforms,
        )
        train_t = get_train_transforms(T=16)
        assert len(train_t.transforms) > 0

    def test_import_dataset(self):
        from src.data.dataset import (
            WLASLKeypointDataset,
            get_dataloader,
        )

    def test_import_preprocess(self):
        from src.data.preprocess import (
            normalize_keypoints, extract_keypoints_mediapipe,
            parse_wlasl_annotations, create_splits, NUM_KEYPOINTS,
        )
        assert NUM_KEYPOINTS == 543

    def test_import_stgcn(self):
        from src.models.stgcn import (
            STGCNEncoder, STGCNBlock, STGCNBranch, SpatialGraphConv,
            build_stgcn_encoder, build_spatial_graph,
        )

    def test_import_prototypical(self):
        from src.models.prototypical import (
            PrototypicalNetwork, build_model,
        )

    def test_import_episode_sampler(self):
        from src.data.episode_sampler import EpisodicBatchSampler

    def test_import_predict(self):
        from src.inference.predict import SignPredictor

    def test_import_live_demo(self):
        from src.inference.live_demo import FrameBuffer, LivePredictor, ASLDisplay

    def test_import_export_onnx(self):
        from src.inference.export_onnx import export_to_onnx, verify_onnx, benchmark_onnx

    def test_import_evaluate(self):
        from src.training.evaluate import (
            compute_metrics, plot_confusion_matrix, find_hard_negatives, evaluate_latency,
        )

    def test_import_train_prototypical(self):
        from src.training.train_prototypical import (
            _split_episode, train_one_epoch, validate, main,
        )
