from __future__ import annotations

import sys
from dataclasses import dataclass
from importlib import import_module
from pathlib import Path
from typing import Any

import cv2
import numpy as np

_MEAN = np.array([0.485, 0.456, 0.406], dtype=np.float32)
_STD = np.array([0.229, 0.224, 0.225], dtype=np.float32)


@dataclass
class _PidNetRuntime:
    model: Any
    torch: Any
    functional: Any
    device: str
    weights_path: Path
    arch: str
    num_classes: int


@dataclass
class _PidNetOnnxRuntime:
    session: Any
    onnx_path: Path
    input_name: str
    output_name: str
    providers: tuple[str, ...]
    input_hw: tuple[int, int]


@dataclass
class _PidNetTensorRtRuntime:
    tensorrt: Any
    torch: Any
    engine: Any
    context: Any
    engine_path: Path
    onnx_path: Path
    input_name: str
    output_name: str
    input_shape: tuple[int, ...]
    precision: str


_RUNTIME_CACHE: dict[tuple[str, str, int, str], _PidNetRuntime] = {}
_ONNX_RUNTIME_CACHE: dict[tuple[str, tuple[str, ...]], _PidNetOnnxRuntime] = {}
_TRT_RUNTIME_CACHE: dict[str, _PidNetTensorRtRuntime] = {}


def _resolve_path(path_value: str | Path, config: dict[str, Any]) -> Path:
    candidate = Path(path_value)
    if candidate.is_absolute():
        return candidate
    config_dir = config.get("_config_dir")
    if config_dir:
        return (Path(config_dir).parent / candidate).resolve()
    return candidate.resolve()


def _iter_extra_python_paths(config: dict[str, Any]) -> list[Path]:
    code_root = Path(__file__).resolve().parents[1]
    candidates: list[Path] = [code_root / "third_party" / "python_pkgs"]

    extra_paths = config.get("extra_python_paths", [])
    if isinstance(extra_paths, (str, Path)):
        extra_paths = [extra_paths]
    for extra_path in extra_paths:
        candidates.append(_resolve_path(extra_path, config))
    return candidates


def _ensure_extra_python_paths(config: dict[str, Any]) -> None:
    for candidate in _iter_extra_python_paths(config):
        if not candidate.exists():
            continue
        candidate_str = str(candidate)
        if candidate_str not in sys.path:
            sys.path.insert(0, candidate_str)


def _normalize_arch_name(arch: str) -> str:
    value = str(arch).strip().lower().replace("_", "-")
    if not value.startswith("pidnet-"):
        value = f"pidnet-{value}"
    return value


def _import_official_pidnet(repo_root: Path):
    repo_root_str = str(repo_root)
    if repo_root_str not in sys.path:
        sys.path.insert(0, repo_root_str)
    torch = import_module("torch")
    functional = import_module("torch.nn.functional")
    pidnet_models = import_module("models.pidnet")
    return torch, functional, pidnet_models


def _select_device(torch_module: Any, device_name: str) -> str:
    if device_name == "auto":
        return "cuda" if torch_module.cuda.is_available() else "cpu"
    if device_name == "cuda" and not torch_module.cuda.is_available():
        raise RuntimeError("PIDNet official backend requested CUDA, but CUDA is not available.")
    return device_name


def _load_state_dict(model: Any, torch_module: Any, weights_path: Path) -> None:
    pretrained = torch_module.load(str(weights_path), map_location="cpu")
    if isinstance(pretrained, dict) and "state_dict" in pretrained:
        pretrained = pretrained["state_dict"]

    model_dict = model.state_dict()
    filtered: dict[str, Any] = {}
    for key, value in pretrained.items():
        normalized = key
        if normalized.startswith("module."):
            normalized = normalized[7:]
        if normalized.startswith("model."):
            normalized = normalized[6:]
        if normalized in model_dict and getattr(value, "shape", None) == model_dict[normalized].shape:
            filtered[normalized] = value

    model_dict.update(filtered)
    model.load_state_dict(model_dict, strict=False)


def _get_runtime(config: dict[str, Any]) -> _PidNetRuntime:
    repo_root = _resolve_path(config["official_pidnet_repo_path"], config)
    weights_path = _resolve_path(config["official_pidnet_weights"], config)
    arch = _normalize_arch_name(config.get("official_pidnet_arch", "pidnet-s"))
    num_classes = int(config.get("official_pidnet_num_classes", 19))
    requested_device = str(config.get("official_pidnet_device", "auto")).lower()
    cache_key = (str(repo_root), str(weights_path), num_classes, requested_device)
    if cache_key in _RUNTIME_CACHE:
        return _RUNTIME_CACHE[cache_key]

    if not repo_root.exists():
        raise FileNotFoundError(f"Official PIDNet repository not found: {repo_root}")
    if not weights_path.exists():
        raise FileNotFoundError(f"Official PIDNet weights not found: {weights_path}")

    torch_module, functional, pidnet_models = _import_official_pidnet(repo_root)
    device = _select_device(torch_module, requested_device)
    model = pidnet_models.get_pred_model(arch, num_classes)
    _load_state_dict(model, torch_module, weights_path)
    model = model.to(device)
    model.eval()

    runtime = _PidNetRuntime(
        model=model,
        torch=torch_module,
        functional=functional,
        device=device,
        weights_path=weights_path,
        arch=arch,
        num_classes=num_classes,
    )
    _RUNTIME_CACHE[cache_key] = runtime
    return runtime


def _resolve_input_hw(config: dict[str, Any]) -> tuple[int, int]:
    input_size = config.get("input_size", [1024, 512])
    if not isinstance(input_size, (list, tuple)) or len(input_size) != 2:
        raise ValueError("PIDNet input_size must be a [height, width] pair.")
    return int(input_size[0]), int(input_size[1])


def _artifact_stem(runtime: _PidNetRuntime, input_hw: tuple[int, int], precision: str) -> str:
    h, w = input_hw
    arch_tag = runtime.arch.replace("-", "_")
    return f"{arch_tag}_{h}x{w}_{precision.lower()}"


def _get_artifact_paths(
    runtime: _PidNetRuntime,
    config: dict[str, Any],
    input_hw: tuple[int, int],
    precision: str,
) -> tuple[Path, Path]:
    artifacts_dir = _resolve_path(config.get("artifacts_dir", "models/pidnet/artifacts"), config)
    artifacts_dir.mkdir(parents=True, exist_ok=True)
    stem = _artifact_stem(runtime, input_hw, precision)
    return artifacts_dir / f"{stem}.onnx", artifacts_dir / f"{stem}.engine"


def _prepare_input_tensor(frame_bgr: np.ndarray, input_hw: tuple[int, int]) -> np.ndarray:
    input_h, input_w = input_hw
    resized = cv2.resize(frame_bgr, (input_w, input_h), interpolation=cv2.INTER_LINEAR)
    image = resized.astype(np.float32)[:, :, ::-1] / 255.0
    image = (image - _MEAN) / _STD
    chw = image.transpose((2, 0, 1)).copy()
    return chw[None, ...]


def _filter_onnx_providers(ort_module: Any, requested: list[str] | tuple[str, ...]) -> tuple[str, ...]:
    available = set(ort_module.get_available_providers())
    filtered = [provider for provider in requested if provider in available]
    if filtered:
        return tuple(filtered)
    if "CPUExecutionProvider" in available:
        return ("CPUExecutionProvider",)
    if not available:
        raise RuntimeError("ONNX Runtime did not report any available providers.")
    return (next(iter(available)),)


def _export_onnx(
    runtime: _PidNetRuntime,
    sample_input: Any,
    onnx_path: Path,
    *,
    align_corners: bool,
    opset_version: int,
) -> None:
    torch_module = runtime.torch
    functional = runtime.functional

    class _PidNetWrapper(torch_module.nn.Module):
        def __init__(self, model: Any) -> None:
            super().__init__()
            self.model = model

        def forward(self, image: Any) -> Any:
            logits = self.model(image)
            return functional.interpolate(
                logits,
                size=image.shape[-2:],
                mode="bilinear",
                align_corners=align_corners,
            )

    wrapper = _PidNetWrapper(runtime.model).eval().to(runtime.device)
    onnx_path.parent.mkdir(parents=True, exist_ok=True)
    with torch_module.inference_mode():
        torch_module.onnx.export(
            wrapper,
            (sample_input,),
            str(onnx_path),
            input_names=["image"],
            output_names=["logits"],
            opset_version=opset_version,
            do_constant_folding=True,
            dynamo=False,
        )


def _build_tensorrt_engine(onnx_path: Path, engine_path: Path, config: dict[str, Any]) -> None:
    _ensure_extra_python_paths(config)
    tensorrt_module = import_module("tensorrt")

    logger_level = str(config.get("trt_logger_level", "warning")).lower()
    logger_map = {
        "internal_error": tensorrt_module.Logger.INTERNAL_ERROR,
        "error": tensorrt_module.Logger.ERROR,
        "warning": tensorrt_module.Logger.WARNING,
        "info": tensorrt_module.Logger.INFO,
        "verbose": tensorrt_module.Logger.VERBOSE,
    }
    logger = tensorrt_module.Logger(logger_map.get(logger_level, tensorrt_module.Logger.WARNING))
    builder = tensorrt_module.Builder(logger)
    network = builder.create_network(
        1 << int(tensorrt_module.NetworkDefinitionCreationFlag.EXPLICIT_BATCH)
    )
    parser = tensorrt_module.OnnxParser(network, logger)
    config_builder = builder.create_builder_config()

    workspace_gb = float(config.get("workspace_size_gb", 2.0))
    config_builder.set_memory_pool_limit(
        tensorrt_module.MemoryPoolType.WORKSPACE,
        int(workspace_gb * (1 << 30)),
    )

    precision = str(config.get("precision", "fp16")).lower()
    if precision == "fp16":
        if not builder.platform_has_fast_fp16:
            raise RuntimeError("TensorRT fp16 was requested, but this platform does not support fast fp16.")
        config_builder.set_flag(tensorrt_module.BuilderFlag.FP16)
    elif precision != "fp32":
        raise ValueError(f"Unsupported TensorRT precision: {precision}")

    with onnx_path.open("rb") as handle:
        parse_ok = parser.parse(handle.read())
    if not parse_ok:
        errors = [str(parser.get_error(i)) for i in range(parser.num_errors)]
        raise RuntimeError("TensorRT failed to parse PIDNet ONNX model:\n" + "\n".join(errors))

    serialized_engine = builder.build_serialized_network(network, config_builder)
    if serialized_engine is None:
        raise RuntimeError(f"TensorRT failed to build PIDNet engine from: {onnx_path}")

    engine_path.parent.mkdir(parents=True, exist_ok=True)
    engine_path.write_bytes(serialized_engine)


def _trt_dtype_to_torch(trt_dtype: Any, torch_module: Any, tensorrt_module: Any) -> Any:
    mapping = {
        tensorrt_module.float32: torch_module.float32,
        tensorrt_module.float16: torch_module.float16,
        tensorrt_module.int32: torch_module.int32,
        tensorrt_module.int8: torch_module.int8,
        tensorrt_module.bool: torch_module.bool,
    }
    if trt_dtype not in mapping:
        raise TypeError(f"Unsupported TensorRT tensor dtype: {trt_dtype}")
    return mapping[trt_dtype]


def _get_onnx_runtime(config: dict[str, Any]) -> _PidNetOnnxRuntime:
    _ensure_extra_python_paths(config)
    ort_module = import_module("onnxruntime")
    runtime = _get_runtime(config)
    input_hw = _resolve_input_hw(config)
    onnx_path, _ = _get_artifact_paths(
        runtime,
        config,
        input_hw,
        precision=str(config.get("precision", "fp16")).lower(),
    )
    if not onnx_path.exists():
        sample = _prepare_input_tensor(np.zeros((input_hw[0], input_hw[1], 3), dtype=np.uint8), input_hw)
        sample_tensor = runtime.torch.from_numpy(sample).to(runtime.device)
        _export_onnx(
            runtime,
            sample_tensor,
            onnx_path,
            align_corners=bool(config.get("official_pidnet_align_corners", True)),
            opset_version=int(config.get("onnx_opset", 18)),
        )

    providers = _filter_onnx_providers(
        ort_module,
        list(config.get("onnx_providers", ["CUDAExecutionProvider", "CPUExecutionProvider"])),
    )
    cache_key = (str(onnx_path), providers)
    if cache_key in _ONNX_RUNTIME_CACHE:
        return _ONNX_RUNTIME_CACHE[cache_key]

    session = ort_module.InferenceSession(str(onnx_path), providers=list(providers))
    runtime_onnx = _PidNetOnnxRuntime(
        session=session,
        onnx_path=onnx_path,
        input_name=session.get_inputs()[0].name,
        output_name=session.get_outputs()[0].name,
        providers=providers,
        input_hw=input_hw,
    )
    _ONNX_RUNTIME_CACHE[cache_key] = runtime_onnx
    return runtime_onnx


def _load_tensorrt_runtime(
    runtime: _PidNetRuntime,
    config: dict[str, Any],
    *,
    onnx_path: Path,
    engine_path: Path,
    input_shape: tuple[int, ...],
) -> _PidNetTensorRtRuntime:
    cache_key = str(engine_path)
    if cache_key in _TRT_RUNTIME_CACHE:
        return _TRT_RUNTIME_CACHE[cache_key]

    _ensure_extra_python_paths(config)
    tensorrt_module = import_module("tensorrt")
    logger = tensorrt_module.Logger(tensorrt_module.Logger.ERROR)
    runtime_trt = tensorrt_module.Runtime(logger)
    engine = runtime_trt.deserialize_cuda_engine(engine_path.read_bytes())
    if engine is None:
        raise RuntimeError(f"TensorRT failed to deserialize PIDNet engine: {engine_path}")

    context = engine.create_execution_context()
    if context is None:
        raise RuntimeError(f"TensorRT failed to create PIDNet execution context: {engine_path}")

    input_names: list[str] = []
    output_names: list[str] = []
    for index in range(engine.num_io_tensors):
        name = engine.get_tensor_name(index)
        mode = engine.get_tensor_mode(name)
        if mode == tensorrt_module.TensorIOMode.INPUT:
            input_names.append(name)
        else:
            output_names.append(name)

    if len(input_names) != 1 or len(output_names) != 1:
        raise RuntimeError("PIDNet TensorRT engine must expose exactly one input and one output tensor.")

    runtime_trt_wrapper = _PidNetTensorRtRuntime(
        tensorrt=tensorrt_module,
        torch=runtime.torch,
        engine=engine,
        context=context,
        engine_path=engine_path,
        onnx_path=onnx_path,
        input_name=input_names[0],
        output_name=output_names[0],
        input_shape=input_shape,
        precision=str(config.get("precision", "fp16")).lower(),
    )
    _TRT_RUNTIME_CACHE[cache_key] = runtime_trt_wrapper
    return runtime_trt_wrapper


def _get_tensorrt_runtime(config: dict[str, Any]) -> _PidNetTensorRtRuntime:
    runtime = _get_runtime(config)
    if runtime.device != "cuda":
        raise RuntimeError("PIDNet TensorRT backend requires CUDA.")

    input_hw = _resolve_input_hw(config)
    precision = str(config.get("precision", "fp16")).lower()
    sample = _prepare_input_tensor(np.zeros((input_hw[0], input_hw[1], 3), dtype=np.uint8), input_hw)
    sample_tensor = runtime.torch.from_numpy(sample).to(runtime.device)
    onnx_path, engine_path = _get_artifact_paths(runtime, config, input_hw, precision)

    if not engine_path.exists():
        if not bool(config.get("build_engine_if_missing", True)):
            raise FileNotFoundError(
                f"PIDNet TensorRT engine not found and auto-build is disabled: {engine_path}"
            )
        if not onnx_path.exists():
            _export_onnx(
                runtime,
                sample_tensor,
                onnx_path,
                align_corners=bool(config.get("official_pidnet_align_corners", True)),
                opset_version=int(config.get("onnx_opset", 18)),
            )
        _build_tensorrt_engine(onnx_path, engine_path, config)

    return _load_tensorrt_runtime(
        runtime,
        config,
        onnx_path=onnx_path,
        engine_path=engine_path,
        input_shape=tuple(int(v) for v in sample_tensor.shape),
    )


def infer_official_pidnet(frame_bgr: np.ndarray, config: dict[str, Any]) -> dict[str, Any]:
    runtime = _get_runtime(config)
    input_hw = _resolve_input_hw(config)
    tensor_np = _prepare_input_tensor(frame_bgr, input_hw)
    tensor = runtime.torch.from_numpy(tensor_np).to(runtime.device)

    with runtime.torch.inference_mode():
        logits = runtime.model(tensor)
        logits = runtime.functional.interpolate(
            logits,
            size=tensor.shape[-2:],
            mode="bilinear",
            align_corners=bool(config.get("official_pidnet_align_corners", True)),
        )
        probs = runtime.torch.softmax(logits, dim=1)[0]
        road_indices = [int(v) for v in config.get("official_pidnet_road_class_indices", [0, 1])]
        road_prob = probs[road_indices].sum(dim=0).detach().cpu().numpy().astype(np.float32)
        pred = runtime.torch.argmax(probs, dim=0).detach().cpu().numpy().astype(np.int32)

    road_prob = cv2.resize(road_prob, (frame_bgr.shape[1], frame_bgr.shape[0]), interpolation=cv2.INTER_LINEAR)
    pred = cv2.resize(pred.astype(np.float32), (frame_bgr.shape[1], frame_bgr.shape[0]), interpolation=cv2.INTER_NEAREST).astype(np.int32)
    return {
        "road_probability": road_prob,
        "prediction": pred,
        "backend": f"official_pidnet:{runtime.arch}",
        "weights_path": str(runtime.weights_path),
        "device": runtime.device,
        "metadata": {
            "input_size": [int(input_hw[0]), int(input_hw[1])],
        },
    }


def infer_onnx_pidnet(frame_bgr: np.ndarray, config: dict[str, Any]) -> dict[str, Any]:
    runtime = _get_onnx_runtime(config)
    tensor = _prepare_input_tensor(frame_bgr, runtime.input_hw)
    logits = runtime.session.run([runtime.output_name], {runtime.input_name: tensor})[0]
    logits = np.asarray(logits, dtype=np.float32)
    probs = np.exp(logits - np.max(logits, axis=1, keepdims=True))
    probs = probs / np.maximum(np.sum(probs, axis=1, keepdims=True), 1e-6)
    road_indices = [int(v) for v in config.get("official_pidnet_road_class_indices", [0, 1])]
    road_prob = np.sum(probs[0, road_indices], axis=0).astype(np.float32)
    pred = np.argmax(probs[0], axis=0).astype(np.int32)

    road_prob = cv2.resize(road_prob, (frame_bgr.shape[1], frame_bgr.shape[0]), interpolation=cv2.INTER_LINEAR)
    pred = cv2.resize(pred.astype(np.float32), (frame_bgr.shape[1], frame_bgr.shape[0]), interpolation=cv2.INTER_NEAREST).astype(np.int32)
    return {
        "road_probability": road_prob,
        "prediction": pred,
        "backend": f"onnx_pidnet:{_normalize_arch_name(config.get('official_pidnet_arch', 'pidnet-s'))}",
        "providers": list(runtime.providers),
        "metadata": {
            "onnx_path": str(runtime.onnx_path),
            "input_size": [int(runtime.input_hw[0]), int(runtime.input_hw[1])],
        },
    }


def infer_tensorrt_pidnet(frame_bgr: np.ndarray, config: dict[str, Any]) -> dict[str, Any]:
    runtime = _get_runtime(config)
    trt_runtime = _get_tensorrt_runtime(config)
    input_hw = _resolve_input_hw(config)
    input_np = _prepare_input_tensor(frame_bgr, input_hw)
    input_tensor = runtime.torch.from_numpy(input_np).to(runtime.device)

    context = trt_runtime.context
    context.set_input_shape(trt_runtime.input_name, tuple(int(v) for v in input_tensor.shape))
    context.set_tensor_address(trt_runtime.input_name, input_tensor.data_ptr())

    output_shape = tuple(int(v) for v in context.get_tensor_shape(trt_runtime.output_name))
    output_dtype = _trt_dtype_to_torch(
        trt_runtime.engine.get_tensor_dtype(trt_runtime.output_name),
        trt_runtime.torch,
        trt_runtime.tensorrt,
    )
    output_tensor = trt_runtime.torch.empty(output_shape, device="cuda", dtype=output_dtype)
    context.set_tensor_address(trt_runtime.output_name, output_tensor.data_ptr())

    stream = trt_runtime.torch.cuda.current_stream().cuda_stream
    if not context.execute_async_v3(stream):
        raise RuntimeError(f"TensorRT execution failed for PIDNet engine: {trt_runtime.engine_path}")
    trt_runtime.torch.cuda.synchronize()

    road_indices = [int(v) for v in config.get("official_pidnet_road_class_indices", [0, 1])]
    probs = trt_runtime.torch.softmax(output_tensor.float(), dim=1)
    road_prob = probs[0, road_indices].sum(dim=0).detach().cpu().numpy().astype(np.float32)
    pred = trt_runtime.torch.argmax(probs[0], dim=0).detach().cpu().numpy().astype(np.int32)

    road_prob = cv2.resize(road_prob, (frame_bgr.shape[1], frame_bgr.shape[0]), interpolation=cv2.INTER_LINEAR)
    pred = cv2.resize(pred.astype(np.float32), (frame_bgr.shape[1], frame_bgr.shape[0]), interpolation=cv2.INTER_NEAREST).astype(np.int32)
    return {
        "road_probability": road_prob,
        "prediction": pred,
        "backend": f"tensorrt_pidnet:{runtime.arch}",
        "metadata": {
            "onnx_path": str(trt_runtime.onnx_path),
            "engine_path": str(trt_runtime.engine_path),
            "input_shape": list(trt_runtime.input_shape),
            "input_size": [int(input_hw[0]), int(input_hw[1])],
            "precision": trt_runtime.precision,
        },
    }
