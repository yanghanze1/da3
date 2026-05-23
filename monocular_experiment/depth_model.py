from __future__ import annotations

import sys
from dataclasses import dataclass
from importlib import import_module
from pathlib import Path
from typing import Any

import cv2
import numpy as np


@dataclass
class _Da3Runtime:
    model: Any
    torch: Any
    device: str
    repo_path: Path
    model_dir: Path
    model_name: str


@dataclass
class _TensorRtRuntime:
    tensorrt: Any
    torch: Any
    engine: Any
    context: Any
    engine_path: Path
    onnx_path: Path
    input_name: str
    output_names: tuple[str, ...]
    input_shape: tuple[int, ...]
    precision: str


_RUNTIME_CACHE: dict[tuple[str, str, str], _Da3Runtime] = {}
_TRT_RUNTIME_CACHE: dict[str, _TensorRtRuntime] = {}


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


def _select_device(torch_module: Any, device_name: str) -> str:
    if device_name == "auto":
        return "cuda" if torch_module.cuda.is_available() else "cpu"
    if device_name == "cuda" and not torch_module.cuda.is_available():
        raise RuntimeError("DA3 backend requested CUDA, but CUDA is not available.")
    return device_name


def _ensure_repo_importable(repo_path: Path) -> None:
    src_path = repo_path / "src"
    if not src_path.exists():
        raise FileNotFoundError(f"Depth-Anything-3 src directory not found: {src_path}")
    src_path_str = str(src_path)
    if src_path_str not in sys.path:
        sys.path.insert(0, src_path_str)


def _get_runtime(config: dict[str, Any]) -> _Da3Runtime:
    repo_path = _resolve_path(config["repo_path"], config)
    model_dir = _resolve_path(config["model_dir"], config)
    model_name = str(config.get("model_name", "DA3-SMALL")).strip().upper()
    requested_device = str(config.get("device", "auto")).lower()
    cache_key = (str(repo_path), str(model_dir), requested_device)
    if cache_key in _RUNTIME_CACHE:
        return _RUNTIME_CACHE[cache_key]

    if not repo_path.exists():
        raise FileNotFoundError(f"Depth-Anything-3 repository not found: {repo_path}")
    if not model_dir.exists():
        raise FileNotFoundError(f"DA3 local model directory not found: {model_dir}")

    _ensure_extra_python_paths(config)
    _ensure_repo_importable(repo_path)
    torch_module = import_module("torch")
    da3_api = import_module("depth_anything_3.api")
    device = _select_device(torch_module, requested_device)
    model = da3_api.DepthAnything3.from_pretrained(str(model_dir))
    model = model.to(device=torch_module.device(device))
    model.eval()

    runtime = _Da3Runtime(
        model=model,
        torch=torch_module,
        device=device,
        repo_path=repo_path,
        model_dir=model_dir,
        model_name=model_name,
    )
    _RUNTIME_CACHE[cache_key] = runtime
    return runtime


def _mock_relative_depth(frame_bgr: np.ndarray) -> np.ndarray:
    gray = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2GRAY).astype(np.float32) / 255.0
    h, w = gray.shape
    y = np.linspace(0.0, 1.0, h, dtype=np.float32)[:, None]
    x = np.linspace(-1.0, 1.0, w, dtype=np.float32)[None, :]
    depth = 0.8 + 3.2 * (1.0 - y) + 0.25 * (1.0 - gray) + 0.05 * (x**2)
    return np.maximum(depth, 1e-3).astype(np.float32)


def _resize_map_to_frame(image_map: np.ndarray | None, frame_shape: tuple[int, int, int]) -> np.ndarray | None:
    if image_map is None:
        return None
    frame_h, frame_w = frame_shape[:2]
    if image_map.shape[:2] == (frame_h, frame_w):
        return image_map.astype(np.float32, copy=False)
    return cv2.resize(image_map.astype(np.float32), (frame_w, frame_h), interpolation=cv2.INTER_LINEAR)


def _prepare_preprocessed_tensor(
    runtime: _Da3Runtime,
    frame_bgr: np.ndarray,
    *,
    process_res: int,
    process_res_method: str,
) -> tuple[np.ndarray, Any]:
    frame_rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
    imgs_cpu, _, _ = runtime.model._preprocess_inputs(
        [frame_rgb],
        None,
        None,
        process_res,
        process_res_method,
    )
    imgs = imgs_cpu.to(runtime.torch.device(runtime.device), non_blocking=True)[None].float()
    return frame_rgb, imgs.contiguous()


def _artifact_stem(
    runtime: _Da3Runtime,
    input_shape: tuple[int, ...],
    *,
    process_res: int,
    process_res_method: str,
    precision: str,
) -> str:
    model_tag = runtime.model_name.lower().replace("-", "_")
    method_tag = process_res_method.lower().replace("-", "_")
    shape_tag = "x".join(str(value) for value in input_shape)
    return f"{model_tag}_{shape_tag}_{method_tag}_r{process_res}_{precision.lower()}"


def _get_artifact_paths(
    runtime: _Da3Runtime,
    config: dict[str, Any],
    input_shape: tuple[int, ...],
    *,
    process_res: int,
    process_res_method: str,
    precision: str,
) -> tuple[Path, Path]:
    artifacts_dir = _resolve_path(config.get("artifacts_dir", "models/da3/artifacts"), config)
    artifacts_dir.mkdir(parents=True, exist_ok=True)
    stem = _artifact_stem(
        runtime,
        input_shape,
        process_res=process_res,
        process_res_method=process_res_method,
        precision=precision,
    )
    return artifacts_dir / f"{stem}.onnx", artifacts_dir / f"{stem}.engine"


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


def _export_onnx(
    runtime: _Da3Runtime,
    sample_input: Any,
    onnx_path: Path,
    *,
    opset_version: int,
) -> None:
    torch_module = runtime.torch

    class _Da3TensorRtWrapper(torch_module.nn.Module):
        def __init__(self, da3_model: Any) -> None:
            super().__init__()
            self.da3_model = da3_model

        def forward(self, image: Any) -> tuple[Any, Any]:
            output = self.da3_model.model(image, None, None, [], False, False, "saddle_balanced")
            return output["depth"], output["depth_conf"]

    wrapper = _Da3TensorRtWrapper(runtime.model).eval().to(runtime.torch.device(runtime.device))
    onnx_path.parent.mkdir(parents=True, exist_ok=True)
    with torch_module.inference_mode():
        torch_module.onnx.export(
            wrapper,
            (sample_input,),
            str(onnx_path),
            input_names=["image"],
            output_names=["depth", "confidence"],
            opset_version=opset_version,
            do_constant_folding=True,
            dynamo=False,
        )


def _build_tensorrt_engine(
    onnx_path: Path,
    engine_path: Path,
    config: dict[str, Any],
) -> None:
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

    workspace_gb = float(config.get("workspace_size_gb", 4.0))
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
        joined = "\n".join(errors)
        raise RuntimeError(f"TensorRT failed to parse ONNX model:\n{joined}")

    serialized_engine = builder.build_serialized_network(network, config_builder)
    if serialized_engine is None:
        raise RuntimeError(f"TensorRT failed to build engine from: {onnx_path}")

    engine_path.parent.mkdir(parents=True, exist_ok=True)
    engine_path.write_bytes(serialized_engine)


def _load_tensorrt_runtime(
    runtime: _Da3Runtime,
    config: dict[str, Any],
    *,
    onnx_path: Path,
    engine_path: Path,
    input_shape: tuple[int, ...],
) -> _TensorRtRuntime:
    cache_key = str(engine_path)
    if cache_key in _TRT_RUNTIME_CACHE:
        return _TRT_RUNTIME_CACHE[cache_key]

    _ensure_extra_python_paths(config)
    tensorrt_module = import_module("tensorrt")
    logger = tensorrt_module.Logger(tensorrt_module.Logger.ERROR)
    runtime_trt = tensorrt_module.Runtime(logger)
    engine = runtime_trt.deserialize_cuda_engine(engine_path.read_bytes())
    if engine is None:
        raise RuntimeError(f"TensorRT failed to deserialize engine: {engine_path}")

    context = engine.create_execution_context()
    if context is None:
        raise RuntimeError(f"TensorRT failed to create execution context: {engine_path}")

    input_names: list[str] = []
    output_names: list[str] = []
    for index in range(engine.num_io_tensors):
        name = engine.get_tensor_name(index)
        mode = engine.get_tensor_mode(name)
        if mode == tensorrt_module.TensorIOMode.INPUT:
            input_names.append(name)
        else:
            output_names.append(name)

    if len(input_names) != 1:
        raise RuntimeError(f"Expected exactly 1 TensorRT input tensor, got {len(input_names)}")
    if len(output_names) < 1:
        raise RuntimeError("TensorRT engine does not expose any output tensors.")

    trt_runtime = _TensorRtRuntime(
        tensorrt=tensorrt_module,
        torch=runtime.torch,
        engine=engine,
        context=context,
        engine_path=engine_path,
        onnx_path=onnx_path,
        input_name=input_names[0],
        output_names=tuple(output_names),
        input_shape=input_shape,
        precision=str(config.get("precision", "fp16")).lower(),
    )
    _TRT_RUNTIME_CACHE[cache_key] = trt_runtime
    return trt_runtime


def _get_tensorrt_runtime(
    runtime: _Da3Runtime,
    config: dict[str, Any],
    *,
    sample_input: Any,
    process_res: int,
    process_res_method: str,
) -> _TensorRtRuntime:
    if runtime.device != "cuda":
        raise RuntimeError("The TensorRT DA3 backend requires CUDA.")

    precision = str(config.get("precision", "fp16")).lower()
    onnx_path, engine_path = _get_artifact_paths(
        runtime,
        config,
        tuple(int(v) for v in sample_input.shape),
        process_res=process_res,
        process_res_method=process_res_method,
        precision=precision,
    )

    if not engine_path.exists():
        if not bool(config.get("build_engine_if_missing", True)):
            raise FileNotFoundError(
                f"TensorRT engine not found and auto-build is disabled: {engine_path}"
            )
        if not onnx_path.exists():
            _export_onnx(
                runtime,
                sample_input,
                onnx_path,
                opset_version=int(config.get("onnx_opset", 18)),
            )
        _build_tensorrt_engine(onnx_path, engine_path, config)

    return _load_tensorrt_runtime(
        runtime,
        config,
        onnx_path=onnx_path,
        engine_path=engine_path,
        input_shape=tuple(int(v) for v in sample_input.shape),
    )


def _infer_relative_depth_tensorrt(
    frame_bgr: np.ndarray,
    config: dict[str, Any],
    *,
    process_res: int,
    process_res_method: str,
) -> dict[str, Any]:
    runtime = _get_runtime(config)
    _, sample_input = _prepare_preprocessed_tensor(
        runtime,
        frame_bgr,
        process_res=process_res,
        process_res_method=process_res_method,
    )
    trt_runtime = _get_tensorrt_runtime(
        runtime,
        config,
        sample_input=sample_input,
        process_res=process_res,
        process_res_method=process_res_method,
    )

    context = trt_runtime.context
    context.set_input_shape(trt_runtime.input_name, tuple(int(v) for v in sample_input.shape))
    context.set_tensor_address(trt_runtime.input_name, sample_input.data_ptr())

    outputs: dict[str, Any] = {}
    for output_name in trt_runtime.output_names:
        shape = tuple(int(v) for v in context.get_tensor_shape(output_name))
        dtype = _trt_dtype_to_torch(
            trt_runtime.engine.get_tensor_dtype(output_name),
            trt_runtime.torch,
            trt_runtime.tensorrt,
        )
        outputs[output_name] = trt_runtime.torch.empty(shape, device="cuda", dtype=dtype)
        context.set_tensor_address(output_name, outputs[output_name].data_ptr())

    stream = trt_runtime.torch.cuda.current_stream().cuda_stream
    if not context.execute_async_v3(stream):
        raise RuntimeError(f"TensorRT execution failed for engine: {trt_runtime.engine_path}")
    trt_runtime.torch.cuda.synchronize()

    depth_tensor = outputs.get("depth")
    if depth_tensor is None:
        first_output = trt_runtime.output_names[0]
        depth_tensor = outputs[first_output]
    confidence_tensor = outputs.get("confidence")
    if confidence_tensor is None and len(trt_runtime.output_names) > 1:
        confidence_tensor = outputs[trt_runtime.output_names[1]]

    relative_depth = depth_tensor.detach().float().cpu().numpy()[0, 0].astype(np.float32)
    confidence = None
    if confidence_tensor is not None:
        confidence = confidence_tensor.detach().float().cpu().numpy()[0, 0].astype(np.float32)
    relative_depth = _resize_map_to_frame(relative_depth, frame_bgr.shape)
    confidence = _resize_map_to_frame(confidence, frame_bgr.shape)

    return {
        "backend": f"tensorrt_da3:{runtime.model_name}",
        "relative_depth": relative_depth,
        "confidence": confidence,
        "metadata": {
            "model_name": runtime.model_name,
            "model_dir": str(runtime.model_dir),
            "device": runtime.device,
            "process_res": process_res,
            "process_res_method": process_res_method,
            "onnx_path": str(trt_runtime.onnx_path),
            "engine_path": str(trt_runtime.engine_path),
            "input_shape": list(trt_runtime.input_shape),
            "precision": trt_runtime.precision,
        },
    }


def _infer_relative_depth_official(
    frame_bgr: np.ndarray,
    config: dict[str, Any],
    *,
    process_res: int,
    process_res_method: str,
) -> dict[str, Any]:
    runtime = _get_runtime(config)
    frame_rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
    prediction = runtime.model.inference(
        [frame_rgb],
        process_res=process_res,
        process_res_method=process_res_method,
    )
    relative_depth = np.asarray(prediction.depth[0], dtype=np.float32)
    confidence = None
    if getattr(prediction, "conf", None) is not None:
        confidence = np.asarray(prediction.conf[0], dtype=np.float32)
    relative_depth = _resize_map_to_frame(relative_depth, frame_bgr.shape)
    confidence = _resize_map_to_frame(confidence, frame_bgr.shape)

    return {
        "backend": f"official_da3:{runtime.model_name}",
        "relative_depth": relative_depth,
        "confidence": confidence,
        "metadata": {
            "model_name": runtime.model_name,
            "model_dir": str(runtime.model_dir),
            "device": runtime.device,
            "process_res": process_res,
            "process_res_method": process_res_method,
        },
    }


def infer_relative_depth(frame_bgr: np.ndarray, config: dict[str, Any]) -> dict[str, Any]:
    backend = str(config.get("backend", "official")).lower()
    process_res = int(config.get("process_res", 504))
    process_res_method = str(
        config.get("process_res_method", config.get("input_policy", "upper_bound_resize"))
    )

    if backend == "mock":
        depth = _mock_relative_depth(frame_bgr)
        return {
            "backend": "mock_da3",
            "relative_depth": depth,
            "confidence": np.ones_like(depth, dtype=np.float32),
            "metadata": {
                "model_name": str(config.get("model_name", "DA3-SMALL")),
                "process_res": process_res,
                "process_res_method": process_res_method,
            },
        }
    if backend == "official":
        return _infer_relative_depth_official(
            frame_bgr,
            config,
            process_res=process_res,
            process_res_method=process_res_method,
        )
    if backend == "tensorrt":
        return _infer_relative_depth_tensorrt(
            frame_bgr,
            config,
            process_res=process_res,
            process_res_method=process_res_method,
        )
    raise ValueError(f"Unsupported depth backend: {backend}")
