from __future__ import annotations

import json
import logging
import os
import platform
import secrets
import socket
import subprocess
import sys
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Iterable, Iterator, List, Optional

import requests

from .compute_capabilities import force_cpu_mode


logger = logging.getLogger(__name__)

RUNNER_CONTRACT_VERSION = 1
APPLE_INTELLIGENCE_MODEL_ID = "mirid/apple-intelligence"
TRUE_VALUES = {"1", "true", "yes", "on"}


class LocalRuntimeUnavailable(RuntimeError):
    pass


def current_platform_key(system: Optional[str] = None, machine: Optional[str] = None) -> str:
    system_name = (system or platform.system()).strip().lower()
    machine_name = (machine or platform.machine()).strip().lower()
    architecture = "aarch64" if machine_name in {"arm64", "aarch64"} else "x86_64"
    operating_system = {
        "windows": "windows",
        "darwin": "macos",
        "linux": "linux",
    }.get(system_name, system_name or "unknown")
    return f"{operating_system}-{architecture}"


def _project_root() -> Path:
    return Path(__file__).resolve().parents[2]


def _default_manifest_path() -> Path:
    configured = os.environ.get("MIRID_RUNNER_MANIFEST")
    if configured:
        return Path(configured).expanduser().resolve()
    frozen_root = getattr(sys, "_MEIPASS", None)
    if frozen_root:
        frozen_manifest = Path(frozen_root) / "runners" / "manifest.json"
        if frozen_manifest.is_file():
            return frozen_manifest
    return _project_root() / "runtime" / "model-runners.json"


def _default_runner_root() -> Path:
    configured = os.environ.get("MIRID_RUNNER_ROOT")
    if configured:
        return Path(configured).expanduser().resolve()
    frozen_root = getattr(sys, "_MEIPASS", None)
    if frozen_root:
        return Path(frozen_root) / "runners"
    return _project_root() / "build" / "model-runners"


def _hidden_process_flags() -> Dict[str, Any]:
    if os.name != "nt":
        return {}
    startupinfo = subprocess.STARTUPINFO()
    startupinfo.dwFlags |= subprocess.STARTF_USESHOWWINDOW
    return {
        "creationflags": getattr(subprocess, "CREATE_NO_WINDOW", 0),
        "startupinfo": startupinfo,
    }


@dataclass(frozen=True)
class RunnerProbe:
    runner_id: str
    available: bool
    accelerator: str
    engine: str
    executable: str
    detail: str = ""

    def as_dict(self) -> Dict[str, Any]:
        return {
            "id": self.runner_id,
            "available": self.available,
            "accelerator": self.accelerator,
            "engine": self.engine,
            "detail": self.detail,
        }


class RuntimeRegistry:
    def __init__(
        self,
        manifest_path: Optional[Path] = None,
        runner_root: Optional[Path] = None,
        platform_key: Optional[str] = None,
    ) -> None:
        self.manifest_path = Path(manifest_path or _default_manifest_path())
        self.runner_root = Path(runner_root or _default_runner_root())
        self.platform_key = platform_key or current_platform_key()
        self._manifest = self._read_manifest()
        self._probe_cache: Dict[str, RunnerProbe] = {}

    @property
    def contract_version(self) -> int:
        return int(self._manifest.get("contractVersion", 0))

    @property
    def versions(self) -> Dict[str, str]:
        return dict(self._manifest.get("versions") or {})

    def _read_manifest(self) -> Dict[str, Any]:
        try:
            data = json.loads(self.manifest_path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError) as error:
            logger.warning("Model runner manifest is unavailable: %s", error)
            return {"schemaVersion": 1, "contractVersion": 0, "runners": []}
        if int(data.get("contractVersion", 0)) != RUNNER_CONTRACT_VERSION:
            logger.warning(
                "Model runner contract %s is not supported by backend contract %s",
                data.get("contractVersion"),
                RUNNER_CONTRACT_VERSION,
            )
            return {"schemaVersion": 1, "contractVersion": 0, "runners": []}
        return data

    def executable_for(self, candidate: Dict[str, Any]) -> Path:
        executable = Path(str(candidate.get("executable") or ""))
        return executable if executable.is_absolute() else self.runner_root / executable

    def candidates_for(self, model_format: str = "gguf") -> List[Dict[str, Any]]:
        disabled = os.environ.get("MIRID_NATIVE_RUNNERS", "").strip().lower() in {"0", "false", "no", "off"}
        if disabled:
            return []
        requested = os.environ.get("MIRID_LOCAL_BACKEND", "").strip().lower()
        candidates = [
            candidate
            for candidate in self._manifest.get("runners", [])
            if candidate.get("platform") == self.platform_key
            and model_format in (candidate.get("modelFormats") or [])
        ]
        if force_cpu_mode():
            candidates = [candidate for candidate in candidates if candidate.get("accelerator") == "cpu"]
        elif requested and requested != "auto":
            candidates = [
                candidate
                for candidate in candidates
                if requested in {
                    str(candidate.get("id", "")).lower(),
                    str(candidate.get("accelerator", "")).lower(),
                    str(candidate.get("engine", "")).lower(),
                }
            ]
        return sorted(candidates, key=lambda item: int(item.get("priority", 0)), reverse=True)

    def has_installed_candidate(self, model_format: str = "gguf") -> bool:
        return any(self.executable_for(candidate).is_file() for candidate in self.candidates_for(model_format))

    def probe(self, candidate: Dict[str, Any], refresh: bool = False) -> RunnerProbe:
        runner_id = str(candidate.get("id") or "unknown")
        if not refresh and runner_id in self._probe_cache:
            return self._probe_cache[runner_id]
        executable = self.executable_for(candidate)
        if not executable.is_file():
            result = RunnerProbe(
                runner_id=runner_id,
                available=False,
                accelerator=str(candidate.get("accelerator") or "cpu"),
                engine=str(candidate.get("engine") or "local"),
                executable=str(executable),
                detail="not installed",
            )
            self._probe_cache[runner_id] = result
            return result
        command = [str(executable), *[str(value) for value in candidate.get("probeArgs", [])]]
        try:
            completed = subprocess.run(
                command,
                cwd=executable.parent,
                capture_output=True,
                text=True,
                encoding="utf-8",
                errors="replace",
                timeout=float(os.environ.get("MIRID_RUNNER_PROBE_TIMEOUT", "12")),
                check=False,
                **_hidden_process_flags(),
            )
            output = f"{completed.stdout}\n{completed.stderr}".strip()
            markers = [str(marker).lower() for marker in candidate.get("probeMarkers", [])]
            markers_match = not markers or any(marker in output.lower() for marker in markers)
            available = completed.returncode == 0 and markers_match
            detail = "ready" if available else (output[-500:] or f"exit code {completed.returncode}")
        except (OSError, subprocess.TimeoutExpired) as error:
            available = False
            detail = str(error)
        result = RunnerProbe(
            runner_id=runner_id,
            available=available,
            accelerator=str(candidate.get("accelerator") or "cpu"),
            engine=str(candidate.get("engine") or "local"),
            executable=str(executable),
            detail=detail,
        )
        self._probe_cache[runner_id] = result
        return result

    def available_candidates(self, model_format: str = "gguf", refresh: bool = False) -> List[Dict[str, Any]]:
        return [
            candidate
            for candidate in self.candidates_for(model_format)
            if self.probe(candidate, refresh=refresh).available
        ]

    def capabilities(self, refresh: bool = False, diagnose_all: bool = False) -> Dict[str, Any]:
        formats: Dict[str, Dict[str, Any]] = {}
        probes: List[RunnerProbe] = []
        for model_format in ("gguf", "mlx", "system"):
            selected: Optional[RunnerProbe] = None
            for candidate in self.candidates_for(model_format):
                probe = self.probe(candidate, refresh=refresh)
                probes.append(probe)
                if probe.available and selected is None:
                    selected = probe
                    if not diagnose_all:
                        break
            formats[model_format] = {
                "available": selected is not None,
                "selected": selected.as_dict() if selected else None,
            }
        unique_probes = {probe.runner_id: probe.as_dict() for probe in probes}
        return {
            "contract_version": self.contract_version,
            "platform": self.platform_key,
            "versions": self.versions,
            "formats": formats,
            "runners": list(unique_probes.values()),
        }


def _reserve_loopback_port() -> int:
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as listener:
        listener.bind(("127.0.0.1", 0))
        return int(listener.getsockname()[1])


def _runner_log_path(runner_id: str, port: int) -> Path:
    log_dir = Path.home() / ".LiangLocal" / "logs"
    log_dir.mkdir(parents=True, exist_ok=True)
    safe_runner_id = "".join(char if char.isalnum() or char in {"-", "_"} else "-" for char in runner_id)
    return log_dir / f"model-{safe_runner_id}-{port}.log"


class OpenAICompatibleModel:
    def __init__(
        self,
        candidate: Dict[str, Any],
        executable: Path,
        model_name: str,
        model_path: Optional[str],
        context_length: int,
        params: Optional[Dict[str, Any]] = None,
    ) -> None:
        self.candidate = dict(candidate)
        self.executable = executable
        self.model_name = model_name
        self.model_path = model_path
        self.context_length = int(context_length)
        self.params = dict(params or {})
        self.runtime_id = str(candidate.get("id") or "local")
        self.accelerator = str(candidate.get("accelerator") or "cpu")
        self.gpu_usage_mode = "native_sidecar"
        self.port = _reserve_loopback_port()
        self.api_key = secrets.token_urlsafe(24)
        self.base_url = f"http://127.0.0.1:{self.port}"
        self.process: Optional[subprocess.Popen] = None
        self._log_handle = None
        self._session = requests.Session()

    @property
    def request_model_name(self) -> str:
        if self.candidate.get("launchKind") == "mlx-server":
            return "default_model"
        return self.model_name

    def _command(self) -> List[str]:
        launch_kind = str(self.candidate.get("launchKind") or "llama-server")
        command = [str(self.executable)]
        if launch_kind == "llama-server":
            if not self.model_path:
                raise LocalRuntimeUnavailable("A GGUF model path is required.")
            command.extend([
                "--model", str(self.model_path),
                "--host", "127.0.0.1",
                "--port", str(self.port),
                "--ctx-size", str(self.context_length),
                "--api-key", self.api_key,
                "--parallel", "1",
            ])
            if self.accelerator == "cpu":
                command.extend([
                    "--device", "none",
                    "--gpu-layers", "0",
                    "--threads", str(max(1, (os.cpu_count() or 2) - 1)),
                ])
            else:
                command.extend(["--gpu-layers", "auto"])
            mmproj_path = self.params.get("clip_model_path") or self.params.get("mmproj_path")
            if mmproj_path:
                command.extend(["--mmproj", str(mmproj_path)])
            if self.params.get("embedding"):
                command.append("--embedding")
            tensor_split = self.params.get("tensor_split")
            if isinstance(tensor_split, (list, tuple)) and tensor_split:
                command.extend(["--tensor-split", ",".join(str(value) for value in tensor_split)])
        elif launch_kind == "mlx-server":
            if not self.model_path:
                raise LocalRuntimeUnavailable("An MLX model path or repository is required.")
            command.extend([
                "--model", str(self.model_path),
                "--host", "127.0.0.1",
                "--port", str(self.port),
            ])
        elif launch_kind == "apple-foundation":
            command.extend([
                "--host", "127.0.0.1",
                "--port", str(self.port),
                "--api-key", self.api_key,
            ])
        else:
            raise LocalRuntimeUnavailable(f"Unknown local runner kind: {launch_kind}")
        return command

    def start(self) -> "OpenAICompatibleModel":
        log_path = _runner_log_path(self.runtime_id, self.port)
        self._log_handle = open(log_path, "a", encoding="utf-8")
        self._log_handle.write(f"\n--- starting {time.strftime('%Y-%m-%d %H:%M:%S')} ---\n")
        self._log_handle.flush()
        self.process = subprocess.Popen(
            self._command(),
            cwd=self.executable.parent,
            stdout=self._log_handle,
            stderr=subprocess.STDOUT,
            env=os.environ.copy(),
            **_hidden_process_flags(),
        )
        deadline = time.monotonic() + float(os.environ.get("MIRID_MODEL_START_TIMEOUT", "600"))
        last_error = "starting"
        while time.monotonic() < deadline:
            if self.process.poll() is not None:
                raise LocalRuntimeUnavailable(
                    f"{self.runtime_id} stopped while loading the model. See {log_path}."
                )
            try:
                response = self._session.get(
                    f"{self.base_url}/health",
                    headers=self._headers(),
                    timeout=2,
                )
                if response.status_code == 200:
                    logger.info("Local model ready through %s on port %s", self.runtime_id, self.port)
                    return self
                last_error = f"health status {response.status_code}"
            except requests.RequestException as error:
                last_error = str(error)
            time.sleep(0.5)
        raise LocalRuntimeUnavailable(f"{self.runtime_id} did not become ready: {last_error}")

    def _headers(self) -> Dict[str, str]:
        return {
            "Authorization": f"Bearer {self.api_key}",
            "Content-Type": "application/json",
        }

    def _request(self, path: str, payload: Dict[str, Any], stream: bool = False):
        response = self._session.post(
            f"{self.base_url}{path}",
            headers=self._headers(),
            json=payload,
            stream=stream,
            timeout=(10, float(os.environ.get("MIRID_GENERATION_TIMEOUT", "3600"))),
        )
        if not response.ok:
            try:
                detail = response.json()
            except ValueError:
                detail = response.text
            raise LocalRuntimeUnavailable(f"Local generation failed ({response.status_code}): {detail}")
        if not stream:
            return response.json()
        return self._stream_events(response)

    @staticmethod
    def _stream_events(response: requests.Response) -> Iterator[Dict[str, Any]]:
        try:
            for raw_line in response.iter_lines(decode_unicode=True):
                line = (raw_line or "").strip()
                if not line.startswith("data:"):
                    continue
                data = line[5:].strip()
                if not data or data == "[DONE]":
                    break
                try:
                    yield json.loads(data)
                except json.JSONDecodeError:
                    logger.debug("Ignored malformed local runner event: %s", data[:200])
        finally:
            response.close()

    def create_completion(self, prompt: Optional[str] = None, **kwargs):
        payload = {
            "model": self.request_model_name,
            "prompt": prompt or "",
            "max_tokens": int(kwargs.get("max_tokens", 1024)),
            "temperature": float(kwargs.get("temperature", 0.7)),
            "top_p": float(kwargs.get("top_p", 0.9)),
            "stream": bool(kwargs.get("stream", False)),
        }
        for source, target in (
            ("top_k", "top_k"),
            ("repeat_penalty", "repeat_penalty"),
            ("repetition_penalty", "repeat_penalty"),
            ("stop", "stop"),
            ("seed", "seed"),
            ("echo", "echo"),
        ):
            if source in kwargs and kwargs[source] is not None:
                payload[target] = kwargs[source]
        return self._request("/v1/completions", payload, stream=payload["stream"])

    def create_chat_completion(self, messages: Optional[List[Dict[str, Any]]] = None, **kwargs):
        payload = {
            "model": self.request_model_name,
            "messages": messages or [],
            "max_tokens": int(kwargs.get("max_tokens", 1024)),
            "temperature": float(kwargs.get("temperature", 0.7)),
            "top_p": float(kwargs.get("top_p", 0.9)),
            "stream": bool(kwargs.get("stream", False)),
        }
        for key in ("tools", "tool_choice", "stop", "seed"):
            if key in kwargs and kwargs[key] is not None:
                payload[key] = kwargs[key]
        return self._request("/v1/chat/completions", payload, stream=payload["stream"])

    def __call__(self, prompt: Optional[str] = None, **kwargs):
        if kwargs.get("messages") is not None:
            messages = kwargs.pop("messages")
            return self.create_chat_completion(messages=messages, **kwargs)
        return self.create_completion(prompt=prompt, **kwargs)

    def embed(self, text: str):
        response = self._request(
            "/v1/embeddings",
            {"model": self.request_model_name, "input": text},
        )
        data = response.get("data") or []
        if not data:
            raise LocalRuntimeUnavailable("The local model returned no embedding.")
        return data[0].get("embedding")

    def shutdown(self) -> None:
        process = self.process
        self.process = None
        if process is not None and process.poll() is None:
            process.terminate()
            try:
                process.wait(timeout=10)
            except subprocess.TimeoutExpired:
                process.kill()
                process.wait(timeout=5)
        self._session.close()
        if self._log_handle is not None:
            self._log_handle.close()
            self._log_handle = None

    def unload(self) -> Dict[str, str]:
        self.shutdown()
        return {"status": "success"}

    def __del__(self):
        try:
            self.shutdown()
        except Exception:
            pass


OpenAICompatibleModel.__module__ = "llama_cpp.server"


class LocalRuntimeBroker:
    def __init__(self, registry: Optional[RuntimeRegistry] = None) -> None:
        self.registry = registry or RuntimeRegistry()

    def has_candidates(self, model_format: str = "gguf") -> bool:
        return self.registry.has_installed_candidate(model_format)

    def capabilities(self, refresh: bool = False, diagnose_all: bool = False) -> Dict[str, Any]:
        return self.registry.capabilities(refresh=refresh, diagnose_all=diagnose_all)

    def start_model(
        self,
        model_name: str,
        model_path: Optional[str],
        context_length: int,
        model_format: str = "gguf",
        params: Optional[Dict[str, Any]] = None,
    ) -> OpenAICompatibleModel:
        failures: List[str] = []
        for candidate in self.registry.candidates_for(model_format):
            probe = self.registry.probe(candidate)
            if not probe.available:
                failures.append(f"{probe.runner_id}: {probe.detail}")
                continue
            runner = OpenAICompatibleModel(
                candidate=candidate,
                executable=self.registry.executable_for(candidate),
                model_name=model_name,
                model_path=model_path,
                context_length=context_length,
                params=params,
            )
            try:
                return runner.start()
            except Exception as error:
                runner.shutdown()
                failures.append(f"{probe.runner_id}: {error}")
                logger.warning("Local runner %s failed; trying the next option: %s", probe.runner_id, error)
        detail = "; ".join(failures) if failures else "no compatible runner is installed"
        raise LocalRuntimeUnavailable(f"Mirid could not start a local model: {detail}")


def format_for_model(model_name: str, model_path: Optional[str] = None) -> str:
    if model_name == APPLE_INTELLIGENCE_MODEL_ID:
        return "system"
    candidate = str(model_path or model_name).lower()
    if candidate.startswith("mlx:") or candidate.endswith(".mlx"):
        return "mlx"
    return "gguf"
