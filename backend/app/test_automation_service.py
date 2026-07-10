"""
Tests for voice sculpt automation pipeline.
Run: pytest backend/app/test_automation_service.py -v
"""

from __future__ import annotations

import asyncio
from pathlib import Path

import pytest

from .automation_service import (
    AutomationConfig,
    AutomationService,
    GPUQueue,
    PipelineTask,
    SculptRequest,
    SubprocessResult,
    _MIN_VOICE_PTH_BYTES,
    _is_voice_model_pth_file,
    discover_environment,
    discover_rvc_models,
    write_sculpt_env_bat,
)


def _run(coro):
    return asyncio.get_event_loop().run_until_complete(coro)


@pytest.fixture
def tmp_config(tmp_path, monkeypatch):
    work = tmp_path / "work"
    work.mkdir()
    voices = tmp_path / "voices"
    voices.mkdir()

    from . import automation_service as mod
    monkeypatch.setattr(mod, "_VOICE_REFERENCES_DIR", voices)

    cfg = AutomationConfig(
        work_dir=work,
        max_concurrent=1,
        job_timeout_sec=60.0,
        gpu_id=None,
        yt_dlp_bin="yt-dlp",
        audio_separator_bin="audio-separator",
        audio_separator_model_dir=work / "models",
        uvr_model_filename="UVR-MDX-NET-Voc_FT.onnx",
        applio_root=tmp_path / "applio",
        applio_python=str(tmp_path / "applio" / "python.exe"),
        applio_models_dir=tmp_path / "applio" / "logs",
        applio_default_pth=tmp_path / "applio" / "logs" / "voice.pth",
        applio_default_index=tmp_path / "applio" / "logs" / "voice.index",
        applio_accent_pth=None,
        applio_accent_index=None,
        ffmpeg_bin="ffmpeg",
    )
    (tmp_path / "applio").mkdir()
    (tmp_path / "applio" / "core.py").write_text("# stub")
    (tmp_path / "applio" / "python.exe").write_text("")
    (tmp_path / "applio" / "logs").mkdir(parents=True)
    (tmp_path / "applio" / "logs" / "voice.pth").write_bytes(b"pth")
    (tmp_path / "applio" / "logs" / "voice.index").write_bytes(b"idx")
    return cfg, voices


def test_gpu_queue_serializes(tmp_config):
    queue = GPUQueue()
    order: list[str] = []

    async def fake(label: str):
        order.append(f"{label}_start")
        await asyncio.sleep(0.05)
        order.append(f"{label}_end")
        return SubprocessResult(0, "", "")

    async def run_all():
        await asyncio.gather(
            queue.run(lambda: fake("a")),
            queue.run(lambda: fake("b")),
        )

    _run(run_all())
    assert order.index("a_start") < order.index("a_end")
    assert order.index("b_start") < order.index("b_end")
    assert (order == ["a_start", "a_end", "b_start", "b_end"]) or (
        order == ["b_start", "b_end", "a_start", "a_end"]
    )


def test_pipeline_success_mocked_subprocess(tmp_path, tmp_config, monkeypatch):
    cfg, voices_dir = tmp_config
    from . import automation_service as mod

    input_wav = tmp_path / "input.wav"
    input_wav.write_bytes(b"RIFF" + b"\x00" * 40)

    call_labels: list[str] = []

    async def mock_subprocess(cmd, *, cwd, env, gpu, label):
        call_labels.append(label)
        if label == "uvr":
            out_dir = Path(cmd[cmd.index("--output_dir") + 1])
            out_dir.mkdir(parents=True, exist_ok=True)
            (out_dir / "track_Vocals.wav").write_bytes(b"vocals")
        elif label == "applio":
            out = Path(cmd[cmd.index("--output_path") + 1])
            out.parent.mkdir(parents=True, exist_ok=True)
            out.write_bytes(b"converted")
        elif label == "ffmpeg":
            out = Path(cmd[-1])
            out.parent.mkdir(parents=True, exist_ok=True)
            out.write_bytes(b"RIFF" + b"\x00" * 44)
        return SubprocessResult(0, "ok", "")

    monkeypatch.setattr(
        mod.shutil,
        "which",
        lambda name: f"/usr/bin/{name}" if name in ("audio-separator", "ffmpeg") else None,
    )

    events: list[dict] = []

    async def emit(ev):
        events.append(ev)

    async def run_task():
        task = PipelineTask(
            job_id="testjob",
            request=SculptRequest(
                source=str(input_wav),
                source_type="local_path",
                output_name="test_voice",
                skip_rvc=False,
                skip_uvr=False,
            ),
            work_root=cfg.work_dir,
            config=cfg,
            gpu_queue=GPUQueue(),
            run_subprocess=mock_subprocess,
            emit=emit,
        )
        return await task.run()

    final = _run(run_task())
    assert final.parent == voices_dir
    assert final.name.endswith(".wav")
    assert final.is_file()

    progress_steps = [e["step"] for e in events if e.get("type") == "progress"]
    assert progress_steps == [1, 2, 3]
    assert "uvr" in call_labels
    assert "applio" in call_labels
    assert "ffmpeg" in call_labels


def test_multi_clip_sculpt_then_blend(tmp_path, tmp_config, monkeypatch):
    """Multiple clips: UVR each, morph blend, optional one RVC pass."""
    cfg, voices_dir = tmp_config
    from . import automation_service as mod
    from . import voice_morph as vm

    input_a = tmp_path / "a.wav"
    input_b = tmp_path / "b.wav"
    input_a.write_bytes(b"RIFF" + b"\x00" * 40)
    input_b.write_bytes(b"RIFF" + b"\x00" * 40)

    morph_calls: list[tuple] = []
    applio_outputs: list[str] = []

    def fake_morph(paths, output_path, *, sr=44100, weights=None):
        morph_calls.append((list(paths), weights))
        output_path.parent.mkdir(parents=True, exist_ok=True)
        output_path.write_bytes(b"RIFF" + b"\x00" * 44)
        return output_path

    monkeypatch.setattr(vm, "morph_voice_files", fake_morph)

    async def mock_subprocess(cmd, *, cwd, env, gpu, label):
        if label == "uvr":
            out_dir = Path(cmd[cmd.index("--output_dir") + 1])
            out_dir.mkdir(parents=True, exist_ok=True)
            (out_dir / "track_Vocals.wav").write_bytes(b"vocals")
        elif label == "applio":
            out = Path(cmd[cmd.index("--output_path") + 1])
            out.parent.mkdir(parents=True, exist_ok=True)
            out.write_bytes(b"converted")
            applio_outputs.append(str(out))
        elif label == "ffmpeg":
            out = Path(cmd[-1])
            out.parent.mkdir(parents=True, exist_ok=True)
            out.write_bytes(b"RIFF" + b"\x00" * 44)
        return SubprocessResult(0, "ok", "")

    monkeypatch.setattr(
        mod.shutil,
        "which",
        lambda name: f"/usr/bin/{name}" if name in ("audio-separator", "ffmpeg") else None,
    )

    async def run_task():
        task = PipelineTask(
            job_id="multiclip",
            request=SculptRequest(
                sources=[str(input_a), str(input_b)],
                source_type="local_path",
                output_name="blended_voice",
                combine_mode="morph",
                skip_uvr=True,
                skip_rvc=True,
            ),
            work_root=cfg.work_dir,
            config=cfg,
            gpu_queue=GPUQueue(),
            run_subprocess=mock_subprocess,
            emit=lambda ev: None,
        )
        return await task.run()

    final = _run(run_task())
    assert final.is_file()
    assert final.parent == voices_dir
    assert len(morph_calls) == 1
    assert len(applio_outputs) == 0


def test_preflight_missing_audio_separator(tmp_config):
    cfg, _ = tmp_config
    cfg.audio_separator_bin = "/nonexistent/audio-separator-xyz"
    svc = AutomationService(config=cfg)
    result = _run(svc.preflight(for_rvc=False))
    assert result["ready"] is False
    assert any(m["missing_tool"] == "audio-separator" for m in result["missing"])


def test_sculpt_stream_emits_done(tmp_path, tmp_config, monkeypatch):
    cfg, voices_dir = tmp_config
    from . import automation_service as mod

    input_wav = tmp_path / "raw.wav"
    input_wav.write_bytes(b"RIFF")

    async def mock_subprocess(cmd, *, cwd, env, gpu, label):
        if label == "uvr":
            out_dir = Path(cmd[cmd.index("--output_dir") + 1])
            (out_dir / "x_Vocals.wav").write_bytes(b"v")
        elif label == "applio":
            Path(cmd[cmd.index("--output_path") + 1]).write_bytes(b"c")
        elif label == "ffmpeg":
            Path(cmd[-1]).write_bytes(b"RIFF")
        return SubprocessResult(0, "", "")

    monkeypatch.setattr(mod.shutil, "which", lambda n: f"/bin/{n}")

    svc = AutomationService(config=cfg)
    svc._run_subprocess = mock_subprocess  # type: ignore[method-assign]

    async def collect():
        out = []
        async for ev in svc.sculpt_stream(
            SculptRequest(source=str(input_wav), source_type="local_path", output_name="done_voice")
        ):
            out.append(ev)
        return out

    collected = _run(collect())
    assert any(e.get("type") == "done" for e in collected)
    done = next(e for e in collected if e.get("type") == "done")
    assert done["voice_id"].endswith(".wav")
    assert (voices_dir / done["voice_id"]).is_file()


def test_discover_finds_venv_audio_separator(tmp_path, monkeypatch):
    from . import automation_service as mod

    venv_scripts = tmp_path / "venv" / "Scripts"
    venv_scripts.mkdir(parents=True)
    sep = venv_scripts / "audio-separator.exe"
    sep.write_text("")

    monkeypatch.setattr(mod, "_PROJECT_ROOT", tmp_path)
    found = mod.discover_environment()
    assert found.get("audio_separator_bin") == str(sep)


def test_write_sculpt_env_bat(tmp_path):
    bat = tmp_path / "sculpt.env.bat"
    write_sculpt_env_bat({
        "audio_separator_bin": str(tmp_path / "sep.exe"),
        "ffmpeg_bin": "ffmpeg",
    }, path=bat)
    text = bat.read_text(encoding="utf-8")
    assert "AUDIO_SEPARATOR_BIN=" in text
    assert "FFMPEG_BIN=ffmpeg" in text


def test_parse_huggingface_url():
    from .automation_service import parse_huggingface_url

    repo, rev, file = parse_huggingface_url(
        "https://huggingface.co/SomeAuthor/CoolVoice/tree/main"
    )
    assert repo == "SomeAuthor/CoolVoice"
    assert rev == "main"
    assert file is None

    repo2, rev2, file2 = parse_huggingface_url(
        "https://huggingface.co/SomeAuthor/CoolVoice/resolve/main/model.pth"
    )
    assert repo2 == "SomeAuthor/CoolVoice"
    assert rev2 == "main"
    assert file2 == "model.pth"


def test_resolve_named_applio_model(tmp_path):
    from .automation_service import AutomationConfig

    applio = tmp_path / "applio"
    logs = applio / "logs" / "CoolVoice"
    logs.mkdir(parents=True)
    pth = logs / "CoolVoice.pth"
    pth.write_bytes(b"x" * (_MIN_VOICE_PTH_BYTES + 1))
    index = logs / "CoolVoice.index"
    index.write_text("index")

    cfg = AutomationConfig(
        work_dir=tmp_path / "work",
        max_concurrent=1,
        job_timeout_sec=60,
        gpu_id=None,
        yt_dlp_bin="yt-dlp",
        audio_separator_bin="audio-separator",
        audio_separator_model_dir=tmp_path / "models",
        uvr_model_filename="model.onnx",
        applio_root=applio,
        applio_python=None,
        applio_models_dir=applio / "logs",
        applio_default_pth=None,
        applio_default_index=None,
        applio_accent_pth=None,
        applio_accent_index=None,
        ffmpeg_bin="ffmpeg",
    )
    resolved_pth, resolved_index = cfg.resolve_applio_model("CoolVoice")
    assert resolved_pth == pth
    assert resolved_index == index


def test_youtube_requires_ytdlp(tmp_config, monkeypatch):
    cfg, _ = tmp_config
    from . import automation_service as mod

    monkeypatch.setattr(mod.shutil, "which", lambda n: "/bin/ffmpeg" if n == "ffmpeg" else None)

    svc = AutomationService(config=cfg)
    result = _run(svc.preflight(for_youtube=True, for_rvc=False))
    if not any(m.get("missing_tool") == "yt-dlp" for m in result["missing"]):
        pytest.skip("yt-dlp present on system")
    assert result["ready"] is False


def test_discover_rvc_models_filters_training_stubs(tmp_path):
    applio = tmp_path / "applio"
    logs = applio / "logs"
    (logs / "f0D32k").mkdir(parents=True)
    (logs / "f0D32k" / "f0D32k.pth").write_bytes(b"x" * (_MIN_VOICE_PTH_BYTES + 1))
    (logs / "RealVoice").mkdir()
    voice = logs / "RealVoice" / "RealVoice.pth"
    voice.write_bytes(b"x" * (_MIN_VOICE_PTH_BYTES + 1))

    models = discover_rvc_models(applio)
    assert len(models) == 1
    assert models[0]["name"] == "RealVoice"


def test_preflight_ready_without_index(tmp_path, monkeypatch):
    from . import automation_service as mod

    applio = tmp_path / "applio"
    applio.mkdir()
    (applio / "core.py").write_text("# stub")
    py = applio / "env" / "python.exe"
    py.parent.mkdir(parents=True)
    py.write_text("")
    logs = applio / "logs" / "MyVoice"
    logs.mkdir(parents=True)
    pth = logs / "MyVoice.pth"
    pth.write_bytes(b"x" * (_MIN_VOICE_PTH_BYTES + 1))

    cfg = AutomationConfig(
        work_dir=tmp_path / "work",
        max_concurrent=1,
        job_timeout_sec=60,
        gpu_id=None,
        yt_dlp_bin="yt-dlp",
        audio_separator_bin="/bin/audio-separator",
        audio_separator_model_dir=tmp_path / "models",
        uvr_model_filename="model.onnx",
        applio_root=applio,
        applio_python=str(py),
        applio_models_dir=applio / "logs",
        applio_default_pth=pth,
        applio_default_index=None,
        applio_accent_pth=None,
        applio_accent_index=None,
        ffmpeg_bin="/bin/ffmpeg",
    )
    monkeypatch.setattr(mod.shutil, "which", lambda n: f"/bin/{n}")

    svc = AutomationService(config=cfg)
    result = _run(svc.preflight(for_rvc=True))
    assert result["rvc_ready"] is True
    assert result["ready"] is True
    assert any("index" in w.lower() for w in result["warnings"])
    assert not any(m.get("missing_tool") == "applio-index" for m in result["missing"])
