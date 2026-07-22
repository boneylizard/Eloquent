from pathlib import Path


MAIN_SOURCE = (Path(__file__).parent / "main.py").read_text(encoding="utf-8")


def test_process_lifecycle_is_not_exposed_over_http():
    for route in (
        "/system/shutdown",
        "/system/restart",
        "/tts/shutdown",
        "/tts/restart",
    ):
        assert route not in MAIN_SOURCE
