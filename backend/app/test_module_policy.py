from backend.app.module_policy import module_enabled


def test_chess_remains_retired_when_environment_requests_it(monkeypatch):
    monkeypatch.setenv("MIRID_AVAILABLE_MODULES", "chess,voice")
    monkeypatch.setenv("MIRID_ENABLED_MODULES", "chess")

    assert module_enabled("chess") is False


def test_chatlog_condenser_remains_retired_when_environment_requests_it(monkeypatch):
    monkeypatch.setenv("MIRID_AVAILABLE_MODULES", "chatlog_condenser,voice")
    monkeypatch.setenv("MIRID_ENABLED_MODULES", "chatlog_condenser")

    assert module_enabled("chatlog_condenser") is False
