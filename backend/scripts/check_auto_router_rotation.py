import json
import os
import tempfile
from pathlib import Path
import sys


def main() -> int:
    repo_root = Path(__file__).resolve().parents[2]
    backend_root = repo_root / "backend"
    if str(backend_root) not in sys.path:
        sys.path.insert(0, str(backend_root))

    # Isolate Path.home() lookups so we do not touch real user settings.
    with tempfile.TemporaryDirectory(prefix="router-rotation-") as tmp:
        home = Path(tmp)
        os.environ["USERPROFILE"] = str(home)
        os.environ["HOME"] = str(home)
        settings_dir = home / ".LiangLocal"
        settings_dir.mkdir(parents=True, exist_ok=True)
        settings_path = settings_dir / "settings.json"
        settings_path.write_text(
            json.dumps(
                {
                    "apiEndpointRoundRobinEnabled": True,
                    "customApiEndpoints": [
                        {
                            "id": "endpoint-a",
                            "name": "Endpoint A",
                            "model": "provider/a",
                            "url": "https://example-a.test/v1",
                            "enabled": True,
                            "rotate_enabled": True,
                        },
                        {
                            "id": "endpoint-b",
                            "name": "Endpoint B",
                            "model": "provider/b",
                            "url": "https://example-b.test/v1",
                            "enabled": True,
                            "rotate_enabled": True,
                        },
                        {
                            "id": "endpoint-c",
                            "name": "Endpoint C",
                            "model": "provider/c",
                            "url": "https://example-c.test/v1",
                            "enabled": True,
                            "rotate_enabled": True,
                        },
                    ],
                },
                indent=2,
            ),
            encoding="utf-8",
        )

        from app.openai_compat import get_configured_endpoint

        picks = []
        for _ in range(9):
            chosen = get_configured_endpoint("endpoint-a", request_purpose="user_chat")
            picks.append((chosen or {}).get("id"))

        if len(set(picks)) < 2:
            raise AssertionError(f"Router did not rotate across candidates: {picks}")

        for idx in range(1, len(picks)):
            if picks[idx] == picks[idx - 1]:
                raise AssertionError(f"Router repeated consecutively at index {idx}: {picks}")

        print(f"rotation_sequence={picks}")
        print("rotation_check=PASS")
        return 0


if __name__ == "__main__":
    raise SystemExit(main())
