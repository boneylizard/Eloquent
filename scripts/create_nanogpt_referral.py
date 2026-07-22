from __future__ import annotations

import argparse
from getpass import getpass
import json
import os
from pathlib import Path
from urllib.error import HTTPError, URLError
from urllib.request import Request, urlopen


ENDPOINT = "https://nano-gpt.com/api/invitations/create"
DEFAULT_MANIFEST = (
    Path(__file__).resolve().parents[1]
    / "frontend"
    / "src"
    / "config"
    / "providerPromotions.json"
)


def create_referral(api_key: str, issuer_name: str) -> dict:
    payload = json.dumps({
        "type": "referralLink",
        "issuerName": issuer_name,
        "issuerNote": "Referred through Mirid",
    }).encode("utf-8")
    request = Request(
        ENDPOINT,
        data=payload,
        method="POST",
        headers={
            "Content-Type": "application/json",
            "x-api-key": api_key,
        },
    )
    try:
        with urlopen(request, timeout=30) as response:
            result = json.loads(response.read().decode("utf-8"))
    except HTTPError as error:
        detail = error.read().decode("utf-8", errors="replace")
        raise RuntimeError(f"NanoGPT returned HTTP {error.code}: {detail}") from error
    except URLError as error:
        raise RuntimeError(f"NanoGPT could not be reached: {error.reason}") from error
    if not result.get("url") or not result.get("redeemCode"):
        raise RuntimeError("NanoGPT returned no referral URL or redeem code.")
    return result


def write_referral_to_manifest(manifest_path: Path, referral: dict) -> None:
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    promotion = next(
        (item for item in manifest.get("providers", []) if item.get("providerId") == "nanogpt"),
        None,
    )
    if promotion is None:
        raise RuntimeError("The promotions manifest has no NanoGPT provider entry.")
    promotion["status"] = "referral"
    promotion["referralUrl"] = referral["url"]
    manifest_path.write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Create a NanoGPT referral link for Mirid without storing the API key.",
    )
    parser.add_argument("--issuer-name", default="Mirid")
    parser.add_argument("--api-key-env", default="NANOGPT_API_KEY")
    parser.add_argument("--write-manifest", action="store_true")
    parser.add_argument("--manifest", type=Path, default=DEFAULT_MANIFEST)
    args = parser.parse_args()

    api_key = os.environ.get(args.api_key_env, "").strip()
    if not api_key:
        api_key = getpass("NanoGPT API key: ").strip()
    if not api_key:
        parser.error("A NanoGPT API key is required.")

    referral = create_referral(api_key, args.issuer_name)
    if args.write_manifest:
        write_referral_to_manifest(args.manifest.resolve(), referral)

    print(json.dumps({
        "provider": "nanogpt",
        "campaign": "MIRID",
        "referralUrl": referral["url"],
        "redeemCode": referral["redeemCode"],
        "manifestUpdated": bool(args.write_manifest),
        "customerDiscountActive": False,
    }, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
