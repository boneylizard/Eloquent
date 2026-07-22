import argparse
import platform
import sys


def probe() -> int:
    if platform.system() != "Darwin" or platform.machine().lower() not in {"arm64", "aarch64"}:
        print("unavailable: Apple Silicon is required")
        return 1
    try:
        import mlx.core as mx
        import mlx_lm
    except Exception as error:
        print(f"unavailable: {error}")
        return 1
    print(f"available: MLX {mx.__version__}, MLX LM {mlx_lm.__version__}")
    return 0


def main() -> int:
    parser = argparse.ArgumentParser(add_help=False)
    parser.add_argument("--probe", action="store_true")
    known, _ = parser.parse_known_args()
    if known.probe:
        return probe()

    from mlx_lm.server import main as server_main

    server_main()
    return 0


if __name__ == "__main__":
    sys.exit(main())
