"""Remove only generated outputs before replaying the fixed bootstrap inputs."""

from __future__ import annotations

import shutil
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[1]
GENERATED_DIRS = (
    Path("data/datamart"),
    Path("artifacts/development"),
    Path("artifacts/evaluation"),
    Path("artifacts/train"),
)
REQUIRED_SOURCES = (
    Path("config/feature_contract.json"),
    Path("config/development_config.json"),
    Path("data/raw/H1.csv"),
    Path("data/raw/H2.csv"),
    Path("data/external/source_manifest.json"),
)


def main() -> None:
    root = PROJECT_ROOT.resolve(strict=True)
    missing = [str(path) for path in REQUIRED_SOURCES if not (root / path).is_file()]
    if missing:
        raise FileNotFoundError("Bootstrap sources missing: " + ", ".join(missing))

    for relative in GENERATED_DIRS:
        target = root / relative
        resolved = target.resolve(strict=False)
        if root not in resolved.parents or target.is_symlink():
            raise ValueError(f"Unsafe generated-output path: {target}")
        if target.exists():
            if not target.is_dir():
                raise ValueError(f"Expected generated-output directory: {target}")
            shutil.rmtree(target)
            print(f"Removed generated output: {relative}", flush=True)

    print("Bootstrap run state is ready", flush=True)


if __name__ == "__main__":
    main()
