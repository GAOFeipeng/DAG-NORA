"""Build the three native numerical libraries with the frozen compiler flags."""

from pathlib import Path
import platform
import shutil
import subprocess
import sys


def main():
    if sys.platform != "linux" or platform.machine().lower() not in ("x86_64", "amd64"):
        raise SystemExit("This release supports Linux x86-64, including WSL2.")
    compiler = shutil.which("gcc")
    if compiler is None:
        raise SystemExit("GCC with libquadmath is required (Ubuntu: build-essential).")
    native = Path(__file__).resolve().parents[1] / "src/certfix3_rc6/native"
    for stem in ("native_difference", "quad_inverse", "stable_products"):
        subprocess.run(
            [
                compiler, "-O2", "-std=gnu11", "-shared", "-fPIC",
                "-ffp-contract=off", "-fno-fast-math",
                str(native / f"{stem}.c"), "-o", str(native / f"{stem}.so"),
                "-lquadmath", "-lm",
            ],
            check=True,
        )
    print("Built the three native numerical libraries.")


if __name__ == "__main__":
    main()
