"""PyInstaller entry point: freezes to the `dn-engine` executable."""
import os

# Single-request inference: pin OpenMP/MKL to one thread before torch/lightgbm
# load, as insurance against multi-runtime OpenMP contention in the frozen
# bundle on macOS (see memory/libomp-symlink-fix.md). Harmless for throughput.
os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("MKL_NUM_THREADS", "1")

from drum_dynamics.serve.__main__ import main  # noqa: E402

if __name__ == "__main__":
    main()
