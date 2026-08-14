# PyInstaller one-folder spec for the Dynamics Needed inference engine.
from PyInstaller.utils.hooks import collect_all

datas, binaries, hiddenimports = [], [], []
for pkg in ("lightgbm", "sklearn", "torch"):
    d, b, h = collect_all(pkg)
    datas += d; binaries += b; hiddenimports += h

EXCLUDES = ["partitura", "music21", "matplotlib", "fluidsynth", "pyfluidsynth",
            "jupyter", "ipykernel", "nbconvert",
            "notebook", "IPython"]

a = Analysis(
    ["run_engine.py"],
    pathex=[],
    binaries=binaries,
    datas=datas,
    hiddenimports=hiddenimports + ["drum_dynamics.serve.__main__"],
    excludes=EXCLUDES,
    noarchive=False,
)
pyz = PYZ(a.pure)
exe = EXE(pyz, a.scripts, [], exclude_binaries=True, name="dn-engine",
          console=True, disable_windowed_traceback=False)
coll = COLLECT(exe, a.binaries, a.datas, name="dn-engine")
