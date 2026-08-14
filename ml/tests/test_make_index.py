import os, sys
from xml.etree import ElementTree as ET
sys.path.insert(0, os.path.join("plugin", "reaper", "reapack"))
import make_index


def test_index_has_package_and_sources():
    xml = make_index.build_index("0.1.0", "2026-08-14T00:00:00Z",
                                 "https://raw.example/yalishanda42/dynamics-needed/main/plugin/reaper")
    root = ET.fromstring(xml)
    assert root.tag == "index"
    cat = root.find("category")
    assert cat.get("name") == "MIDI Editor"
    pkg = cat.find("reapack")
    assert pkg.get("name") == "dynamics_needed.py"
    assert pkg.get("type") == "script"
    ver = pkg.find("version")
    assert ver.get("name") == "0.1.0"
    srcs = [s.get("file") for s in ver.findall("source")]
    # main script has no file attr (installs under package name); helpers do
    files = set(f for f in srcs if f)
    assert {"dn_paths.py", "bootstrap.py", "engine_client.py", "dn_core.py",
            "predict_worker.py"}.issubset(files)
    assert any(s.text and s.text.endswith("dynamics_needed.py") for s in ver.findall("source"))
