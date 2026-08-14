"""Generate the ReaPack index.xml for the Dynamics Needed package.

Run: python plugin/reaper/reapack/make_index.py --version 0.1.0 \
        --time 2026-08-14T00:00:00Z \
        --raw-base https://raw.githubusercontent.com/yalishanda42/dynamics-needed/main/plugin/reaper
"""
from __future__ import annotations

import argparse
import os
from xml.etree import ElementTree as ET
from xml.dom import minidom

INDEX_NAME = "Dynamics Needed"
CATEGORY = "MIDI Editor"
MAIN = "dynamics_needed.py"
HELPERS = ["dn_paths.py", "bootstrap.py", "engine_client.py", "dn_core.py", "predict_worker.py"]
DESCRIPTION = ("Dynamics Needed - restore humanized drum-note velocities in the "
               "MIDI editor. On first run it downloads its inference engine.")


def build_index(script_version, time_iso, raw_base_url):
    base = raw_base_url.rstrip("/")
    index = ET.Element("index", {"version": "1", "name": INDEX_NAME})
    cat = ET.SubElement(index, "category", {"name": CATEGORY})
    pkg = ET.SubElement(cat, "reapack", {"name": MAIN, "type": "script"})
    meta = ET.SubElement(pkg, "metadata")
    desc = ET.SubElement(meta, "description")
    desc.text = DESCRIPTION
    ver = ET.SubElement(pkg, "version", {"name": script_version, "author": "Dynamics Needed", "time": time_iso})
    # Main file: main="true", installs under the package name (no file attr).
    main_src = ET.SubElement(ver, "source", {"main": "true"})
    main_src.text = "{}/{}".format(base, MAIN)
    for h in HELPERS:
        s = ET.SubElement(ver, "source", {"file": h})
        s.text = "{}/{}".format(base, h)
    rough = ET.tostring(index, encoding="unicode")
    return minidom.parseString(rough).toprettyxml(indent="  ")


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--version", required=True)
    p.add_argument("--time", required=True)
    p.add_argument("--raw-base", required=True)
    p.add_argument("--out", default=os.path.join("plugin", "reaper", "reapack", "index.xml"))
    args = p.parse_args()
    xml = build_index(args.version, args.time, args.raw_base)
    with open(args.out, "w") as fh:
        fh.write(xml)
    print("wrote", args.out)


if __name__ == "__main__":
    main()
