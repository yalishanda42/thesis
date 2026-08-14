import hashlib, io, json, zipfile
import os, sys
sys.path.insert(0, os.path.join("plugin", "reaper"))
import bootstrap
import dn_paths


def test_asset_name_and_urls():
    assert bootstrap.asset_name("0.1.0", "macos-arm64") == "dn-engine-0.1.0-macos-arm64.zip"
    base = "https://example/releases/download"
    assert bootstrap.asset_url(base, "0.1.0", "linux-x64") == \
        "https://example/releases/download/engine-v0.1.0/dn-engine-0.1.0-linux-x64.zip"
    assert bootstrap.sums_url(base, "0.1.0") == \
        "https://example/releases/download/engine-v0.1.0/SHA256SUMS"


def test_release_base_url_env_override(monkeypatch):
    monkeypatch.setenv("DN_RELEASE_BASE_URL", "file:///tmp/rel")
    assert bootstrap.release_base_url() == "file:///tmp/rel"
    monkeypatch.delenv("DN_RELEASE_BASE_URL", raising=False)
    assert bootstrap.GITHUB_OWNER_REPO in bootstrap.release_base_url()


def test_sha256_and_verify(tmp_path):
    p = tmp_path / "a.zip"
    p.write_bytes(b"hello")
    digest = hashlib.sha256(b"hello").hexdigest()
    assert bootstrap.sha256_of(str(p)) == digest
    sums = "{}  a.zip\ndeadbeef  other.zip\n".format(digest)
    assert bootstrap.expected_sum(sums, "a.zip") == digest
    assert bootstrap.expected_sum(sums, "missing.zip") is None
    assert bootstrap.verify(str(p), sums, "a.zip") is True
    assert bootstrap.verify(str(p), sums, "other.zip") is False


def _make_engine_zip(platform_key):
    buf = io.BytesIO()
    exe = "dn-engine.exe" if platform_key == "windows-x64" else "dn-engine"
    with zipfile.ZipFile(buf, "w") as z:
        z.writestr(exe, b"#!binary\n")
        z.writestr("weights/mdn_meta.json", b"{}")
        z.writestr("_internal/marker", b"x")
    return buf.getvalue()


def test_install_happy_path(tmp_path, monkeypatch):
    rp = str(tmp_path / "RES")
    os.makedirs(rp)
    pk = "linux-x64"
    zip_bytes = _make_engine_zip(pk)
    name = bootstrap.asset_name("0.1.0", pk)
    sums = "{}  {}\n".format(hashlib.sha256(zip_bytes).hexdigest(), name)

    def fetch(url):
        if url.endswith("SHA256SUMS"):
            return sums.encode()
        if url.endswith(name):
            return zip_bytes
        raise AssertionError("unexpected url " + url)

    cfg = bootstrap.install(rp, "0.1.0", pk, fetch=fetch)
    assert bootstrap.is_installed(rp, "0.1.0") is True
    assert cfg["engine_path"] == dn_paths.engine_exe(rp, "0.1.0")
    assert cfg["proc_dir"] == dn_paths.weights_dir(rp, "0.1.0")
    assert cfg["engine_version"] == "0.1.0"
    on_disk = json.load(open(dn_paths.config_path(rp)))
    assert on_disk == cfg
    assert os.path.isfile(os.path.join(dn_paths.weights_dir(rp, "0.1.0"), "mdn_meta.json"))


def test_install_checksum_mismatch_raises(tmp_path):
    rp = str(tmp_path / "RES"); os.makedirs(rp)
    pk = "linux-x64"; name = bootstrap.asset_name("0.1.0", pk)
    def fetch(url):
        if url.endswith("SHA256SUMS"):
            return ("deadbeef  " + name + "\n").encode()
        return b"corrupt-bytes"
    try:
        bootstrap.install(rp, "0.1.0", pk, fetch=fetch)
        assert False, "expected BootstrapError"
    except bootstrap.BootstrapError:
        pass
    assert bootstrap.is_installed(rp, "0.1.0") is False


def test_install_prunes_old_versions(tmp_path):
    rp = str(tmp_path / "RES"); os.makedirs(rp)
    old = dn_paths.engine_dir(rp, "0.0.9"); os.makedirs(old)
    open(os.path.join(old, "stale"), "w").close()
    pk = "linux-x64"; zip_bytes = _make_engine_zip(pk); name = bootstrap.asset_name("0.1.0", pk)
    sums = "{}  {}\n".format(hashlib.sha256(zip_bytes).hexdigest(), name)
    fetch = lambda u: sums.encode() if u.endswith("SHA256SUMS") else zip_bytes
    bootstrap.install(rp, "0.1.0", pk, fetch=fetch)
    assert not os.path.isdir(old)
    assert bootstrap.is_installed(rp, "0.1.0")
