import hashlib
import os, sys
sys.path.insert(0, os.path.join("plugin", "reaper"))
import bootstrap


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
