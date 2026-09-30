"""tests/test_artifacts.py — Step 0.3 exit criteria."""
from __future__ import annotations

from m2a.artifacts import ArtifactStore, content_hash


def test_same_inputs_same_hash(tmp_path):
    f = tmp_path / "f.mid"
    f.write_bytes(b"midi")
    h1 = content_hash(str(f), config={"a": 1}, code_version="0.1")
    h2 = content_hash(str(f), config={"a": 1}, code_version="0.1")
    assert h1 == h2


def test_different_config_different_hash(tmp_path):
    f = tmp_path / "f.mid"
    f.write_bytes(b"midi")
    h1 = content_hash(str(f), config={"a": 1}, code_version="0.1")
    h2 = content_hash(str(f), config={"a": 2}, code_version="0.1")
    assert h1 != h2


def test_different_code_version_different_hash(tmp_path):
    f = tmp_path / "f.mid"
    f.write_bytes(b"midi")
    h1 = content_hash(str(f), config={}, code_version="0.1")
    h2 = content_hash(str(f), config={}, code_version="0.2")
    assert h1 != h2


def test_file_order_independent(tmp_path):
    """[v1.1] content_hash sorts file_paths, so call order must not matter."""
    a = tmp_path / "a.wav"
    b = tmp_path / "b.wav"
    a.write_bytes(b"aaa")
    b.write_bytes(b"bbb")
    h1 = content_hash(str(a), str(b), config={}, code_version="0.1")
    h2 = content_hash(str(b), str(a), config={}, code_version="0.1")
    assert h1 == h2


def test_cache_hit_detection(tmp_path):
    store = ArtifactStore(str(tmp_path))
    assert not store.is_complete("s1", "abc")
    store.mark_complete("s1", "abc")
    assert store.is_complete("s1", "abc")


def test_stage_dir_created(tmp_path):
    store = ArtifactStore(str(tmp_path))
    d = store.stage_dir("analysis", "deadbeef")
    assert d.exists() and d.is_dir()


def test_different_stages_independent(tmp_path):
    store = ArtifactStore(str(tmp_path))
    store.mark_complete("render", "abc")
    assert not store.is_complete("score", "abc")
    assert store.is_complete("render", "abc")
