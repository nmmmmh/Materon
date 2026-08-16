"""Contract and regression checks for test v36 protocol static in the TF2.9 multi-backbone engine."""

from pathlib import Path
from materon_tf29_engine.training.task_registry import (
    ALL_SMALL_TASKS,
    STRUCTURE_SMALL_TASKS,
    COMPOSITION_SMALL_TASKS,
)

ROOT = Path(__file__).resolve().parents[1]


def test_small_task_registry():
    assert len(ALL_SMALL_TASKS) == 6
    assert set(STRUCTURE_SMALL_TASKS) == {
        "matbench_dielectric",
        "matbench_jdft2d",
        "matbench_phonons",
    }
    assert set(COMPOSITION_SMALL_TASKS) == {
        "matbench_expt_gap",
        "matbench_expt_is_metal",
        "matbench_steels",
    }


def test_formal_identity_is_structural():
    text = (ROOT / "scripts" / "run_seed.py").read_text(encoding="utf-8")
    assert "assert_structural_identity" in text
    assert "--strict-cross-graph-identity" in text
    assert "cross_graph_diagnostic_only" in text


def test_persistent_cache_has_lock_and_complete_marker():
    text = (
        ROOT / "materon" / "training" / "cache.py"
    ).read_text(encoding="utf-8")
    assert "fcntl.flock" in text
    assert "COMPLETE" in text
    assert "GRAPH CACHE HIT" in text or "GRAPH CACHE %s" in text


def test_epoch_logger_prints_performance():
    text = (
        ROOT / "materon" / "training" / "callbacks.py"
    ).read_text(encoding="utf-8")
    assert "train_mae" in text
    assert "train_roc_auc" in text
    assert "throughput" in text
    assert "alpha" in text


def test_launcher_serializes_cache_path_and_does_not_rebuild_per_worker():
    text=(ROOT/"scripts"/"launch_4gpu.py").read_text(encoding="utf-8")
    assert 'config["cache_dir"] = str(args.cache_dir)' in text
    assert "--rebuild-cache" not in text


def test_resume_logs_are_appended():
    text=(ROOT/"scripts"/"launch_4gpu.py").read_text(encoding="utf-8")
    assert 'open(log_path, "a"' in text
    assert "[RESUME/START]" in text
