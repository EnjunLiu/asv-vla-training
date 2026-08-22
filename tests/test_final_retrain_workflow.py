from pathlib import Path


ROOT = Path("D:/asv-vla-training")


def test_final_collector_is_isolated_and_long_enough() -> None:
    script = (ROOT / "scripts/collect_final_retrain.ps1").read_text(encoding="utf-8")
    assert "data\\episodes\\final_retrain" in script
    assert "experiments\\final_retrain" not in script
    assert "MaxRuntimeSeconds=215" in script
    assert "refusing to overwrite" in script


def test_final_training_entrypoint_accepts_explicit_paths() -> None:
    script = (ROOT / "run_final_retrain.py").read_text(encoding="utf-8")
    assert "--data-root" in script
    assert "--run-root" in script
    assert "--embedding-path" in script
    assert "final_retrain" in script
