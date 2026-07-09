from pathlib import Path

from blammoe.config import AlignConfig, ExperimentConfig, PoolingConfig

CONFIGS = Path(__file__).resolve().parent.parent / "configs"


def test_rung1_yaml_loads_with_nested_sections():
    cfg = ExperimentConfig.from_yaml(CONFIGS / "rung1.yaml")
    assert cfg.rung == 1
    assert cfg.model_a == "olmoe-base"
    assert cfg.layers == [3, 7, 11]
    # nested sections parse into their dataclasses, not raw dicts
    assert isinstance(cfg.pooling, PoolingConfig)
    assert cfg.pooling.span_chars == 20
    assert isinstance(cfg.align, AlignConfig)
    assert cfg.align.method == "orthogonal_procrustes"


def test_defaults_apply_for_absent_keys():
    cfg = ExperimentConfig(rung=2, model_a="a", model_b="b")
    assert cfg.pooling.boundary_margin == 3
    assert cfg.align.ridge_lambda == 0.0
    assert cfg.seed == 0
