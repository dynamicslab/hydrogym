"""Unit tests of hydrogym.power (no solver needed)."""

import numpy as np
import pytest

from hydrogym.power import PowerConfig, forms_from_info, net_saving, power_forms, power_info, transpiration_forms


def test_forms_ordering_and_single_positive_actuator():
    f = power_forms([0.3])
    assert f["signed"] == f["positive"] == f["abs"] == pytest.approx(0.3)
    f = power_forms([0.5, -0.2, 0.1])
    assert f["signed"] == pytest.approx(0.4)
    assert f["positive"] == pytest.approx(0.6)
    assert f["abs"] == pytest.approx(0.8)
    assert f["signed"] <= f["positive"] <= f["abs"]


def test_reward_unchanged_at_zero_weight():
    cfg = PowerConfig.from_config({"power_cost": "signed"})
    r = -1.2345678901234567
    assert cfg.apply(r, power_forms([3.0, -1.0])) == r


@pytest.mark.parametrize("cost,expected", [("signed", 2.0), ("positive", 3.0), ("abs", 4.0)])
def test_reward_charge(cost, expected):
    cfg = PowerConfig.from_config({"power_weight": 0.5, "power_cost": cost})
    assert cfg.apply(-1.0, power_forms([3.0, -1.0])) == pytest.approx(-1.0 - 0.5 * expected)


def test_config_precedence_and_validation():
    cfg = PowerConfig.from_config({"power_weight": 1.0}, {"power_weight": 2.0}, power_cost="positive")
    assert (cfg.weight, cfg.cost, cfg.sampling) == (2.0, "positive", None)
    assert cfg.resolve_sampling("mean") == "mean"
    assert PowerConfig.from_config(power_sampling="end").resolve_sampling("mean") == "end"
    with pytest.raises(ValueError):
        PowerConfig(cost="bogus")
    with pytest.raises(ValueError):
        PowerConfig(weight=-1.0)
    with pytest.raises(ValueError):
        PowerConfig(sampling="every")
    with pytest.raises(TypeError):
        PowerConfig.from_config(power_wieght=1.0)


def test_info_roundtrip_and_net_saving():
    info = power_info([0.2, -0.1], sampling="mean", drag_coefficient=1.5, objective={"forces": [1.5]})
    assert info["actuation_power"] == pytest.approx(0.1)
    assert info["power_sampling"] == "mean"
    assert info["drag_coefficient"] == 1.5
    np.testing.assert_allclose(info["actuation_power_per_actuator"], [0.2, -0.1])
    assert forms_from_info(info) == power_forms([0.2, -0.1])
    assert net_saving(1.5, 0.1, 2.0) == pytest.approx(0.2)
    with pytest.raises(ValueError):
        power_info([0.1], sampling="sometimes")


def test_transpiration_forms_bewley_hasegawa():
    pi_p = np.array([0.4, -0.3, 0.1])
    pi_k = np.array([0.05, -0.02, 0.0])
    w = np.array([1.0, 2.0, 0.5])
    f = transpiration_forms(pi_p, pi_k, w)
    assert f["signed"] == pytest.approx(0.45 - 0.64 + 0.05)
    assert f["positive"] == pytest.approx(0.45 + 0.05)
    assert f["abs"] == pytest.approx(0.45 + 0.64 + 0.05)
    # separate terms: a point with pi_p = +1, pi_k = -1 costs 1 (positive) and 2 (abs), not 0
    g = transpiration_forms([1.0], [-1.0], [1.0])
    assert (g["signed"], g["positive"], g["abs"]) == (0.0, 1.0, 2.0)
    with pytest.raises(ValueError):
        transpiration_forms([1.0], [1.0, 2.0], [1.0])


def test_forms_with_array_namespace():
    f = power_forms([0.5, -0.2, 0.1], xp=np)
    assert float(f["signed"]) == pytest.approx(0.4)
    assert float(f["positive"]) == pytest.approx(0.6)
    assert float(f["abs"]) == pytest.approx(0.8)
    cfg = PowerConfig(weight=2.0, cost="positive")
    assert float(cfg.apply(np.float64(-1.0), f)) == pytest.approx(-1.0 - 2.0 * 0.6)


def test_transpiration_forms_xp():
    from hydrogym.power import transpiration_forms

    pi_p = np.array([1.0, -2.0, 0.5])
    pi_k = np.array([0.1, -0.3, 0.2])
    w = np.array([0.5, 1.0, 2.0])
    f = transpiration_forms(pi_p, pi_k, w)
    assert f["signed"] == pytest.approx(0.5 * 1.1 + 1.0 * -2.3 + 2.0 * 0.7)
    assert f["positive"] == pytest.approx(0.5 * 1.1 + 0.0 + 2.0 * 0.7)
    assert f["abs"] == pytest.approx(0.5 * 1.1 + 1.0 * 2.3 + 2.0 * 0.7)
    assert f["signed"] <= f["positive"] <= f["abs"]
    g = transpiration_forms(pi_p, pi_k, w, xp=np)
    assert {k: float(v) for k, v in g.items()} == pytest.approx(f)
