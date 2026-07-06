"""Smoke tests for MyUncle (tissue_atavism). Run with: pytest"""
import numpy as np
import MyUncle as mu


def test_bistable_engine():
    """The order-repair engine has fixed points across the erosion-flux sweep."""
    eng = mu.Engine()
    stab, unstab = eng.bifurcation(np.linspace(0.02, 0.6, 200))
    assert len(stab) > 0


def test_dS_dU_identity():
    """von Mises order-entropy bridge: dS/dU = -kappa (exact)."""
    assert float(mu.dS_dUchi(3.0)) == -3.0


def test_protocol_ordering():
    """Adaptive+SIM-block extends median survival beyond MTD (the paper's core result)."""
    P = dict(mu.TISSUE_ATAVISM); P["M"] = 80
    coh = mu._init_cohort(P)
    med = {p: float(np.median(mu.simulate_protocol(p, P, coh)["surv"])) for p in mu.PROTOCOLS}
    assert med["A2A3"] > med["MTD"]
    assert med["MTD"] <= med["A3_simblock"] <= med["A2A3"]


def test_phylostratigraphy_permutation():
    """Atavism premise recovered on synthetic data with a correlation-robust permutation null."""
    eT, eN, age = mu.synthetic_expression_cohort(G=800, nT=40, nN=40)
    ana = mu.phylostratigraphy_analysis(eT, eN, age, n_perm=300)
    assert ana["effect"] > 0
    assert ana["p_perm"] < 0.05
