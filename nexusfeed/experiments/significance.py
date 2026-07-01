"""Statistical significance testing: two-proportion z-test, Mann-Whitney U,
sample ratio mismatch check, and Bonferroni correction for multiple metrics.

Encodes the "five checks before calling a result significant" from the
blueprint's interview prep section.
"""
from __future__ import annotations

import math

from scipy import stats

from nexusfeed.types import ExperimentMetrics, SignificanceResult


def two_proportion_z_test(
    control_conversions: int, control_total: int, treatment_conversions: int, treatment_total: int
) -> tuple[float, float]:
    """Returns (z_statistic, p_value) for a two-sided test of equal proportions."""
    if control_total == 0 or treatment_total == 0:
        return 0.0, 1.0
    p1 = control_conversions / control_total
    p2 = treatment_conversions / treatment_total
    p_pool = (control_conversions + treatment_conversions) / (control_total + treatment_total)
    se = math.sqrt(p_pool * (1 - p_pool) * (1 / control_total + 1 / treatment_total))
    if se == 0:
        return 0.0, 1.0
    z = (p2 - p1) / se
    p_value = 2 * (1 - stats.norm.cdf(abs(z)))
    return z, p_value


def mann_whitney_u_test(control_samples: list[float], treatment_samples: list[float]) -> tuple[float, float]:
    if not control_samples or not treatment_samples:
        return 0.0, 1.0
    statistic, p_value = stats.mannwhitneyu(control_samples, treatment_samples, alternative="two-sided")
    return float(statistic), float(p_value)


def sample_ratio_mismatch(
    control_n: int, treatment_n: int, expected_ratio: float = 1.0, tolerance: float = 0.02
) -> bool:
    """Returns True if the observed split matches the expected ratio within
    tolerance (i.e. SRM is NOT present — result is trustworthy on this axis).
    Uses a chi-square goodness-of-fit test against the expected split.
    """
    total = control_n + treatment_n
    if total == 0:
        return True
    expected_control = total * (expected_ratio / (1 + expected_ratio))
    expected_treatment = total - expected_control
    chi2, p_value = stats.chisquare(
        [control_n, treatment_n], f_exp=[expected_control, expected_treatment]
    )
    return bool(p_value > 0.001)  # SRM check uses a strict threshold per industry convention


def bonferroni_correction(alpha: float, num_metrics: int) -> float:
    return alpha / num_metrics if num_metrics > 0 else alpha


def required_sample_size(
    baseline_rate: float, minimum_detectable_effect: float, power: float = 0.8, alpha: float = 0.05
) -> int:
    """Approximate per-arm sample size for a two-proportion test, using the
    normal approximation — good enough for pre-registration sanity checks.
    """
    z_alpha = stats.norm.ppf(1 - alpha / 2)
    z_beta = stats.norm.ppf(power)
    p1 = baseline_rate
    p2 = baseline_rate * (1 + minimum_detectable_effect)
    p_bar = (p1 + p2) / 2
    numerator = (
        z_alpha * math.sqrt(2 * p_bar * (1 - p_bar)) + z_beta * math.sqrt(p1 * (1 - p1) + p2 * (1 - p2))
    ) ** 2
    denominator = (p2 - p1) ** 2
    return math.ceil(numerator / denominator) if denominator > 0 else 0


def evaluate_experiment(
    control: ExperimentMetrics,
    treatment: ExperimentMetrics,
    expected_ratio: float = 1.0,
    num_metrics_tested: int = 1,
) -> SignificanceResult:
    z, p_value = two_proportion_z_test(
        control.clicks, control.impressions, treatment.clicks, treatment.impressions
    )
    srm_ok = sample_ratio_mismatch(control.impressions, treatment.impressions, expected_ratio)
    alpha = bonferroni_correction(0.05, num_metrics_tested)

    relative_lift = (treatment.ctr - control.ctr) / control.ctr if control.ctr > 0 else 0.0

    return SignificanceResult(
        metric="ctr",
        control_value=control.ctr,
        treatment_value=treatment.ctr,
        relative_lift=relative_lift,
        p_value=p_value,
        is_significant=(p_value < alpha) and srm_ok,
        sample_ratio_ok=srm_ok,
    )
