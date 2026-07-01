from nexusfeed.experiments.significance import (
    bonferroni_correction,
    evaluate_experiment,
    required_sample_size,
    sample_ratio_mismatch,
    two_proportion_z_test,
)
from nexusfeed.types import ExperimentMetrics


def test_two_proportion_z_test_detects_lift():
    z, p_value = two_proportion_z_test(
        control_conversions=500, control_total=10000, treatment_conversions=650, treatment_total=10000
    )
    assert p_value < 0.05
    assert z > 0  # z = (p_treatment - p_control) / se; treatment has higher conversion here


def test_two_proportion_z_test_no_difference():
    z, p_value = two_proportion_z_test(
        control_conversions=500, control_total=10000, treatment_conversions=505, treatment_total=10000
    )
    assert p_value > 0.05


def test_sample_ratio_mismatch_flags_bad_split():
    assert sample_ratio_mismatch(45000, 55000, expected_ratio=1.0) == False  # noqa: E712 - SRM present
    assert sample_ratio_mismatch(50100, 49900, expected_ratio=1.0) == True  # noqa: E712 - fine


def test_bonferroni_correction():
    assert bonferroni_correction(0.05, 5) == 0.01


def test_required_sample_size_positive():
    n = required_sample_size(baseline_rate=0.1, minimum_detectable_effect=0.02, power=0.8)
    assert n > 0


def test_evaluate_experiment_significant_lift():
    control = ExperimentMetrics(experiment_name="e", variant="control", impressions=10000, clicks=500)
    treatment = ExperimentMetrics(experiment_name="e", variant="treatment", impressions=10000, clicks=650)
    result = evaluate_experiment(control, treatment)
    assert result.relative_lift > 0
    assert result.is_significant is True
