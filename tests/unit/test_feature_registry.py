from nexusfeed.features.feature_registry import FeatureRegistry


def test_register_and_get_version():
    registry = FeatureRegistry()
    record = registry.register("v2", "adds dwell-time feature", ["dwell_ms"])
    assert record.version == "v2"
    assert registry.get("v2") is record


def test_get_missing_version_returns_none():
    registry = FeatureRegistry()
    assert registry.get("nonexistent") is None


def test_current_returns_settings_default():
    registry = FeatureRegistry()
    assert registry.current() == registry.settings.current_feature_version


def test_list_versions_sorted_by_computed_at():
    registry = FeatureRegistry()
    registry.register("v1", "first", [])
    registry.register("v2", "second", [])
    versions = registry.list_versions()
    assert [v.version for v in versions] == ["v1", "v2"]
