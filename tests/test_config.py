"""Validasi konfigurasi — memastikan FEATURE_COLS konsisten."""

from src.config import (
    FEATURE_COLS,
    FEATURE_INDEX_MAP,
    HOLIDAY_INTENSITY_FEATURES,
    LAG_ROLL_FEATURES,
    SEASONALITY_FEATURES,
    TEMPORAL_PEAK_FEATURES,
)


class TestFeatureConfig:
    """Test integritas konfigurasi fitur."""

    def test_feature_count_is_52(self):
        """FEATURE_COLS harus persis 52 fitur."""
        assert len(FEATURE_COLS) == 52, f"Expected 52 features, got {len(FEATURE_COLS)}"

    def test_no_duplicate_features(self):
        """Tidak ada nama fitur yang duplikat."""
        assert len(FEATURE_COLS) == len(set(FEATURE_COLS)), "Duplikat ditemukan di FEATURE_COLS!"

    def test_feature_groups_sum_to_52(self):
        """Jumlah dari semua sub-group harus 52."""
        total = (
            len(SEASONALITY_FEATURES)
            + len(HOLIDAY_INTENSITY_FEATURES)
            + len(TEMPORAL_PEAK_FEATURES)
            + len(LAG_ROLL_FEATURES)
        )
        assert total == 52, f"Feature groups total {total}, expected 52"

    def test_lag_roll_has_33_features(self):
        """LAG_ROLL_FEATURES harus persis 33 fitur."""
        assert len(LAG_ROLL_FEATURES) == 33, f"Expected 33 lag/roll features, got {len(LAG_ROLL_FEATURES)}"

    def test_seasonality_has_14_features(self):
        """SEASONALITY_FEATURES harus persis 14 fitur."""
        assert len(SEASONALITY_FEATURES) == 14, f"Expected 14 seasonality features, got {len(SEASONALITY_FEATURES)}"

    def test_holiday_intensity_has_4_features(self):
        """HOLIDAY_INTENSITY_FEATURES harus persis 4 fitur."""
        assert len(HOLIDAY_INTENSITY_FEATURES) == 4, f"Expected 4 holiday intensity features, got {len(HOLIDAY_INTENSITY_FEATURES)}"

    def test_temporal_peak_has_1_feature(self):
        """TEMPORAL_PEAK_FEATURES harus persis 1 fitur."""
        assert len(TEMPORAL_PEAK_FEATURES) == 1, f"Expected 1 temporal peak feature, got {len(TEMPORAL_PEAK_FEATURES)}"

    def test_feature_index_map_matches(self):
        """FEATURE_INDEX_MAP harus konsisten dengan FEATURE_COLS."""
        assert len(FEATURE_INDEX_MAP) == len(FEATURE_COLS)
        for i, name in enumerate(FEATURE_COLS):
            assert FEATURE_INDEX_MAP[i] == name, f"Mismatch at index {i}: expected {name}, got {FEATURE_INDEX_MAP[i]}"

    def test_is_peak_day_in_features(self):
        """is_peak_day harus ada di FEATURE_COLS."""
        assert "is_peak_day" in FEATURE_COLS

    def test_demand_lag_1_in_features(self):
        """demand_lag_1 harus ada di FEATURE_COLS."""
        assert "demand_lag_1" in FEATURE_COLS

    def test_holiday_intensity_in_features(self):
        """holiday_intensity harus ada di FEATURE_COLS."""
        assert "holiday_intensity" in FEATURE_COLS
