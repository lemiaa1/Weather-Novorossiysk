"""Проверки реального архива, временного разделения и отсутствия текущего таргета в признаках."""
import sys
import unittest
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from weather_model import choose_threshold, make_features, prepare_data


class WeatherTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.blob = (ROOT / "data" / "novorossiysk_rp5.csv").read_bytes()
        cls.data = prepare_data(cls.blob)

    def test_real_archive_and_split(self):
        p = self.data
        self.assertEqual((len(p.daily), len(p.train), len(p.valid), len(p.test)), (1826, 1540, 174, 92))
        self.assertEqual(len(p.features), 43)
        self.assertEqual(str(p.test.index.min().date()), "2022-04-19")
        self.assertEqual(str(p.test.index.max().date()), "2022-07-19")
        self.assertLess(p.train.index.max(), p.valid.index.min())
        self.assertLess(p.valid.index.max(), p.test.index.min())
        self.assertNotIn("rain", p.features)
        self.assertEqual(p.daily["precipitation_level"].value_counts().to_dict(), {0: 1234, 1: 470, 2: 122})

    def test_target_day_does_not_change_own_features(self):
        p = self.data
        original, features, _ = make_features(p.daily)
        changed = p.daily.copy()
        day = p.test.index[0]
        changed.loc[day, "rain"] = 1 - changed.loc[day, "rain"]
        changed.loc[day, "precipitation_level"] = 2 if changed.loc[day, "rain"] else 0
        new, _, _ = make_features(changed)
        pd.testing.assert_series_equal(original.loc[day, features], new.loc[day, features])

    def test_lags_match_actual_history(self):
        p = self.data
        day = p.test.index[0]
        for lag in range(1, 21):
            self.assertEqual(p.test.loc[day, f"rain_lag_{lag}"], p.daily.loc[day - pd.Timedelta(days=lag), "rain"])
            self.assertEqual(p.test.loc[day, f"precipitation_level_lag_{lag}"], p.daily.loc[day - pd.Timedelta(days=lag), "precipitation_level"])

    def test_threshold_uses_both_classes(self):
        threshold, table = choose_threshold(np.array([0, 0, 1, 1]), np.array([.1, .2, .8, .9]))
        self.assertAlmostEqual(threshold, .5)
        self.assertAlmostEqual(table.iloc[0]["balanced_accuracy"], 1.)

    def test_bad_files_are_reported(self):
        for blob in (b"", b"date;rain\n2020-01-01;0\n"):
            with self.assertRaises(ValueError):
                prepare_data(blob)
        with self.assertRaises(ValueError):
            prepare_data(self.blob, strong_mm=-1)


if __name__ == "__main__":
    unittest.main()
