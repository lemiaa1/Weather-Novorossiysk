"""Полный сценарий интерфейса с реальным обучением, без подмены LightAutoML."""
import sys
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from streamlit.testing.v1 import AppTest


class AppTests(unittest.TestCase):
    def test_train_predict_and_invalidate(self):
        app = AppTest.from_file(str(ROOT / "app.py"), default_timeout=180).run()
        self.assertFalse(list(app.exception))
        self.assertEqual(len(app.tabs), 3)
        app.selectbox(key="timeout").set_value(60).run()
        app.button(key="train").click().run(timeout=180)
        self.assertFalse(list(app.exception))
        self.assertFalse(list(app.error))
        self.assertTrue(list(app.success))
        app.button(key="forecast").click().run()
        self.assertFalse(list(app.exception))
        self.assertTrue(any("Прогноз на 20.07.2022" in item.label for item in app.metric))
        # Смена границы категорий не должна оставлять старый прогноз активным.
        app.number_input(key="strong_mm").set_value(12.0).run()
        self.assertFalse(list(app.exception))
        self.assertFalse(any("Прогноз на 20.07.2022" in item.label for item in app.metric))
        # Пустая загрузка обрабатывается подсказкой, а не трассировкой.
        app.radio(key="source").set_value("Загрузить CSV RP5").run()
        self.assertFalse(list(app.exception))
        self.assertTrue(any("Выберите CSV" in item.value for item in app.info))


if __name__ == "__main__":
    unittest.main()
