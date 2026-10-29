"""Обработка RP5 и модель из ДЗ3. Интерфейс Streamlit находится в app.py."""
from __future__ import annotations

import io
import os
import time
from dataclasses import dataclass
from typing import Any

os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("OMP_THREAD_LIMIT", "1")

import numpy as np
import pandas as pd
from sklearn.metrics import (
    accuracy_score, balanced_accuracy_score, confusion_matrix,
    f1_score, precision_score, recall_score, roc_auc_score,
)

DATE_COLUMN = "Местное время в Новороссийске"
TARGET = "rain"
HISTORY_DAYS = 20
CATEGORY_NAMES = {0: "Нет осадков", 1: "Небольшие осадки", 2: "Сильные осадки"}


@dataclass
class PreparedData:
    daily: pd.DataFrame
    train: pd.DataFrame
    valid: pd.DataFrame
    test: pd.DataFrame
    features: list[str]
    categories: list[str]
    source_start: pd.Timestamp
    source_end: pd.Timestamp
    unknown_days: int


@dataclass
class ModelResult:
    model: Any
    prepared: PreparedData
    threshold: float
    threshold_table: pd.DataFrame
    test_probability: np.ndarray
    metrics: dict[str, float]
    confusion: np.ndarray
    baseline: dict[str, float]
    fixed_threshold_metrics: dict[str, float]
    training_seconds: float
    strong_mm: float


def prepare_data(csv_bytes: bytes, strong_mm: float = 10.0) -> PreparedData:
    """Не заполняет отсутствующие осадки нулями. Сохраняет разделение ДЗ3."""
    if not np.isfinite(strong_mm) or strong_mm <= 0:
        raise ValueError("Граница сильных осадков должна быть положительным числом.")
    if not csv_bytes:
        raise ValueError("CSV пуст. Выберите файл RP5.")
    # В оригинальном архиве перед заголовком находятся строки комментариев #.
    # Принимается также тот же CSV без этих строк.
    try:
        lines = csv_bytes.decode("utf-8-sig").splitlines()
    except UnicodeDecodeError as exc:
        raise ValueError("Нужен CSV RP5 в кодировке UTF-8.") from exc
    header = next((i for i, line in enumerate(lines) if DATE_COLUMN in line and "RRR" in line), None)
    if header is None:
        raise ValueError("Не найдены столбцы «Местное время в Новороссийске» и RRR. Используйте архив RP5 Новороссийска.")
    try:
        raw = pd.read_csv(io.BytesIO(csv_bytes), sep=";", skiprows=header,
                          encoding="utf-8-sig", index_col=False,
                          usecols=[DATE_COLUMN, "RRR"])
    except (ValueError, pd.errors.ParserError) as exc:
        raise ValueError("Не удалось прочитать таблицу. Проверьте CSV с разделителем «;».") from exc
    dates = pd.to_datetime(raw[DATE_COLUMN], format="%d.%m.%Y %H:%M", errors="coerce")
    if dates.isna().any():
        raise ValueError("В архиве есть пустые или неверные даты наблюдений.")
    text = raw["RRR"].fillna("").astype(str).str.strip()
    amount = pd.to_numeric(text.str.replace(",", ".", regex=False), errors="coerce")
    level = pd.Series(np.nan, index=raw.index)
    level.loc[text.eq("Осадков нет") | amount.eq(0)] = 0
    level.loc[text.eq("Следы осадков") | (amount.gt(0) & amount.lt(strong_mm))] = 1
    level.loc[amount.ge(strong_mm)] = 2
    bad = text.ne("") & level.isna()
    if bad.any():
        raise ValueError("Неизвестные значения RRR: " + ", ".join(text.loc[bad].unique()[:5]))
    frame = pd.DataFrame({"date": dates.dt.normalize(), "precipitation_level": level})
    frame[TARGET] = (level > 0).astype(float).where(level.notna())
    daily = frame.groupby("date")[[TARGET, "precipitation_level"]].max().sort_index().asfreq("D")
    if daily.empty:
        raise ValueError("В архиве нет наблюдений.")
    source_start, source_end = daily.index.min(), daily.index.max()
    known = daily[TARGET].notna()
    unknown_days = int((~known).sum())
    if not known.any():
        raise ValueError("В архиве нет известных значений осадков.")
    groups = (~known).cumsum()
    run_sizes = daily.loc[known].groupby(groups.loc[known]).size()
    daily = daily.loc[known & groups.eq(run_sizes.idxmax())].copy().astype(int)
    if len(daily) < 200:
        raise ValueError("Нужно не менее 200 последовательных дней с известными осадками.")
    if not (daily.index.to_series().diff().dropna() == pd.Timedelta(days=1)).all():
        raise ValueError("Выбранный участок должен быть непрерывным.")
    table, feature_names, category_columns = make_features(daily)
    train_end = int(len(daily) * 0.95)
    valid_start = int(train_end * 0.9)
    valid_date, test_date = daily.index[valid_start], daily.index[train_end]
    train = table.loc[table.index < valid_date].copy()
    valid = table.loc[(table.index >= valid_date) & (table.index < test_date)].copy()
    test = table.loc[table.index >= test_date].copy()
    for name, part in (("обучение", train), ("проверка", valid), ("тест", test)):
        if part.empty or part[TARGET].nunique() != 2:
            raise ValueError(f"В части «{name}» нужны оба класса: дни с осадками и без них.")
    if not test.index.equals(daily.index[train_end:]):
        raise ValueError("Даты теста не совпали с исходным рядом.")
    return PreparedData(daily, train, valid, test, feature_names, category_columns,
                        source_start, source_end, unknown_days)


def make_features(daily: pd.DataFrame) -> tuple[pd.DataFrame, list[str], list[str]]:
    """Каждая строка — прогнозируемый день; все погодные признаки строго из прошлого."""
    table = pd.DataFrame(index=daily.index)
    table["day"], table["month"], table["year"] = daily.index.day, daily.index.month, daily.index.year
    for lag in range(1, HISTORY_DAYS + 1):
        table[f"rain_lag_{lag}"] = daily[TARGET].shift(lag)
    categories = []
    for lag in range(1, HISTORY_DAYS + 1):
        column = f"precipitation_level_lag_{lag}"
        table[column] = daily["precipitation_level"].shift(lag)
        categories.append(column)
    table[TARGET] = daily[TARGET]
    table = table.dropna()
    features = [name for name in table if name != TARGET]
    return table, features, categories


def choose_threshold(y: np.ndarray, probability: np.ndarray) -> tuple[float, pd.DataFrame]:
    """Выбор по Balanced Accuracy только на проверочной части."""
    y, probability = np.asarray(y), np.asarray(probability).ravel()
    if len(y) != len(probability) or not np.isfinite(probability).all() or set(np.unique(y)) != {0, 1}:
        raise ValueError("Для подбора порога нужны конечные вероятности и оба класса.")
    rows = []
    for threshold in np.linspace(0, 1, 101):
        labels = (probability > threshold).astype(int)
        rain_recall = float((labels[y == 1] == 1).mean())
        dry_recall = float((labels[y == 0] == 0).mean())
        rows.append({"threshold": threshold, "balanced_accuracy": (rain_recall + dry_recall) / 2,
                     "rain_recall": rain_recall, "dry_recall": dry_recall,
                     "distance_to_05": abs(threshold - 0.5)})
    results = pd.DataFrame(rows).sort_values(["balanced_accuracy", "distance_to_05"], ascending=[False, True])
    return float(results.iloc[0]["threshold"]), results.drop(columns="distance_to_05")


def measure(y: np.ndarray, labels: np.ndarray, probability: np.ndarray | None = None) -> dict[str, float]:
    values = {"Accuracy": accuracy_score(y, labels),
              "Balanced Accuracy": balanced_accuracy_score(y, labels),
              "Precision": precision_score(y, labels, zero_division=0),
              "Recall": recall_score(y, labels, zero_division=0),
              "F1": f1_score(y, labels, zero_division=0)}
    if probability is not None:
        values["ROC AUC"] = roc_auc_score(y, probability)
    return {key: float(value) for key, value in values.items()}


def train_model(data: PreparedData, strong_mm: float = 10.0, timeout: int = 600) -> ModelResult:
    # Импорты тяжёлой библиотеки выполняются при явном нажатии кнопки обучения.
    from lightautoml.automl.presets.tabular_presets import TabularAutoML
    from lightautoml.tasks import Task
    import torch

    np.random.seed(42)
    threads = 1
    torch.set_num_threads(threads)
    start = time.perf_counter()
    model = TabularAutoML(
        task=Task("binary", loss="logloss", metric="auc"),
        timeout=int(timeout), cpu_limit=threads, gpu_ids=None,
        general_params={"use_algos": [["linear_l2", "lgb", "lgb_tuned"]]},
        reader_params={"n_jobs": 1, "random_state": 42},
    )
    validation = model.fit_predict(data.train.reset_index(drop=True),
                                   roles={"target": TARGET, "category": data.categories},
                                   valid_data=data.valid.reset_index(drop=True), verbose=1)
    threshold, threshold_table = choose_threshold(data.valid[TARGET].to_numpy(), validation.data[:, 0])
    probability = model.predict(data.test[data.features].reset_index(drop=True)).data[:, 0]
    if len(probability) != len(data.test) or not np.isfinite(probability).all():
        raise ValueError("Модель вернула некорректные прогнозы.")
    y = data.test[TARGET].to_numpy()
    labels = (probability > threshold).astype(int)
    majority = int(data.train[TARGET].mode().iloc[0])
    return ModelResult(model, data, threshold, threshold_table, probability,
                       measure(y, labels, probability), confusion_matrix(y, labels, labels=[0, 1]),
                       measure(y, np.full(len(y), majority)), measure(y, (probability > 0.5).astype(int), probability),
                       time.perf_counter() - start, strong_mm)


def forecast_next(result: ModelResult, history_end: pd.Timestamp) -> tuple[pd.Timestamp, float, int]:
    """Фиксированная обученная модель; история доступна только до history_end."""
    data = result.prepared
    end = pd.Timestamp(history_end).normalize()
    # Прогноз для проверки допускается только после конца проверочного периода.
    if end < data.valid.index.max() or end > data.daily.index.max():
        raise ValueError("Выберите конец истории в доступном тестовом периоде.")
    history = data.daily.loc[:end].tail(HISTORY_DAYS)
    if len(history) != HISTORY_DAYS:
        raise ValueError("Для прогноза нужно 20 известных предыдущих дней.")
    target_date = end + pd.Timedelta(days=1)
    row = {"day": target_date.day, "month": target_date.month, "year": target_date.year}
    for lag in range(1, HISTORY_DAYS + 1):
        row[f"rain_lag_{lag}"] = float(history[TARGET].iloc[-lag])
        row[f"precipitation_level_lag_{lag}"] = float(history["precipitation_level"].iloc[-lag])
    frame = pd.DataFrame([row], columns=data.features)
    probability = float(result.model.predict(frame).data[0, 0])
    if not np.isfinite(probability):
        raise ValueError("Не удалось вычислить вероятность.")
    return target_date, probability, int(probability > result.threshold)
