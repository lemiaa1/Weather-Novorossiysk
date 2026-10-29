"""Интерфейс учебного прогноза осадков. Запуск: python -m streamlit run app.py."""
from __future__ import annotations

import hashlib
from pathlib import Path

# Сначала задаём настройки вычислений из weather_model, затем импортируем
# библиотеки интерфейса, которые также могут загружать NumPy/OpenMP.
from weather_model import CATEGORY_NAMES, forecast_next, prepare_data, train_model

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd
import streamlit as st

BASE_DIR = Path(__file__).resolve().parent
DEFAULT_CSV = BASE_DIR / "data" / "novorossiysk_rp5.csv"

st.set_page_config(page_title="Осадки в Новороссийске", page_icon="☔", layout="wide")
st.title("☔ Осадки в Новороссийске")
st.write("LightAutoML · прогноз осадков на следующий день по данным предыдущих 20 дней")
st.caption("Ачатова Амелия Павловна · ИУСбд-01-23")


@st.cache_data(show_spinner=False)
def load_data(blob: bytes, strong_mm: float):
    return prepare_data(blob, strong_mm)


@st.cache_resource(show_spinner=False, max_entries=2)
def fit_cached(blob: bytes, strong_mm: float, timeout: int):
    return train_model(prepare_data(blob, strong_mm), strong_mm, timeout)


with st.sidebar:
    st.header("Данные и обучение")
    source = st.radio("Источник CSV", ["Архив из проекта", "Загрузить CSV RP5"], key="source")
    if source == "Архив из проекта":
        if not DEFAULT_CSV.exists():
            st.error("В папке data не найден novorossiysk_rp5.csv. Распакуйте весь архив проекта.")
            st.stop()
        blob = DEFAULT_CSV.read_bytes()
    else:
        upload = st.file_uploader("CSV Новороссийска в UTF-8", type=["csv"], key="csv_upload")
        if upload is None:
            st.info("Выберите CSV RP5 с датой наблюдения и столбцом RRR.")
            st.stop()
        blob = upload.getvalue()
    strong_mm = float(st.number_input("Граница сильных осадков, мм за период", min_value=1.0,
                                      max_value=100.0, value=10.0, step=1.0, key="strong_mm"))
    st.caption("Кнопками − и + задайте, от скольких мм считать осадки сильными. Начните с 10 мм.")
    timeout = int(st.selectbox("Максимальное время обучения, секунд", [60, 180, 600], index=2, key="timeout"))
    train_clicked = st.button("Обучить LightAutoML", type="primary", use_container_width=True, key="train")
    st.caption("Нажмите стрелку и выберите время на подбор модели. Для начала — 60 с. Затем нажмите «Обучить LightAutoML».")

try:
    prepared = load_data(blob, strong_mm)
except ValueError as exc:
    st.error(str(exc))
    st.stop()

# Ключ связывает модель с конкретными данными и настройками.
model_key = hashlib.sha256(blob + f"|{strong_mm}|{timeout}".encode()).hexdigest()
if train_clicked:
    with st.spinner("LightAutoML подбирает модели. Дождитесь завершения…"):
        try:
            result = fit_cached(blob, strong_mm, timeout)
        except Exception as exc:
            st.error("Обучение не завершилось. Проверьте установку зависимостей и вывод терминала.")
            st.exception(exc)
        else:
            st.session_state["trained_result"] = result
            st.session_state["trained_key"] = model_key
            st.session_state.pop("day_forecast", None)
            st.session_state.pop("history_end", None)

result = st.session_state.get("trained_result") if st.session_state.get("trained_key") == model_key else None

def show_metrics(values):
    columns = st.columns(3)
    for i, (name, value) in enumerate(values.items()):
        columns[i % 3].metric(name, f"{value:.2%}" if name in ("Accuracy", "Balanced Accuracy", "Precision", "Recall") else f"{value:.3f}")


data_tab, score_tab, forecast_tab = st.tabs(["📊 Данные", "🎯 Проверка модели", "🌦 Прогноз дня"])
with data_tab:
    st.subheader("Архив погоды")
    c1, c2, c3 = st.columns(3)
    c1.metric("Дней в выбранном ряду", len(prepared.daily))
    c2.metric("Дней с осадками", f"{prepared.daily['rain'].mean():.2%}")
    c3.metric("Признаков модели", len(prepared.features))
    st.write(f"**Используемый период:** {prepared.daily.index.min():%d.%m.%Y} — {prepared.daily.index.max():%d.%m.%Y}")
    st.caption(f"Фактические даты исходного архива: {prepared.source_start:%d.%m.%Y} — {prepared.source_end:%d.%m.%Y}. "
               f"Полностью неизвестных дней по осадкам внутри архива: {prepared.unknown_days}. Выбран самый длинный непрерывный участок.")
    with st.expander("Как подготовлены данные и категории"):
        st.write("Из CSV читаются дата и RRR. «Осадков нет» = 0, «Следы осадков» = небольшие осадки. Пропуски не заменяются сухой погодой.")
        st.write(f"Категории: 0 — нет; 1 — следы или больше 0, но меньше {strong_mm:g} мм; 2 — не меньше {strong_mm:g} мм за период RRR.")
        st.write("Категория дня — максимум среди известных периодов, заканчивающихся в эту дату. RRR обычно относится к 12 часам, поэтому это приближённый признак дня, а не сумма за календарные сутки. Одна известная запись не гарантирует полного покрытия суток.")
        st.write("В модель входят день, месяц, год, 20 прошлых бинарных значений осадков и 20 прошлых категорий. Таргет — наличие осадков в прогнозируемый день. Его категория в признаки не передаётся.")
    split = pd.DataFrame([
        ["Обучение", prepared.train.index.min().date(), prepared.train.index.max().date(), len(prepared.train)],
        ["Проверка и выбор порога", prepared.valid.index.min().date(), prepared.valid.index.max().date(), len(prepared.valid)],
        ["Итоговый тест", prepared.test.index.min().date(), prepared.test.index.max().date(), len(prepared.test)],
    ], columns=["Часть", "Начало", "Конец", "Дней"])
    st.dataframe(split, hide_index=True, use_container_width=True)
    st.caption("Первые 20 дней нужны для истории первых обучающих примеров. Проверочная часть выбирает настройки и порог; итоговый тест не передаётся в fit_predict.")
    # Сезонный график помогает увидеть различия между месяцами, а не только общую долю.
    months = prepared.daily.groupby(prepared.daily.index.month)["rain"].mean().reindex(range(1, 13))
    fig, ax = plt.subplots(figsize=(9, 2.8))
    ax.bar(range(1, 13), months.values * 100, color="#277da8", width=0.65)
    ax.set_xticks(range(1, 13))
    ax.set_xlabel("Месяц"); ax.set_ylabel("Дней с осадками, %")
    ax.set_ylim(0, 100); ax.spines[["top", "right"]].set_visible(False)
    fig.tight_layout(); st.pyplot(fig); plt.close(fig)
    counts = prepared.daily["precipitation_level"].map(CATEGORY_NAMES).value_counts().reindex(CATEGORY_NAMES.values(), fill_value=0)
    st.dataframe(counts.rename("Дней").to_frame(), use_container_width=True)
    with st.expander("Посмотреть подготовленную таблицу"):
        st.dataframe(prepared.daily.tail(60), use_container_width=True)

with score_tab:
    st.subheader("Проверка на будущих датах")
    if result is None:
        st.info("Нажмите «Обучить LightAutoML» в левой панели. После изменения CSV или границы категорий модель нужно обучить с новыми настройками.")
    else:
        st.success(f"Обучение завершено за {result.training_seconds:.1f} с. Порог ответа: {result.threshold:.2f}.")
        show_metrics(result.metrics)
        c = result.confusion
        st.write(f"Верных прогнозов: **{int(c.trace())} из {int(c.sum())}**. Обнаружено **{int(c[1, 1])} из {int(c[1].sum())}** дней с осадками.")
        matrix = pd.DataFrame(c, index=["Факт: без осадков", "Факт: осадки"], columns=["Прогноз: без осадков", "Прогноз: осадки"])
        st.dataframe(matrix, use_container_width=True)
        st.caption(f"Ложных тревог — {int(c[0, 1])}; пропущенных дней с осадками — {int(c[1, 0])}. Строки — фактическая погода, столбцы — ответ модели.")
        scores = pd.DataFrame({
            "Модель / правило": ["LightAutoML: выбранный порог", "LightAutoML: порог 0.5", "Постоянный прогноз класса большинства"],
            "Accuracy, %": [result.metrics["Accuracy"] * 100, result.fixed_threshold_metrics["Accuracy"] * 100, result.baseline["Accuracy"] * 100],
            "Balanced Accuracy, %": [result.metrics["Balanced Accuracy"] * 100, result.fixed_threshold_metrics["Balanced Accuracy"] * 100, result.baseline["Balanced Accuracy"] * 100],
        })
        st.dataframe(scores.round(2), hide_index=True, use_container_width=True)
        if result.metrics["Accuracy"] <= result.baseline["Accuracy"]:
            st.warning("По обычной accuracy модель не превзошла постоянный прогноз. Оцените также обнаружение осадков и ложные тревоги.")
        else:
            st.info("На выбранном тесте accuracy выше, чем у постоянного прогноза. Это не гарантирует такой же результат в другом сезоне.")
        with st.expander("Как выбирается порог"):
            st.write("Проверяются пороги 0.00–1.00 с шагом 0.01. Выбирается максимум Balanced Accuracy на проверочной части: среднее полноты для дождливых и сухих дней. При равенстве выбирается порог ближе к 0.5.")
            st.dataframe(result.threshold_table.head(8).round(3), hide_index=True, use_container_width=True)
            st.write("ROC AUC оценивает порядок вероятностей, а не долю верных ответов. Порог по тестовой accuracy не выбирается.")
        chart = pd.DataFrame({"Фактические осадки": result.prepared.test["rain"], "Оценка вероятности": result.test_probability})
        st.line_chart(chart)
        st.caption("В ДЗ2 на этом тесте ARIMA дала 81.52%, LSTM — 70.65%. Это результаты прошлого запуска; здесь они не переобучаются. При загрузке другого CSV сравнение с этими числами неприменимо.")

with forecast_tab:
    st.subheader("Прогноз на день после выбранной истории")
    st.write("Это работа с историческим архивом, а не актуальная сводка погоды. Модель получает только сведения до выбранной даты включительно.")
    if result is None:
        st.info("Сначала обучите модель в левой панели.")
    else:
        end = st.date_input("История известна по дату включительно", value=prepared.daily.index.max().date(),
                            min_value=prepared.valid.index.max().date(), max_value=prepared.daily.index.max().date(),
                            key="history_end")
        if st.button("Получить прогноз на следующий день", key="forecast"):
            try:
                date, probability, label = forecast_next(result, pd.Timestamp(end))
                st.session_state["day_forecast"] = (model_key, str(end), date, probability, label)
            except Exception as exc:
                st.error(f"Прогноз не получен: {exc}")
        saved = st.session_state.get("day_forecast")
        if saved and saved[0] == model_key and saved[1] == str(end):
            _, _, date, probability, label = saved
            st.metric(f"Прогноз на {date:%d.%m.%Y}", "Осадки ожидаются" if label else "Осадки не ожидаются")
            st.write(f"Оценка вероятности осадков: **{probability:.1%}**, порог ответа: **{result.threshold:.2f}**.")
            fig, ax = plt.subplots(figsize=(7, 2.3))
            ax.barh(["Без осадков", "Осадки"], [1 - probability, probability], color=["#f2b461", "#277da8"])
            ax.axvline(result.threshold, color="#666666", linestyle="--", label="Порог для осадков")
            ax.set_xlim(0, 1); ax.set_xlabel("Оценка вероятности"); ax.legend(loc="lower right")
            fig.tight_layout(); st.pyplot(fig); plt.close(fig)
            if date in prepared.daily.index:
                actual = int(prepared.daily.loc[date, "rain"])
                st.write("Архивный факт: **" + ("осадки были" if actual else "осадков не было") + "**.")
            else:
                st.caption("За эту дату в выбранном непрерывном ряду нет факта для сравнения.")
            st.caption("Прогноз бинарный: категории «небольшие/сильные» используются как прошлые признаки; модель не прогнозирует силу осадков.")


