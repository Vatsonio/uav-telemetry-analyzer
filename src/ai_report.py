"""
Модуль AI-аналізу польоту.

Генерує текстовий звіт за допомогою LLM (Google Gemini або Groq).
Аналізує метрики, виявляє аномалії та формує рекомендації.
"""

import os
import json

import pandas as pd
import requests


def detect_anomalies(gps_df: pd.DataFrame, metrics: dict) -> list[str]:
    """
    Виявляє аномалії в польотних даних на основі евристик.

    Returns
    -------
    list[str]
        Список описів виявлених аномалій.
    """
    anomalies = []

    if gps_df.empty:
        return ["GPS-дані відсутні"]

    # Різка втрата висоти (>5 м/с вертикально)
    if "vz" in gps_df.columns:
        max_descent = gps_df["vz"].max()  # VZ positive = down in Ardupilot
        if max_descent > 5:
            anomalies.append(
                f"Різка втрата висоти: макс. вертикальна швидкість зниження {max_descent:.1f} м/с"
            )

    # Перевищення швидкості (>20 м/с горизонтально)
    if metrics.get("max_horizontal_speed", 0) > 20:
        anomalies.append(
            f"Висока горизонтальна швидкість: {metrics['max_horizontal_speed']:.1f} м/с"
        )

    # Різкі зміни швидкості (прискорення > 5 м/с²)
    if metrics.get("max_acceleration", 0) > 5:
        anomalies.append(
            f"Високе прискорення: {metrics['max_acceleration']:.1f} м/с²"
        )

    # Коротка тривалість польоту (<10 с)
    if metrics.get("total_duration", 0) < 10:
        anomalies.append(
            f"Дуже короткий політ: {metrics['total_duration']:.1f} с"
        )

    # Низька кількість супутників
    if "nsats" in gps_df.columns:
        min_sats = gps_df["nsats"].min()
        if min_sats < 6:
            anomalies.append(f"Низька кількість GPS-супутників: мін. {min_sats}")

    if not anomalies:
        anomalies.append("Аномалій не виявлено")

    return anomalies


def generate_flight_report(
    metrics: dict,
    gps_df: pd.DataFrame,
    info: dict,
    api_key: str | None = None,
    bat_df: pd.DataFrame = None,
) -> str:
    """
    Генерує текстовий AI-звіт про політ.

    Спочатку намагається використати Google Gemini API (безкоштовний тариф).
    Якщо API ключ відсутній, генерує звіт на основі шаблону.

    Parameters
    ----------
    metrics : dict
        Обчислені метрики польоту.
    gps_df : pd.DataFrame
        GPS дані.
    info : dict
        Метаінформація від парсера.
    api_key : str, optional
        API ключ для Google Gemini.

    Returns
    -------
    str
        Текстовий звіт (Markdown).
    """
    anomalies = detect_anomalies(gps_df, metrics)

    # Статистика батареї
    bat_stats = {}
    if bat_df is not None and not bat_df.empty and "voltage" in bat_df.columns:
        v = bat_df["voltage"].values
        v = v[v > 0]
        if len(v):
            bat_stats = {
                "min_voltage_V": round(float(v.min()), 2),
                "max_voltage_V": round(float(v.max()), 2),
                "mean_voltage_V": round(float(v.mean()), 2),
            }
        if "current" in bat_df.columns:
            c = bat_df["current"].values
            c = c[c >= 0]
            if len(c):
                bat_stats["max_current_A"] = round(float(c.max()), 2)
                bat_stats["mean_current_A"] = round(float(c.mean()), 2)

    # Підготовка контексту для LLM (розширений)
    context = {
        "flight_duration_s": metrics.get("total_duration", 0),
        "total_distance_m": metrics.get("total_distance", 0),
        "horizontal_speed": {
            "max_ms": metrics.get("max_horizontal_speed", 0),
            "max_kmh": round(metrics.get("max_horizontal_speed", 0) * 3.6, 1),
            "avg_ms": metrics.get("avg_horizontal_speed", 0),
            "avg_kmh": round(metrics.get("avg_horizontal_speed", 0) * 3.6, 1),
            "p50_ms": metrics.get("speed_p50", 0),
            "p75_ms": metrics.get("speed_p75", 0),
            "p95_ms": metrics.get("speed_p95", 0),
        },
        "vertical_speed_max_ms": metrics.get("max_vertical_speed", 0),
        "acceleration": {
            "max_ms2": metrics.get("max_acceleration", 0),
            "p95_ms2": metrics.get("acc_p95", 0),
        },
        "altitude": {
            "gain_m": metrics.get("max_altitude_gain", 0),
            "min_amsl_m": metrics.get("alt_min", 0),
            "max_amsl_m": metrics.get("alt_max", 0),
            "mean_amsl_m": metrics.get("alt_mean", 0),
        },
        "gps_quality": {
            "sampling_hz": info.get("gps_sampling_hz", "N/A"),
            "source": info.get("gps_source", "GPS"),
            "mean_hdop": metrics.get("hdop_mean", "N/A"),
            "min_sats": metrics.get("nsats_min", "N/A"),
            "mean_sats": metrics.get("nsats_mean", "N/A"),
        },
        "imu_sampling_hz": info.get("imu_sampling_hz", "N/A"),
        "battery": bat_stats if bat_stats else "дані відсутні",
        "anomalies": anomalies,
    }

    api_key = api_key or os.environ.get("GEMINI_API_KEY", "")

    if api_key:
        return _call_gemini(context, api_key)
    else:
        return _template_report(context)


def _call_gemini(context: dict, api_key: str) -> str:
    """Викликає Google Gemini API для генерації звіту."""
    prompt = f"""Ти - експерт з аналізу телеметрії БПЛА. Проаналізуй дані польоту та надай
структурований звіт українською мовою. Включи:
1. Загальний опис польоту (тривалість, дистанція, характер)
2. Аналіз швидкісних характеристик
3. Оцінку стабільності (на основі прискорень)
4. Виявлені аномалії та ризики
5. Рекомендації для оператора

Дані польоту:
{json.dumps(context, indent=2, ensure_ascii=False)}

Формат відповіді: Markdown з заголовками та списками."""

    url = f"https://generativelanguage.googleapis.com/v1beta/models/gemini-2.5-flash:generateContent?key={api_key}"

    import time

    for attempt in range(3):
        try:
            response = requests.post(
                url,
                json={"contents": [{"parts": [{"text": prompt}]}]},
                headers={"Content-Type": "application/json"},
                timeout=30,
            )
            if response.status_code == 429 and attempt < 2:
                time.sleep(5 * (attempt + 1))
                continue
            response.raise_for_status()
            data = response.json()
            return data["candidates"][0]["content"]["parts"][0]["text"]
        except Exception as e:
            if attempt < 2 and "429" in str(e):
                time.sleep(5 * (attempt + 1))
                continue
            return _template_report(context) + f"\n\n*AI API недоступний: {e}*"


def _template_report(context: dict) -> str:
    """Генерує звіт за шаблоном (без LLM)."""
    duration = context["flight_duration_s"]
    distance = context["total_distance_m"]
    max_h_speed = context["horizontal_speed"]["max_ms"]
    max_v_speed = context["vertical_speed_max_ms"]
    max_acc = context["acceleration"]["max_ms2"]
    alt_gain = context["altitude"]["gain_m"]
    anomalies = context["anomalies"]

    report = f"""### Звіт про політ БПЛА

#### Загальна інформація
- **Тривалість польоту:** {duration:.1f} с ({duration/60:.1f} хв)
- **Загальна дистанція:** {distance:.1f} м ({distance/1000:.2f} км)
- **Частота GPS:** {context['gps_quality']['sampling_hz']} Гц
- **Частота IMU:** {context['imu_sampling_hz']} Гц

#### Швидкісні характеристики
- **Макс. горизонтальна швидкість:** {max_h_speed:.1f} м/с ({max_h_speed * 3.6:.1f} км/год)
- **Макс. вертикальна швидкість:** {max_v_speed:.1f} м/с
- **Макс. прискорення:** {max_acc:.1f} м/с²

#### Висотний профіль
- **Макс. набір висоти:** {alt_gain:.1f} м

#### Виявлені аномалії
"""
    for a in anomalies:
        report += f"- {a}\n"

    # Оцінка польоту
    if max_acc > 10 or max_v_speed > 10:
        assessment = "Агресивний маневровий політ з високими навантаженнями."
    elif max_h_speed > 15:
        assessment = "Швидкісний політ з помірними навантаженнями."
    elif duration < 30:
        assessment = "Короткий тестовий або аварійний політ."
    else:
        assessment = "Стабільний політ з нормальними параметрами."

    report += f"\n#### Загальна оцінка\n{assessment}\n"

    return report
