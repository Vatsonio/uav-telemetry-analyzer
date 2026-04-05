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
    phases: list[dict] | None = None,
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
    import numpy as np
    from .coordinates import haversine

    context = {
        "flight_duration_s": metrics.get("total_duration", 0),
        "flight_duration_min": round(metrics.get("total_duration", 0) / 60, 1),
        "total_distance_m": metrics.get("total_distance", 0),
        "total_distance_km": round(metrics.get("total_distance", 0) / 1000, 2),
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
        "imu_drift_max_speed_ms": metrics.get("max_imu_speed", "N/A"),
        "flight_phases": phases if phases else "не визначено",
    }

    # Додаткова GPS-статистика
    if not gps_df.empty:
        alt = gps_df["alt"].values
        speed = gps_df["speed"].values
        context["altitude_std_m"] = round(float(alt.std()), 2)
        context["speed_std_ms"] = round(float(speed.std()), 2)

        # Відстань старт-фініш (пряма) та чи повернувся
        start_end = haversine(
            gps_df["lat"].iloc[0], gps_df["lng"].iloc[0],
            gps_df["lat"].iloc[-1], gps_df["lng"].iloc[-1],
        )
        context["start_to_end_m"] = round(start_end, 1)
        context["returned_to_start"] = bool(start_end < 50)  # <50 м = повернувся

        # Швидкість: перша/друга половина польоту (тренд)
        mid = len(speed) // 2
        if mid > 0:
            context["speed_first_half_avg_ms"] = round(float(speed[:mid].mean()), 1)
            context["speed_second_half_avg_ms"] = round(float(speed[mid:].mean()), 1)

        # Висота: тренд початок vs кінець
        n10 = max(len(alt) // 10, 1)
        context["altitude_first_10pct_avg"] = round(float(alt[:n10].mean()), 1)
        context["altitude_last_10pct_avg"] = round(float(alt[-n10:].mean()), 1)

        # Час у русі vs зависання
        moving = speed > 1.0
        context["time_moving_pct"] = round(float(moving.sum() / len(moving) * 100), 1)
        context["time_hovering_pct"] = round(100 - context["time_moving_pct"], 1)

        # Макс. зміна висоти між сусідніми точками
        if len(alt) > 1:
            alt_diffs = np.abs(np.diff(alt))
            context["max_alt_jump_m"] = round(float(alt_diffs.max()), 2)

    # Батарея: тренд
    if bat_stats and bat_df is not None and not bat_df.empty:
        v = bat_df["voltage"].values
        v = v[v > 0]
        if len(v) > 10:
            n10 = max(len(v) // 10, 1)
            context["battery_trend"] = {
                "start_voltage": round(float(v[:n10].mean()), 2),
                "end_voltage": round(float(v[-n10:].mean()), 2),
                "drop_V": round(float(v[:n10].mean() - v[-n10:].mean()), 2),
            }

    api_key = api_key or os.environ.get("GEMINI_API_KEY", "")

    if api_key:
        return _call_gemini(context, api_key)
    else:
        return _template_report(context)


def _risk_level(context: dict) -> str:
    """Визначає рівень ризику польоту за метриками."""
    acc_max = context.get("acceleration", {}).get("max_ms2", 0)
    max_v = context.get("vertical_speed_max_ms", 0)
    max_h = context.get("horizontal_speed", {}).get("max_ms", 0)
    hdop = context.get("gps_quality", {}).get("mean_hdop", 1)
    nsats_min = context.get("gps_quality", {}).get("min_sats", 10)
    bat = context.get("battery", {})
    min_v = bat.get("min_voltage_V", 99) if isinstance(bat, dict) else 99

    score = 0
    if acc_max > 8:
        score += 3
    elif acc_max > 5:
        score += 1
    if max_v > 8:
        score += 3
    elif max_v > 5:
        score += 1
    if max_h > 25:
        score += 2
    elif max_h > 20:
        score += 1
    if isinstance(hdop, (int, float)) and hdop > 3:
        score += 2
    if isinstance(nsats_min, (int, float)) and nsats_min < 6:
        score += 2
    if min_v < 10.5:
        score += 3
    elif min_v < 11.0:
        score += 1

    if score >= 5:
        return "CRITICAL"
    if score >= 3:
        return "WARNING"
    return "NOMINAL"


def _call_gemini(context: dict, api_key: str) -> str:
    """Викликає Google Gemini API для генерації звіту."""
    risk = _risk_level(context)
    prompt = f"""Ти - експерт-діагност авіаційної телеметрії ArduPilot.
Твоє завдання - аналізувати польотні логи, знаходити аномалії та визначати
їхні першопричини за методом "5 Чому" (від симптому до проблеми, яку можна виправити).
Пиши УКРАЇНСЬКОЮ.

ПРАВИЛА:
1. Пояснюй ПРИЧИННО-НАСЛІДКОВІ зв'язки, а не просто перераховуй відхилення.
2. Оперуй точними назвами параметрів ArduPilot (GPS, IMU/ACC/GYR, BAT, AHR2, RCOUT, VibeZ, XKF3 тощо) де це доречно.
3. Чітко розрізняй механічні відмови (раптова розбіжність заданого DesRoll/DesPitch та фактичного Roll/Pitch) від проблем налаштування ПІД-регуляторів.
4. Оцінюй загальний рівень ризику: NOMINAL / WARNING / CRITICAL (попередня автооцінка: {risk}).
5. Конкретні числа з даних, порівняння з нормами:
   - Горизонт. швидкість мультикоптера: типово до 15-20 м/с
   - Вертик. швидкість: до 3-5 м/с
   - Прискорення: до 3-5 м/с², >8 - агресивне/аномальне
   - HDOP < 2 = добре, > 3 = ненадійно
   - Мін. супутників: ≥ 8 добре, < 6 критично
   - LiPo: мін. напруга > 3.5 В/cel, < 3.3 В/cel = пошкодження
6. Не вигадуй аномалій яких немає. Не пиши порад без підстав.

СТРУКТУРА (використовуй ### для заголовків):

### Статус: [NOMINAL / WARNING / CRITICAL]

### Executive Summary
1-2 речення з головним висновком щодо стану БПЛА.

### Root Cause Analysis
Для кожної знайденої аномалії - ланцюжок "5 Чому":
**Симптом** → **Чому?** → **Чому?** → … → **Першопричина**
Якщо аномалій не виявлено, написати "аномалій не виявлено".

### Характер польоту
Тип місії (тестовий/робочий/розвідка), тривалість, дистанція.
Чи повернувся БПЛА (start_to_end_m). Час у русі vs зависання.

### Динаміка швидкості
Макс/середня/P95 горизонтальна, макс вертикальна.
Тренд швидкості: перша vs друга половина.

### Висотний профіль
Діапазон, набір, стабільність (σ), макс. стрибок між точками.

### Навантаження та вібрації
Макс/P95 прискорення. Класифікація: спокійний / помірний / агресивний.
Чи є ознаки механічних проблем (високочастотний шум в IMU → вібрації пропелерів/моторів).

### Якість навігації (GPS)
Джерело (GPS/AHR2), HDOP, супутники. Оцінка надійності.
Чи були GPS-глітчі (різкі стрибки координат, втрата fix).

### Енергосистема
Напруга початок→кінець, просідання, макс. струм.
Оцінка стану батареї за трендом напруги під навантаженням.

### IMU дрейф
Макс. IMU-швидкість vs GPS. Коефіцієнт дрейфу.
Причина: накопичення bias акселерометра δv = δa·t.

### Action Plan
1-3 конкретні дієві кроки для інженера (апаратні, програмні, механічні).
Кожен крок пов'язаний з виявленою першопричиною.

ДАНІ ТЕЛЕМЕТРІЇ:
{json.dumps(context, indent=2, ensure_ascii=False)}"""

    models = ["gemini-2.5-flash", "gemini-2.0-flash"]

    import time

    for model in models:
        url = f"https://generativelanguage.googleapis.com/v1beta/models/{model}:generateContent?key={api_key}"
        for attempt in range(2):
            try:
                response = requests.post(
                    url,
                    json={"contents": [{"parts": [{"text": prompt}]}]},
                    headers={"Content-Type": "application/json"},
                    timeout=60,
                )
                if response.status_code == 429 and attempt < 1:
                    time.sleep(5)
                    continue
                response.raise_for_status()
                data = response.json()
                return data["candidates"][0]["content"]["parts"][0]["text"]
            except Exception:
                if attempt < 1 and response.status_code == 429:
                    time.sleep(5)
                    continue
                break  # try next model

    return _template_report(context)


def _template_report(context: dict) -> str:
    """Генерує звіт за шаблоном (без LLM) у форматі експертної діагностики."""
    duration = context["flight_duration_s"]
    distance = context["total_distance_m"]
    hs = context["horizontal_speed"]
    max_h_speed = hs["max_ms"]
    avg_h_speed = hs.get("avg_ms", 0)
    max_v_speed = context["vertical_speed_max_ms"]
    acc = context["acceleration"]
    alt = context["altitude"]
    anomalies = context["anomalies"]
    gps = context["gps_quality"]

    returned = context.get("returned_to_start", False)
    moving_pct = context.get("time_moving_pct", 50)
    risk = _risk_level(context)

    # --- Executive Summary ---
    if risk == "CRITICAL":
        summary = "Виявлено критичні відхилення параметрів. Рекомендовано наземну інспекцію перед наступним польотом."
    elif risk == "WARNING":
        summary = "Параметри переважно в нормі, але є відхилення що потребують уваги інженера."
    else:
        summary = "Політ пройшов штатно, параметри в межах допустимих норм для мультикоптера."

    report = f"### Статус: {risk}\n\n"
    report += f"### Executive Summary\n{summary}\n\n"

    # --- Root Cause Analysis ---
    root_causes = []
    real_anomalies = [a for a in anomalies if "не виявлено" not in a.lower()]

    for a in real_anomalies:
        if "втрата висоти" in a.lower():
            root_causes.append(
                "**Різка втрата висоти** → Чому? Вертикальна швидкість перевищила 5 м/с → "
                "Чому? Можливі причини: (1) різке зменшення тяги моторів (перевірити RCOUT), "
                "(2) втрата GPS fix → EKF fallback (перевірити XKF3.IVN), "
                "(3) сильний низхідний потік. → **Дія:** перевірити логи RCOUT та BAT на момент інциденту."
            )
        if "швидкість" in a.lower() and "горизонт" in a.lower():
            root_causes.append(
                f"**Висока горизонтальна швидкість ({max_h_speed:.1f} м/с)** → Чому? "
                "Перевищення типової норми 15-20 м/с → "
                "Чому? Можливо: (1) вітрова компонента при русі за вітром, "
                "(2) агресивний waypoint з високим WP_SPEED, (3) ручне керування без обмежень. "
                "→ **Дія:** перевірити WPNAV_SPEED або LOIT_SPEED в параметрах."
            )
        if "прискорення" in a.lower():
            root_causes.append(
                f"**Високе прискорення ({acc['max_ms2']:.1f} м/с²)** → Чому? "
                "Різкі зміни тяги → Чому? Можливо: (1) агресивні команди пілота (RCIN), "
                "(2) вібрації пропелерів (перевірити VibeX/Y/Z < 30 м/с²), "
                "(3) розбалансування моторів (різниця RCOUT каналів > 100). "
                "→ **Дія:** перевірити рівень вібрацій та балансування пропелерів."
            )
        if "супутник" in a.lower():
            root_causes.append(
                "**Низька кількість GPS-супутників** → Чому? "
                "Слабкий прийом сигналу → Чому? Можливо: (1) політ в міській забудові (multipath), "
                "(2) антена GPS затінена конструкцією БПЛА, (3) електромагнітні завади від ESC/моторів. "
                "→ **Дія:** перевірити розташування GPS-антени, мін. відстань від силових кабелів."
            )

    report += "### Root Cause Analysis\n"
    if root_causes:
        for rc in root_causes:
            report += f"- {rc}\n"
    else:
        report += "Критичних аномалій не виявлено. Ланцюжок першопричин не потрібен.\n"
    report += "\n"

    # --- Характер польоту ---
    if duration < 30:
        flight_type = "короткий тестовий"
    elif moving_pct < 30:
        flight_type = "стаціонарний (зависання)"
    elif moving_pct > 80 and distance > 500:
        flight_type = "маршрутний"
    else:
        flight_type = "змішаний (маршрут + зависання)"

    report += f"""### Характер польоту
- **Тип місії:** {flight_type}
- **Тривалість:** {duration:.1f} с ({duration/60:.1f} хв)
- **Дистанція:** {distance:.1f} м ({distance/1000:.2f} км)
- **Повернення до старту:** {"так" if returned else "ні"} (відстань: {context.get('start_to_end_m', 'N/A')} м)
- **Час у русі / зависанні:** {moving_pct:.0f}% / {context.get('time_hovering_pct', 0):.0f}%

"""

    # --- Динаміка швидкості ---
    speed_status = "в нормі" if max_h_speed <= 20 else "ПЕРЕВИЩЕННЯ"
    vspeed_status = "в нормі" if max_v_speed <= 5 else "ПЕРЕВИЩЕННЯ"
    first_half = context.get("speed_first_half_avg_ms", 0)
    second_half = context.get("speed_second_half_avg_ms", 0)
    if first_half and second_half:
        if second_half > first_half * 1.3:
            trend = "зростає (+{:.0f}%)".format((second_half / first_half - 1) * 100)
        elif first_half > second_half * 1.3:
            trend = "спадає (–{:.0f}%)".format((1 - second_half / first_half) * 100)
        else:
            trend = "стабільний"
    else:
        trend = "N/A"

    report += f"""### Динаміка швидкості
- **Макс. горизонтальна:** {max_h_speed:.1f} м/с ({max_h_speed * 3.6:.1f} км/год) - {speed_status}
- **Середня горизонтальна:** {avg_h_speed:.1f} м/с ({avg_h_speed * 3.6:.1f} км/год)
- **P50 / P75 / P95:** {hs.get('p50_ms', 0):.1f} / {hs.get('p75_ms', 0):.1f} / {hs.get('p95_ms', 0):.1f} м/с
- **Макс. вертикальна:** {max_v_speed:.1f} м/с - {vspeed_status}
- **Тренд швидкості:** {trend}

"""

    # --- Висотний профіль ---
    alt_std = context.get('altitude_std_m', 'N/A')
    alt_jump = context.get('max_alt_jump_m', 'N/A')
    alt_stability = "стабільний" if isinstance(alt_std, (int, float)) and alt_std < 5 else "нестабільний"

    report += f"""### Висотний профіль
- **Діапазон:** {alt.get('min_amsl_m', 0):.1f} – {alt.get('max_amsl_m', 0):.1f} м AMSL
- **Набір висоти:** {alt['gain_m']:.1f} м
- **Стабільність (σ):** {alt_std} м - {alt_stability}
- **Макс. стрибок між точками:** {alt_jump} м

"""

    # --- Навантаження ---
    acc_max = acc['max_ms2']
    if acc_max > 8:
        load_class = "АГРЕСИВНИЙ - можливі структурні навантаження"
    elif acc_max > 5:
        load_class = "помірний - активне маневрування"
    else:
        load_class = "спокійний - штатний режим"

    report += f"""### Навантаження та вібрації
- **Макс. прискорення:** {acc_max:.1f} м/с² - {load_class}
- **P95 прискорення:** {acc.get('p95_ms2', 0):.1f} м/с²
- **Примітка:** для повної діагностики вібрацій потрібен аналіз VibeX/Y/Z логів (поріг: < 30 м/с²)

"""

    # --- Якість GPS ---
    hdop = gps.get('mean_hdop', 'N/A')
    if isinstance(hdop, (int, float)):
        gps_quality = "добре" if hdop < 2 else ("задовільно" if hdop < 3 else "НЕНАДІЙНО")
    else:
        gps_quality = "N/A"
    nsats_min = gps.get('min_sats', 'N/A')
    sats_status = ""
    if isinstance(nsats_min, (int, float)):
        sats_status = " - добре" if nsats_min >= 8 else (" - КРИТИЧНО" if nsats_min < 6 else " - прийнятно")

    report += f"""### Якість навігації (GPS)
- **Джерело:** {gps['source']} ({gps['sampling_hz']} Гц)
- **HDOP (середній):** {hdop} - {gps_quality}
- **Супутники:** мін. {nsats_min}{sats_status}, середн. {gps.get('mean_sats', 'N/A')}

"""

    # --- Батарея ---
    bat = context.get("battery", "дані відсутні")
    if isinstance(bat, dict) and bat:
        trend_bat = context.get("battery_trend", {})
        report += "### Енергосистема\n"
        report += f"- **Напруга:** {bat.get('min_voltage_V', 0):.2f} – {bat.get('max_voltage_V', 0):.2f} В\n"
        if trend_bat:
            drop = trend_bat.get('drop_V', 0)
            report += f"- **Початок → кінець:** {trend_bat.get('start_voltage', 0):.2f} → {trend_bat.get('end_voltage', 0):.2f} В (–{drop:.2f} В)\n"
        if bat.get("max_current_A"):
            report += f"- **Макс. струм:** {bat['max_current_A']:.1f} А, середній: {bat.get('mean_current_A', 0):.1f} А\n"
        # Оцінка стану батареї
        min_v = bat.get('min_voltage_V', 99)
        if min_v < 10.5:
            report += "- **УВАГА:** мінімальна напруга критично низька - можливе пошкодження LiPo\n"
        report += "\n"
    else:
        report += "### Енергосистема\nДані батареї відсутні в лозі.\n\n"

    # --- IMU дрейф ---
    imu_spd = context.get("imu_drift_max_speed_ms", 0)
    if isinstance(imu_spd, (int, float)) and imu_spd > 0:
        drift_ratio = imu_spd / max_h_speed if max_h_speed > 0 else 0
        report += f"""### IMU дрейф
- **Макс. IMU швидкість:** {imu_spd:.1f} м/с ({imu_spd * 3.6:.1f} км/год)
- **Макс. GPS швидкість:** {max_h_speed:.1f} м/с
- **Коефіцієнт дрейфу:** {drift_ratio:.0f}x
- **Причина:** накопичення bias акселерометра (δv = δa·t). При bias ≈ 0.05 м/с² за {duration:.0f} с дрейф ≈ {0.05 * duration:.1f} м/с.

"""

    # --- Фази ---
    phases = context.get("flight_phases")
    if phases and isinstance(phases, list):
        phase_names = {"takeoff": "Зліт", "cruise": "Круїз", "hover": "Зависання", "landing": "Посадка"}
        report += "### Фази польоту\n"
        for ph in phases:
            name = phase_names.get(ph["phase"], ph["phase"])
            report += f"- **{name}:** {ph['duration']} с ({ph['start']:.0f}–{ph['end']:.0f} с)\n"
        report += "\n"

    # --- Action Plan ---
    report += "### Action Plan\n"
    actions = []
    if real_anomalies:
        for a in real_anomalies:
            if "висот" in a.lower():
                actions.append("Перевірити логи RCOUT та BAT на момент втрати висоти для виключення механічної/енергетичної причини.")
            if "швидкість" in a.lower() and "горизонт" in a.lower():
                actions.append("Перевірити параметри WPNAV_SPEED / LOIT_SPEED та встановити обмеження згідно ТЗ місії.")
            if "прискорення" in a.lower():
                actions.append("Виконати тест вібрацій (VibeX/Y/Z) та перевірити балансування пропелерів.")
            if "супутник" in a.lower():
                actions.append("Перевірити розташування GPS-антени та мінімальну відстань від силових кабелів ESC.")
    if not actions:
        actions.append("Аномалій не виявлено. Рекомендовано штатне ТО згідно регламенту.")

    for i, act in enumerate(actions[:3], 1):
        report += f"{i}. {act}\n"

    return report


def generate_pdf_report(
    metrics: dict, info: dict, anomalies: list[str],
    bat_df: pd.DataFrame = None, phases: list[dict] | None = None,
    ai_report_text: str | None = None,
    figures: dict[str, "go.Figure"] | None = None,
) -> bytes:
    """
    Генерує PDF-звіт з метриками польоту.

    Returns
    -------
    bytes
        PDF-файл у байтах.
    """
    from fpdf import FPDF

    pdf = FPDF()
    pdf.add_page()

    # Шрифт з підтримкою Unicode (кирилиця)
    font_dir = os.path.join(os.path.dirname(__file__), "..", "assets")
    font_regular = os.path.join(font_dir, "DejaVuSans.ttf")
    font_bold = os.path.join(font_dir, "DejaVuSans-Bold.ttf")

    if os.path.exists(font_regular) and os.path.exists(font_bold):
        pdf.add_font("UAVFont", "", font_regular, uni=True)
        pdf.add_font("UAVFont", "B", font_bold, uni=True)
        font_name = "UAVFont"
    else:
        font_name = "Helvetica"

    def section(title: str):
        pdf.ln(3)
        pdf.set_font(font_name, "B", 11)
        pdf.set_fill_color(230, 240, 250)
        pdf.cell(0, 8, f"  {title}", ln=True, fill=True)
        pdf.set_font(font_name, "", 9)

    def row(label: str, value: str):
        pdf.cell(90, 6, f"  {label}", border="B")
        pdf.cell(0, 6, value, border="B", ln=True)

    # --- Заголовок ---
    pdf.set_font(font_name, "B", 16)
    pdf.cell(0, 12, "UAV Flight Telemetry Report", ln=True, align="C")
    pdf.set_font(font_name, "", 9)
    pdf.set_text_color(100, 100, 100)
    pdf.cell(0, 5, "TheDivass | UAV Telemetry Analyzer", ln=True, align="C")
    pdf.set_text_color(0, 0, 0)
    pdf.ln(2)

    # --- Джерела даних ---
    section("Джерела даних")
    row("GPS джерело", info.get("gps_source", "GPS"))
    row("Частота GPS", f"{info.get('gps_sampling_hz', 'N/A')} Гц ({info.get('gps_count', 0)} повідомлень)")
    row("Частота IMU", f"{info.get('imu_sampling_hz', 'N/A')} Гц ({info.get('imu_count', 0)} повідомлень)")

    # --- Основні метрики ---
    section("Основні метрики польоту")
    duration = metrics.get("total_duration", 0)
    row("Тривалість", f"{duration:.1f} с ({duration / 60:.1f} хв)")
    dist = metrics.get("total_distance", 0)
    row("Дистанція (Haversine)", f"{dist:.1f} м ({dist / 1000:.2f} км)")
    max_hs = metrics.get("max_horizontal_speed", 0)
    row("Макс. горизонтальна швидкість", f"{max_hs:.1f} м/с ({max_hs * 3.6:.1f} км/год)")
    avg_hs = metrics.get("avg_horizontal_speed", 0)
    row("Середня горизонтальна швидкість", f"{avg_hs:.1f} м/с ({avg_hs * 3.6:.1f} км/год)")
    row("Макс. вертикальна швидкість", f"{metrics.get('max_vertical_speed', 0):.1f} м/с")

    # --- Висота ---
    section("Висотний профіль")
    row("Набір висоти", f"{metrics.get('max_altitude_gain', 0):.1f} м")
    row("Висота мін / макс", f"{metrics.get('alt_min', 0):.1f} / {metrics.get('alt_max', 0):.1f} м AMSL")
    row("Середня висота", f"{metrics.get('alt_mean', 0):.1f} м AMSL")

    # --- Швидкісні перцентилі ---
    section("Розподіл швидкості")
    row("P50 (медіана)", f"{metrics.get('speed_p50', 0):.1f} м/с")
    row("P75", f"{metrics.get('speed_p75', 0):.1f} м/с")
    row("P95", f"{metrics.get('speed_p95', 0):.1f} м/с")

    # --- Прискорення ---
    section("Динамічні навантаження")
    row("Макс. прискорення (без g)", f"{metrics.get('max_acceleration', 0):.1f} м/с²")
    row("P95 прискорення", f"{metrics.get('acc_p95', 0):.1f} м/с²")

    # --- GPS якість ---
    section("Якість GPS")
    if metrics.get("hdop_mean"):
        row("Середній HDOP", f"{metrics['hdop_mean']:.1f}")
    if metrics.get("nsats_mean"):
        row("Середня кількість супутників", f"{metrics['nsats_mean']:.0f}")
    if metrics.get("nsats_min") is not None:
        row("Мін. супутників", str(metrics.get("nsats_min", "N/A")))

    # --- IMU дрейф ---
    section("IMU інтегрування (демонстрація дрейфу)")
    imu_spd = metrics.get("max_imu_speed", 0)
    row("Макс. IMU швидкість", f"{imu_spd:.1f} м/с ({imu_spd * 3.6:.1f} км/год)")
    row("Макс. GPS швидкість", f"{max_hs:.1f} м/с ({max_hs * 3.6:.1f} км/год)")
    drift = imu_spd - max_hs if imu_spd and max_hs else 0
    row("Дрейф (IMU - GPS)", f"{drift:+.1f} м/с")

    # --- Батарея ---
    if bat_df is not None and not bat_df.empty and "voltage" in bat_df.columns:
        section("Батарея")
        v = bat_df["voltage"].values
        v = v[v > 0]
        if len(v):
            row("Напруга мін / макс", f"{v.min():.2f} / {v.max():.2f} В")
            row("Середня напруга", f"{v.mean():.2f} В")
            row("Просідання", f"{v.max() - v.min():.2f} В")
        if "current" in bat_df.columns:
            c = bat_df["current"].values
            c = c[c >= 0]
            if len(c):
                row("Макс. струм", f"{c.max():.1f} А")
                row("Середній струм", f"{c.mean():.1f} А")

    # --- Фази польоту ---
    if phases:
        section("Фази польоту")
        phase_names = {"takeoff": "Зліт", "cruise": "Круїз", "hover": "Зависання", "landing": "Посадка"}
        for ph in phases:
            name = phase_names.get(ph["phase"], ph["phase"])
            row(name, f"{ph['duration']} с  ({ph['start']:.0f}–{ph['end']:.0f} с)")

    # --- Аномалії ---
    section("Виявлені аномалії")
    for a in anomalies:
        pdf.cell(0, 6, f"  • {a}", ln=True)

    # --- Оцінка ---
    pdf.ln(4)
    pdf.set_font(font_name, "B", 10)
    max_acc = metrics.get("max_acceleration", 0)
    max_vs = metrics.get("max_vertical_speed", 0)
    if max_acc > 10 or max_vs > 10:
        assessment = "Агресивний маневровий політ з високими навантаженнями."
    elif max_hs > 15:
        assessment = "Швидкісний політ з помірними навантаженнями."
    elif duration < 30:
        assessment = "Короткий тестовий або аварійний політ."
    else:
        assessment = "Стабільний політ з нормальними параметрами."
    pdf.cell(0, 7, f"Загальна оцінка: {assessment}", ln=True)

    # --- Графіки ---
    if figures:
        import tempfile

        img_w = pdf.w - 20  # ширина зображення з відступами
        img_h = img_w * (400 / 900)  # пропорція render 900x400
        block_h = img_h + 12  # зображення + заголовок + відступ

        chart_order = [
            ("speed_profile", "Профіль швидкості та висоти"),
            ("trajectory_3d", "3D-траєкторія (ENU)"),
            ("gps_quality", "Якість GPS-сигналу"),
            ("acceleration", "Профіль прискорення"),
            ("imu_comparison", "GPS vs IMU швидкість"),
            ("battery", "Батарея"),
        ]

        pdf.ln(5)
        for key, title in chart_order:
            fig = figures.get(key)
            if fig is None:
                continue
            try:
                img_bytes = fig.to_image(format="png", width=900, height=400, scale=2)
                with tempfile.NamedTemporaryFile(suffix=".png", delete=False) as tmp:
                    tmp.write(img_bytes)
                    tmp_path = tmp.name

                # Нова сторінка тільки якщо блок не влізає
                space_left = pdf.h - pdf.b_margin - pdf.get_y()
                if space_left < block_h:
                    pdf.add_page()

                pdf.set_left_margin(10)
                pdf.set_x(10)
                pdf.set_font(font_name, "B", 11)
                pdf.cell(0, 7, title, ln=True, align="C")
                pdf.ln(1)
                pdf.image(tmp_path, x=10, w=img_w)
                pdf.ln(4)
                os.unlink(tmp_path)
            except Exception:
                continue

    # --- AI-звіт ---
    if ai_report_text:
        # Нова сторінка тільки якщо мало місця для заголовка + перших рядків
        if pdf.h - pdf.b_margin - pdf.get_y() < 40:
            pdf.add_page()
        pdf.ln(5)
        pdf.set_left_margin(10)
        pdf.set_x(10)
        pdf.set_font(font_name, "B", 14)
        pdf.cell(0, 10, "AI-аналіз польоту", ln=True, align="C")
        pdf.ln(3)
        pdf.set_font(font_name, "", 9)
        # Очищуємо markdown-розмітку для PDF
        clean = ai_report_text
        for md_char in ["###", "##", "#", "**", "*", "---"]:
            clean = clean.replace(md_char, "")
        for line in clean.split("\n"):
            line = line.strip()
            if not line:
                pdf.ln(2)
                continue
            pdf.set_x(10)
            pdf.multi_cell(w=0, h=5, text=line)

    # --- Футер ---
    pdf.ln(5)
    pdf.set_font(font_name, "", 7)
    pdf.set_text_color(130, 130, 130)
    pdf.cell(0, 4, "Згенеровано UAV Telemetry Analyzer | TheDivass | BEST HACKath0n 2026", ln=True, align="C")

    return bytes(pdf.output())
