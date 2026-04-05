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
        "imu_drift_max_speed_ms": metrics.get("max_imu_speed", "N/A"),
        "flight_phases": phases if phases else "не визначено",
    }

    # Додаткова GPS-статистика
    if not gps_df.empty:
        alt = gps_df["alt"].values
        speed = gps_df["speed"].values
        context["altitude_std_m"] = round(float(alt.std()), 2)
        context["speed_std_ms"] = round(float(speed.std()), 2)
        # Відстань старт-фініш (пряма)
        from .coordinates import haversine
        context["start_to_end_m"] = round(
            haversine(gps_df["lat"].iloc[0], gps_df["lng"].iloc[0],
                      gps_df["lat"].iloc[-1], gps_df["lng"].iloc[-1]), 1
        )

    api_key = api_key or os.environ.get("GEMINI_API_KEY", "")

    if api_key:
        return _call_gemini(context, api_key)
    else:
        return _template_report(context)


def _call_gemini(context: dict, api_key: str) -> str:
    """Викликає Google Gemini API для генерації звіту."""
    prompt = f"""Ти - експерт з аналізу телеметрії БПЛА. Проаналізуй дані та видай стислий звіт УКРАЇНСЬКОЮ.

ПРАВИЛА:
- Тільки факти і конкретні числа, без води та загальних фраз
- Кожен пункт - 1-2 речення максимум
- Якщо аномалій немає - так і напиши, не вигадуй
- Порівнюй з типовими значеннями для мультикоптерів (макс. ~20 м/с горизонт, ~5 м/с вертик)

СТРУКТУРА:
1. **Загальне** - тип польоту, тривалість, дистанція, повернувся чи ні (start_to_end_m)
2. **Швидкість** - макс/середня/P95, вертикальна. Чи є перевищення
3. **Висота** - діапазон, набір, стабільність (std)
4. **Навантаження** - прискорення макс/P95, чи були різкі маневри
5. **GPS** - якість (HDOP, супутники), джерело даних
6. **Батарея** - просідання напруги, макс струм (якщо є дані)
7. **Фази** - скільки фаз, їх тривалість
8. **IMU дрейф** - макс IMU-швидкість vs GPS (демонстрація похибки інтегрування)
9. **Висновок** - 1 речення: загальна оцінка польоту

Дані:
{json.dumps(context, indent=2, ensure_ascii=False)}"""

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
        chart_order = [
            ("speed_profile", "Профіль швидкості та висоти"),
            ("trajectory_3d", "3D-траєкторія (ENU)"),
            ("gps_quality", "Якість GPS-сигналу"),
            ("acceleration", "Профіль прискорення"),
            ("imu_comparison", "GPS vs IMU швидкість"),
            ("battery", "Батарея"),
        ]
        for key, title in chart_order:
            fig = figures.get(key)
            if fig is None:
                continue
            try:
                img_bytes = fig.to_image(format="png", width=900, height=400, scale=2)
                with tempfile.NamedTemporaryFile(suffix=".png", delete=False) as tmp:
                    tmp.write(img_bytes)
                    tmp_path = tmp.name
                pdf.add_page()
                pdf.set_left_margin(10)
                pdf.set_x(10)
                pdf.set_font(font_name, "B", 12)
                pdf.cell(0, 8, title, ln=True, align="C")
                pdf.ln(2)
                # Вписуємо зображення на ширину сторінки з відступами
                pdf.image(tmp_path, x=10, w=pdf.w - 20)
                os.unlink(tmp_path)
            except Exception:
                continue

    # --- AI-звіт ---
    if ai_report_text:
        pdf.add_page()
        pdf.set_left_margin(10)
        pdf.set_x(10)
        pdf.set_font(font_name, "B", 14)
        pdf.cell(0, 10, "AI-аналіз польоту (Gemini)", ln=True, align="C")
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
