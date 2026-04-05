"""
UAV Flight Telemetry Analysis & 3D Visualization
=================================================
Streamlit веб-додаток для аналізу лог-файлів Ardupilot.

Запуск: streamlit run app.py
"""

import os
import tempfile

import streamlit as st
import pandas as pd

from src.parser import parse_bin_file
from src.metrics import compute_flight_metrics, detect_flight_phases
from src.visualization import (
    create_trajectory_figure,
    create_speed_profile,
    create_imu_comparison,
    create_battery_chart,
    create_gps_quality_chart,
    create_acceleration_chart,
    create_2d_map,
)
from src.ai_report import generate_flight_report, detect_anomalies, generate_pdf_report

# --- Page config ---
st.set_page_config(
    page_title="UAV Telemetry Analyzer",
    page_icon="✈",
    layout="wide",
)

st.title("Система аналiзу телеметрiї та 3D-вiзуалiзацiї польотiв БПЛА")
st.caption("Парсинг лог-файлiв Ardupilot | WGS-84 -> ENU | Haversine & Trapezoidal Integration")

# --- Sidebar: file selection ---
st.sidebar.header("Вхiднi данi")

# Пошук .BIN файлів у поточній директорії (обидва регістри)
data_dir = os.path.dirname(os.path.abspath(__file__))
local_bins = sorted([f for f in os.listdir(data_dir) if f.lower().endswith(".bin")])

source = st.sidebar.radio(
    "Джерело даних:",
    ["Локальнi файли", "Завантажити файл"],
)

bin_path = None

bin_path_2 = None  # Другий файл для порівняння

if source == "Локальнi файли" and local_bins:
    selected_file = st.sidebar.selectbox("Оберiть лог-файл:", local_bins)
    bin_path = os.path.join(data_dir, selected_file)
    # Порівняння двох польотів
    if len(local_bins) > 1:
        compare = st.sidebar.checkbox("Порiвняти з iншим польотом")
        if compare:
            other_bins = [f for f in local_bins if f != selected_file]
            selected_file_2 = st.sidebar.selectbox("Другий лог-файл:", other_bins)
            bin_path_2 = os.path.join(data_dir, selected_file_2)
elif source == "Завантажити файл":
    uploaded = st.sidebar.file_uploader("Завантажте .BIN файл", type=["bin", "BIN"])
    if uploaded:
        upload_key = f"{uploaded.name}_{uploaded.size}"
        if st.session_state.get("upload_key") != upload_key:
            with tempfile.NamedTemporaryFile(delete=False, suffix=".BIN") as tmp:
                tmp.write(uploaded.read())
                bin_path = tmp.name
            st.session_state["upload_key"] = upload_key
            st.session_state["upload_bin_path"] = bin_path
        else:
            bin_path = st.session_state["upload_bin_path"]

if not local_bins and source == "Локальнi файли":
    st.sidebar.warning("Немає .BIN файлiв у директорiї проєкту")

# --- Sidebar: settings ---
st.sidebar.header("Налаштування")

# Візуалізація
st.sidebar.subheader("Вiзуалiзацiя")

color_mode = st.sidebar.selectbox(
    "Колорування траєкторiї:",
    ["speed", "time"],
    format_func=lambda x: "За швидкiстю" if x == "speed" else "За часом",
)

colorscale = st.sidebar.selectbox(
    "Колiрна схема:",
    ["Viridis", "Plasma", "Turbo", "Jet", "Inferno", "Cividis"],
)

speed_unit = st.sidebar.selectbox(
    "Одиницi швидкостi:",
    ["kmh", "ms"],
    format_func=lambda x: "км/год" if x == "kmh" else "м/с",
)

line_width = st.sidebar.slider("Товщина лiнiї траєкторiї", 1, 6, 2)
# 3D опції
st.sidebar.subheader("3D опцiї")
show_terrain = st.sidebar.checkbox("Показати рельєф (SRTM)", value=True)
show_shadow = st.sidebar.checkbox("Проекцiя на землю", value=True)
show_markers = st.sidebar.checkbox("Маркери старт/фiнiш", value=True)

# Фільтрація
st.sidebar.subheader("Фiльтрацiя")
smooth_window = st.sidebar.slider("Згладжування швидкостi (вiкно)", 1, 15, 1, step=2,
                                   help="1 = без згладжування")

# AI
st.sidebar.subheader("AI-звiт")
gemini_key = st.sidebar.text_input("Gemini API Key (опцiонально)", type="password")


# --- Main processing ---
def _load_and_process(filepath: str):
    data = parse_bin_file(filepath)
    metrics = compute_flight_metrics(data["gps"], data["imu"])
    return data, metrics


if bin_path:
    cache_key = f"parsed_{bin_path}"

    if cache_key not in st.session_state:
        with st.spinner("Парсинг лог-файлу..."):
            data, metrics = _load_and_process(bin_path)
        st.session_state[cache_key] = (data, metrics)
        # Видаляємо кеш попереднього файлу
        for k in list(st.session_state.keys()):
            if k.startswith("parsed_") and k != cache_key:
                del st.session_state[k]

    data, metrics = st.session_state[cache_key]
    gps_df = data["gps"]
    imu_df = data["imu"]
    bat_df = data.get("bat", pd.DataFrame())
    info = data["info"]

    # Коефіцієнт для відображення швидкості
    spd_factor = 3.6 if speed_unit == "kmh" else 1.0
    spd_label = "км/год" if speed_unit == "kmh" else "м/с"

    # --- Time range filter ---
    if not gps_df.empty and len(gps_df) > 1:
        t_min = float(gps_df["time_s"].min())
        t_max = float(gps_df["time_s"].max())
        time_range = st.sidebar.slider(
            "Часовий дiапазон (с)",
            min_value=t_min, max_value=t_max,
            value=(t_min, t_max),
            step=0.5,
        )
        mask = (gps_df["time_s"] >= time_range[0]) & (gps_df["time_s"] <= time_range[1])
        gps_df = gps_df[mask].reset_index(drop=True)
        imu_mask = (imu_df["time_s"] >= time_range[0]) & (imu_df["time_s"] <= time_range[1])
        imu_df = imu_df[imu_mask].reset_index(drop=True)
        # Recalculate metrics for filtered range
        metrics = compute_flight_metrics(gps_df, imu_df)

    # --- Smoothing ---
    if smooth_window > 1 and not gps_df.empty:
        gps_df = gps_df.copy()
        gps_df["speed"] = gps_df["speed"].rolling(window=smooth_window, center=True, min_periods=1).mean()
        gps_df["vz"] = gps_df["vz"].rolling(window=smooth_window, center=True, min_periods=1).mean()

    # --- Metrics cards ---
    st.header("Метрики польоту")
    col1, col2, col3 = st.columns(3)
    col4, col5, col6 = st.columns(3)

    col1.metric("Тривалiсть", f"{metrics['total_duration']:.1f} с")
    col2.metric("Дистанцiя (haversine)", f"{metrics['total_distance']:.1f} м")
    col3.metric("Макс. набiр висоти", f"{metrics['max_altitude_gain']:.1f} м")
    col4.metric(
        f"Макс. гориз. швидкiсть ({spd_label})",
        f"{metrics['max_horizontal_speed'] * spd_factor:.1f} {spd_label}",
    )
    col5.metric(
        f"Макс. верт. швидкiсть ({spd_label})",
        f"{metrics['max_vertical_speed'] * spd_factor:.1f} {spd_label}",
    )
    col6.metric("Макс. прискорення", f"{metrics['max_acceleration']:.1f} м/с²")

    # --- Sampling info ---
    with st.expander("Iнформацiя про семплювання"):
        icol1, icol2 = st.columns(2)
        icol1.write(f"**GPS:** {info.get('gps_sampling_hz', 'N/A')} Гц ({info.get('gps_count', 0)} повiдомлень)")
        icol2.write(f"**IMU:** {info.get('imu_sampling_hz', 'N/A')} Гц ({info.get('imu_count', 0)} повiдомлень)")
        if not bat_df.empty:
            st.write(
                f"**BAT:** {len(bat_df)} зап. | "
                f"напруга: {bat_df['voltage'].min():.2f}–{bat_df['voltage'].max():.2f} В"
            )

    # --- Flight phases ---
    phases = detect_flight_phases(gps_df)
    if phases:
        st.header("Фази польоту")
        phase_names = {"takeoff": "Злiт", "cruise": "Круїз", "hover": "Зависання", "landing": "Посадка"}
        phase_colors = {"takeoff": "🟢", "cruise": "🔵", "hover": "🟠", "landing": "🔴"}
        cols = st.columns(len(phases))
        for i, ph in enumerate(phases):
            cols[i].metric(
                f"{phase_colors.get(ph['phase'], '')} {phase_names.get(ph['phase'], ph['phase'])}",
                f"{ph['duration']} с",
                delta=f"{ph['start']:.0f}–{ph['end']:.0f} с",
                delta_color="off",
            )

    # --- 2D Map ---
    st.header("2D-карта траєкторiї")
    flight_map = create_2d_map(gps_df, phases if phases else None)
    if flight_map:
        from streamlit_folium import st_folium
        st_folium(flight_map, width=None, height=450, returned_objects=[])

    # --- 3D Trajectory ---
    st.header("3D-траєкторiя (ENU)")
    fig_3d = create_trajectory_figure(
        gps_df,
        color_by=color_mode,
        speed_unit=speed_unit,
        show_terrain=show_terrain,
        show_shadow=show_shadow,
        show_markers=show_markers,
        bat_df=bat_df if not bat_df.empty else None,
        colorscale_name=colorscale,
        line_width=line_width,
        chart_height=600,
        alt_mode="relative",
    )
    st.plotly_chart(fig_3d, width="stretch", key="chart_3d")

    # --- Speed profile ---
    st.header("Профiль швидкостi та висоти")
    fig_speed = create_speed_profile(gps_df, speed_unit=speed_unit)
    st.plotly_chart(fig_speed, width="stretch", key="chart_speed")

    # --- IMU comparison ---
    st.header("GPS vs IMU швидкiсть")
    st.caption("Демонстрацiя дрейфу при трапецiєвидному iнтегруваннi прискорень IMU")
    imu_vel = metrics.get("imu_velocities", pd.DataFrame())
    if not imu_vel.empty:
        fig_imu = create_imu_comparison(gps_df, imu_vel, speed_unit=speed_unit)
        st.plotly_chart(fig_imu, width="stretch", key="chart_imu")

    # --- Battery ---
    if not bat_df.empty:
        st.header("Батарея")
        fig_bat = create_battery_chart(bat_df)
        st.plotly_chart(fig_bat, width="stretch", key="chart_bat")

    # --- GPS Quality ---
    st.header("Якiсть GPS-сигналу")
    fig_gps_q = create_gps_quality_chart(gps_df)
    st.plotly_chart(fig_gps_q, width="stretch", key="chart_gps_quality")

    # --- Acceleration profile ---
    st.header("Профiль прискорення")
    fig_acc = create_acceleration_chart(imu_df)
    st.plotly_chart(fig_acc, width="stretch", key="chart_acc")

    # --- AI Report ---
    st.header("AI-аналiз польоту")
    ai_col1, ai_col2 = st.columns([3, 1])
    with ai_col1:
        gen_btn = st.button("Згенерувати звiт")
    with ai_col2:
        use_template = st.checkbox("Без API (шаблон)", value=not bool(gemini_key))

    if gen_btn:
        with st.spinner("Генерацiя звiту..."):
            report = generate_flight_report(
                metrics, gps_df, info,
                api_key=None if use_template else (gemini_key or None),
                bat_df=bat_df if not bat_df.empty else None,
                phases=phases if phases else None,
            )
        st.session_state["ai_report"] = report
        st.markdown(report)
    elif "ai_report" in st.session_state:
        st.markdown(st.session_state["ai_report"])

    # --- Export ---
    st.sidebar.subheader("Експорт")
    export_data = {
        "Тривалiсть (с)": [metrics["total_duration"]],
        "Дистанцiя (м)": [metrics["total_distance"]],
        "Макс. гориз. швидкiсть (м/с)": [metrics["max_horizontal_speed"]],
        "Макс. верт. швидкiсть (м/с)": [metrics["max_vertical_speed"]],
        "Макс. прискорення (м/с²)": [metrics["max_acceleration"]],
        "Макс. набiр висоти (м)": [metrics["max_altitude_gain"]],
    }
    csv_data = pd.DataFrame(export_data).to_csv(index=False)
    st.sidebar.download_button(
        "Завантажити метрики (CSV)",
        csv_data,
        file_name="flight_metrics.csv",
        mime="text/csv",
    )

    # PDF-звіт
    anomalies = detect_anomalies(gps_df, metrics)
    ai_text = st.session_state.get("ai_report")

    if st.sidebar.button("Згенерувати PDF"):
        with st.sidebar.status("Генерацiя PDF...", expanded=True) as status:
            pdf_figures = {}

            status.update(label="Рендеринг графiкiв...")
            status.write("Профiль швидкостi...")
            pdf_figures["speed_profile"] = fig_speed

            status.write("Якiсть GPS...")
            pdf_figures["gps_quality"] = fig_gps_q

            status.write("Профiль прискорення...")
            pdf_figures["acceleration"] = fig_acc

            if not imu_vel.empty:
                status.write("GPS vs IMU...")
                pdf_figures["imu_comparison"] = fig_imu
            if not bat_df.empty:
                status.write("Батарея...")
                pdf_figures["battery"] = fig_bat

            status.update(label="Формування PDF...")
            status.write("Метрики та таблицi...")
            try:
                pdf_bytes = generate_pdf_report(
                    metrics, info, anomalies,
                    bat_df=bat_df if not bat_df.empty else None,
                    phases=phases if phases else None,
                    ai_report_text=ai_text,
                    figures=pdf_figures,
                )
                st.session_state["pdf_bytes"] = pdf_bytes
                status.update(label="PDF готовий!", state="complete", expanded=False)
            except Exception as e:
                status.update(label=f"PDF помилка: {e}", state="error")
                st.sidebar.warning(f"PDF помилка: {e}")

    if "pdf_bytes" in st.session_state:
        st.sidebar.download_button(
            "Завантажити звiт (PDF)",
            st.session_state["pdf_bytes"],
            file_name="flight_report.pdf",
            mime="application/pdf",
            key="download_pdf",
        )

    # --- Raw data ---
    with st.expander("Сирi данi GPS"):
        st.dataframe(gps_df.drop(columns=["time_us", "instance"], errors="ignore"), width="stretch")

    with st.expander("Сирi данi IMU (перші 500 рядків)"):
        st.dataframe(imu_df.head(500).drop(columns=["time_us"], errors="ignore"), width="stretch")

    if not bat_df.empty:
        with st.expander("Сирi данi батареї"):
            st.dataframe(bat_df.drop(columns=["time_us"], errors="ignore"), width="stretch")

    # --- Flight comparison ---
    if bin_path_2:
        st.header("Порiвняння польотiв")
        cache_key_2 = f"parsed_{bin_path_2}"
        if cache_key_2 not in st.session_state:
            with st.spinner("Парсинг другого лог-файлу..."):
                data2, metrics2 = _load_and_process(bin_path_2)
            st.session_state[cache_key_2] = (data2, metrics2)
        data2, metrics2 = st.session_state[cache_key_2]

        compare_keys = [
            ("total_duration", "Тривалiсть (с)"),
            ("total_distance", "Дистанцiя (м)"),
            ("max_horizontal_speed", "Макс. гориз. швидк. (м/с)"),
            ("max_vertical_speed", "Макс. верт. швидк. (м/с)"),
            ("max_acceleration", "Макс. прискорення (м/с²)"),
            ("max_altitude_gain", "Набiр висоти (м)"),
        ]

        comp_data = {"Метрика": [], "Полiт 1": [], "Полiт 2": [], "Рiзниця": []}
        for key, label in compare_keys:
            v1 = metrics.get(key, 0)
            v2 = metrics2.get(key, 0)
            comp_data["Метрика"].append(label)
            comp_data["Полiт 1"].append(round(v1, 2))
            comp_data["Полiт 2"].append(round(v2, 2))
            diff = v1 - v2
            comp_data["Рiзниця"].append(f"{diff:+.2f}")

        st.dataframe(pd.DataFrame(comp_data), hide_index=True, width="stretch")

else:
    st.info("Оберiть або завантажте .BIN файл для початку аналiзу.")
