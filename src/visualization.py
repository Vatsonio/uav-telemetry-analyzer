"""
Модуль 3D-візуалізації траєкторії польоту.

Будує інтерактивні 3D-графіки за допомогою Plotly з:
- Траєкторією у локальній системі координат ENU (метри від точки старту)
- Динамічним колоруванням за швидкістю або часом (адаптивна шкала, 95-й перцентиль)
- Маркерами старту/фінішу
- Опорною площиною землі та тіньовою проекцією
- Повними hover-підказками (GPS координати, швидкість, напруга)
"""

import numpy as np
import pandas as pd
import plotly.graph_objects as go

from .coordinates import wgs84_to_enu

_SPEED_FACTOR = {"kmh": 3.6, "ms": 1.0}
_SPEED_LABEL = {"kmh": "км/год", "ms": "м/с"}


def _get_terrain_grid(
    lats: np.ndarray, lngs: np.ndarray,
    lat0: float, lon0: float, alt0: float,
    east: np.ndarray, north: np.ndarray,
) -> tuple[dict, bool]:
    """Завантажує SRTM висоти для сітки навколо траєкторії, повертає ENU координати."""
    try:
        import srtm
        from .coordinates import wgs84_to_enu

        elevation_data = srtm.get_data()

        # Сітка GPS координат навколо траєкторії
        lat_min, lat_max = lats.min(), lats.max()
        lng_min, lng_max = lngs.min(), lngs.max()
        lat_pad = max((lat_max - lat_min) * 0.5, 0.002)
        lng_pad = max((lng_max - lng_min) * 0.5, 0.002)

        grid_n = 30
        grid_lats = np.linspace(lat_min - lat_pad, lat_max + lat_pad, grid_n)
        grid_lngs = np.linspace(lng_min - lng_pad, lng_max + lng_pad, grid_n)

        pts_lat, pts_lng, pts_elev = [], [], []
        for la in grid_lats:
            for lo in grid_lngs:
                h = elevation_data.get_elevation(la, lo)
                if h is not None:
                    pts_lat.append(la)
                    pts_lng.append(lo)
                    pts_elev.append(h)

        if len(pts_elev) < 9:
            return {}, False

        pts_lat = np.array(pts_lat)
        pts_lng = np.array(pts_lng)
        pts_elev = np.array(pts_elev, dtype=float)

        # Конвертуємо в ENU відносно тієї ж точки що й траєкторія
        e, n, u = wgs84_to_enu(pts_lat, pts_lng, np.full_like(pts_elev, alt0), lat0, lon0, alt0)
        # Висота terrain = різниця SRTM від alt0
        terrain_up = pts_elev - alt0

        return {"east": e, "north": n, "elev": terrain_up}, True
    except Exception:
        return {}, False


def _robust_max(values: np.ndarray, percentile: float = 95.0) -> float:
    """Верхній поріг за перцентилем (фільтрація викидів)."""
    valid = values[np.isfinite(values)]
    if len(valid) == 0:
        return 1.0
    return float(np.percentile(valid, percentile))


def _match_voltage(gps_df: pd.DataFrame, bat_df: pd.DataFrame) -> np.ndarray:
    """Прив'язує напругу батареї до GPS-точок по найближчому часу."""
    if bat_df is None or bat_df.empty or "time_s" not in bat_df.columns:
        return np.full(len(gps_df), float("nan"))
    merged = pd.merge_asof(
        gps_df[["time_s"]].copy().sort_values("time_s"),
        bat_df[["time_s", "voltage"]].sort_values("time_s"),
        on="time_s",
        direction="nearest",
    )
    return merged["voltage"].values


def create_trajectory_figure(
    gps_df: pd.DataFrame,
    color_by: str = "speed",
    speed_unit: str = "kmh",
    show_terrain: bool = True,
    show_shadow: bool = True,
    show_markers: bool = True,
    bat_df: pd.DataFrame = None,
    colorscale_name: str = "Viridis",
    line_width: int = 2,
    chart_height: int = 600,
    alt_mode: str = "relative",
) -> go.Figure:
    """
    Створює інтерактивний 3D-графік траєкторії польоту.

    Parameters
    ----------
    gps_df : pd.DataFrame
        GPS дані з колонками: lat, lng, alt, speed, time_s
    color_by : str
        Параметр колорування: 'speed' або 'time'
    speed_unit : str
        Одиниці швидкості: 'kmh' або 'ms'
    show_terrain : bool
        Показувати опорну площину землі та тіньову проекцію
    bat_df : pd.DataFrame, optional
        Дані батареї для відображення напруги у підказці
    """
    if gps_df.empty or len(gps_df) < 2:
        fig = go.Figure()
        fig.add_annotation(text="Недостатньо GPS-даних для візуалізації", showarrow=False)
        return fig

    factor = _SPEED_FACTOR[speed_unit]
    unit_label = _SPEED_LABEL[speed_unit]

    lat0 = gps_df["lat"].iloc[0]
    lon0 = gps_df["lng"].iloc[0]
    alt0 = gps_df["alt"].iloc[0]

    east, north, up = wgs84_to_enu(
        gps_df["lat"].values, gps_df["lng"].values, gps_df["alt"].values,
        lat0, lon0, alt0,
    )

    # AMSL mode: показуємо абсолютну висоту замість відносної
    if alt_mode == "amsl":
        up = gps_df["alt"].values.copy()

    # --- Вибір параметра для колорування ---
    if color_by == "speed":
        color_values = gps_df["speed"].values * factor
        color_label = f"Швидкість ({unit_label})"
        colorscale = colorscale_name
        # Адаптивний верхній поріг (95-й перцентиль)
        cmax = _robust_max(color_values, 95)
        cmin = 0.0
    else:
        color_values = gps_df["time_s"].values - gps_df["time_s"].values[0]
        color_label = "Час (с)"
        colorscale = colorscale_name
        cmax = float(color_values.max()) if len(color_values) else 1.0
        cmin = 0.0

    # Hover з GPS-координатами, швидкістю та напругою
    voltage = _match_voltage(gps_df, bat_df)
    speed_display = gps_df["speed"].values * factor
    custom_data = np.column_stack([
        gps_df["lat"].values,
        gps_df["lng"].values,
        speed_display,
        voltage,
    ])

    has_voltage = not np.all(np.isnan(voltage))
    volt_line = "Напруга: %{customdata[3]:.2f} В<br>" if has_voltage else ""
    hovertemplate = (
        "East: %{x:.1f} м | North: %{y:.1f} м | Up: %{z:.1f} м<br>"
        f"Швидкість: %{{customdata[2]:.1f}} {unit_label}<br>"
        "GPS: %{customdata[0]:.6f}°, %{customdata[1]:.6f}°<br>"
        + volt_line
        + "<extra></extra>"
    )

    fig = go.Figure()

    # --- Terrain: SRTM elevation or flat fallback ---
    if show_terrain:
        terrain_z, terrain_ok = _get_terrain_grid(
            gps_df["lat"].values, gps_df["lng"].values, lat0, lon0, alt0,
            east, north,
        )

        if terrain_ok:
            pad_e = max(terrain_z["east"].max() - terrain_z["east"].min(), 50) * 0.05
            pad_n = max(terrain_z["north"].max() - terrain_z["north"].min(), 50) * 0.05
            x_g = np.linspace(terrain_z["east"].min() + pad_e, terrain_z["east"].max() - pad_e, 40)
            y_g = np.linspace(terrain_z["north"].min() + pad_n, terrain_z["north"].max() - pad_n, 40)
        else:
            traj_h_range = max(east.max() - east.min(), north.max() - north.min())
            pad = max(traj_h_range * 0.08, 5.0)
            x_g = np.linspace(east.min() - pad, east.max() + pad, 10)
            y_g = np.linspace(north.min() - pad, north.max() + pad, 10)
        xx, yy = np.meshgrid(x_g, y_g)

        if terrain_ok:
            from scipy.interpolate import griddata
            zz = griddata(
                (terrain_z["east"], terrain_z["north"]),
                terrain_z["elev"],
                (xx, yy),
                method="cubic",
                fill_value=0.0,
            )
            terrain_colorscale = [
                [0.0, "rgb(60,100,40)"],
                [0.3, "rgb(101,140,74)"],
                [0.6, "rgb(160,150,100)"],
                [1.0, "rgb(139,115,85)"],
            ]
        else:
            zz = np.zeros_like(xx)
            terrain_colorscale = [[0, "rgb(101,140,74)"], [1, "rgb(139,115,85)"]]

        surface_kwargs = dict(
            x=xx, y=yy, z=zz,
            colorscale=terrain_colorscale,
            showscale=False,
            opacity=0.70 if terrain_ok else 0.30,
            name="Рельєф" if terrain_ok else "Поверхня землі",
            hoverinfo="skip",
        )
        if terrain_ok:
            surface_kwargs["surfacecolor"] = zz
            surface_kwargs["contours"] = dict(
                z=dict(show=True, usecolormap=True, project_z=True, highlightcolor="white", size=1),
            )
        fig.add_trace(go.Surface(**surface_kwargs))

        # Тіньова проекція траєкторії на terrain
        if show_shadow:
            if terrain_ok:
                from scipy.interpolate import griddata as _gd
                shadow_z = _gd(
                    (terrain_z["east"], terrain_z["north"]),
                    terrain_z["elev"],
                    (east, north),
                    method="linear",
                    fill_value=0.0,
                )
            else:
                shadow_z = np.zeros_like(up)

            fig.add_trace(go.Scatter3d(
                x=east, y=north, z=shadow_z,
                mode="lines",
                line=dict(color="rgba(60,60,60,0.35)", width=1),
                name="Проекція на землю",
                hoverinfo="skip",
                showlegend=True,
            ))

    # --- Основна траєкторія ---
    fig.add_trace(go.Scatter3d(
        x=east, y=north, z=up,
        mode="lines+markers",
        marker=dict(
            size=3,
            color=color_values,
            colorscale=colorscale,
            cmin=cmin,
            cmax=cmax,
            colorbar=dict(title=color_label, x=1.02, len=0.6, thickness=15),
            showscale=True,
        ),
        line=dict(color="rgba(100,100,100,0.3)", width=line_width),
        customdata=custom_data,
        hovertemplate=hovertemplate,
        name="Траєкторія",
    ))

    # Маркери старту/фінішу
    if show_markers:
        fig.add_trace(go.Scatter3d(
            x=[east[0]], y=[north[0]], z=[up[0]],
            mode="markers+text",
            marker=dict(size=8, color="green", symbol="diamond"),
            text=["START"],
            textposition="top center",
            name="Старт",
            hoverinfo="skip",
        ))
        fig.add_trace(go.Scatter3d(
            x=[east[-1]], y=[north[-1]], z=[up[-1]],
            mode="markers+text",
            marker=dict(size=8, color="red", symbol="diamond"),
            text=["END"],
            textposition="top center",
            name="Фініш",
            hoverinfo="skip",
        ))

    z_label = "Alt AMSL (м)" if alt_mode == "amsl" else "Up (м)"
    fig.update_layout(
        title="",
        scene=dict(
            xaxis=dict(title="East (м)", showspikes=False),
            yaxis=dict(title="North (м)", showspikes=False),
            zaxis=dict(title=z_label, showspikes=False),
            aspectmode="data",
        ),
        margin=dict(l=0, r=0, t=40, b=0),
        height=chart_height,
        legend=dict(
            orientation="h",
            yanchor="bottom",
            y=1.0,
            xanchor="left",
            x=0,
            font=dict(size=11),
            bgcolor="rgba(0,0,0,0)",
        ),
    )

    return fig


def create_speed_profile(gps_df: pd.DataFrame, speed_unit: str = "kmh") -> go.Figure:
    """Створює графік профілю швидкості та висоти від часу."""
    if gps_df.empty:
        return go.Figure()

    factor = _SPEED_FACTOR[speed_unit]
    unit_label = _SPEED_LABEL[speed_unit]

    t = gps_df["time_s"].values - gps_df["time_s"].values[0]
    h_speed = gps_df["speed"].values * factor
    v_speed = gps_df["vz"].abs().values * factor
    alt_gain = gps_df["alt"].values - gps_df["alt"].values[0]

    # Адаптивний діапазон осі Y — базується на горизонтальній швидкості,
    # але з мінімум 5 м/с (18 км/год) щоб vz спайки не заповнювали весь графік
    # коли дрон майже нерухомий (h_speed≈0). vz-аномалії що перевищують ymax
    # все одно показуються як бари, але лише реальні викиди.
    speed_ymax = max(_robust_max(h_speed, 97), 5.0 * factor) * 1.15

    fig = go.Figure()

    # Hover з реальними значеннями
    fig.add_trace(go.Scatter(
        x=t, y=h_speed,
        name=f"Горизонтальна швидкість ({unit_label})",
        line=dict(color="blue"),
        hovertemplate=(
            "Час: %{x:.1f} с<br>"
            f"Гориз. швидкість: %{{y:.1f}} {unit_label}"
            "<extra></extra>"
        ),
    ))
    fig.add_trace(go.Scatter(
        x=t, y=v_speed,
        name=f"|Вертикальна швидкість| ({unit_label})",
        line=dict(color="orange"),
        hovertemplate=(
            "Час: %{x:.1f} с<br>"
            f"|Верт. швидкість|: %{{y:.1f}} {unit_label}"
            "<extra></extra>"
        ),
    ))
    fig.add_trace(go.Scatter(
        x=t, y=alt_gain,
        name="Набір висоти (м)",
        yaxis="y2",
        line=dict(color="green", dash="dash"),
        hovertemplate=(
            "Час: %{x:.1f} с<br>"
            "Висота відносно старту: %{y:.1f} м"
            "<extra></extra>"
        ),
    ))

    fig.update_layout(
        title="Профіль швидкості та висоти",
        xaxis_title="Час (с)",
        yaxis=dict(
            title=f"Швидкість ({unit_label})",
            range=[0, speed_ymax],
        ),
        yaxis2=dict(title="Висота (м)", overlaying="y", side="right"),
        height=400,
        legend=dict(x=0, y=1.15, orientation="h"),
    )

    return fig


def create_imu_comparison(
    gps_df: pd.DataFrame,
    imu_velocities: pd.DataFrame,
    speed_unit: str = "kmh",
) -> go.Figure:
    """
    Порівняння GPS-швидкості з IMU-інтегрованою швидкістю.
    Демонструє дрейф при подвійному інтегруванні.
    """
    if gps_df.empty or imu_velocities.empty:
        return go.Figure()

    factor = _SPEED_FACTOR[speed_unit]
    unit_label = _SPEED_LABEL[speed_unit]

    fig = go.Figure()

    t_gps = gps_df["time_s"].values - gps_df["time_s"].values[0]
    fig.add_trace(go.Scatter(
        x=t_gps, y=gps_df["speed"].values * factor,
        name=f"GPS швидкість ({unit_label})",
        line=dict(color="blue", width=2),
        hovertemplate=(
            "Час: %{x:.1f} с<br>"
            f"GPS швидкість: %{{y:.1f}} {unit_label}"
            "<extra></extra>"
        ),
    ))

    t_imu = imu_velocities["time_s"].values - imu_velocities["time_s"].values[0]
    imu_speed = np.sqrt(
        imu_velocities["vel_x"] ** 2
        + imu_velocities["vel_y"] ** 2
        + imu_velocities["vel_z"] ** 2
    ) * factor
    fig.add_trace(go.Scatter(
        x=t_imu, y=imu_speed.values,
        name=f"IMU інтегрована швидкість (з дрейфом) ({unit_label})",
        line=dict(color="red", width=1, dash="dot"),
        hovertemplate=(
            "Час: %{x:.1f} с<br>"
            f"IMU швидкість: %{{y:.1f}} {unit_label}"
            "<extra></extra>"
        ),
    ))

    fig.update_layout(
        title="Порівняння GPS vs IMU швидкості (демонстрація дрейфу інтегрування)",
        xaxis_title="Час (с)",
        yaxis_title=f"Швидкість ({unit_label})",
        height=400,
    )

    return fig
