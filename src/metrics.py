"""
Модуль обчислення польотних метрик.

Реалізовано:
- Обчислення ключових кінематичних показників польоту
- Трапецієвидне інтегрування (метод трапецій) для отримання швидкостей з прискорень
- Розрахунок загальної дистанції через haversine

Теоретичне обґрунтування:

    Метод трапецій — чисельний метод інтегрування, що апроксимує площу під
    кривою сумою трапецій. Для функції f(t), дискретизованої у точках t_i:
        ∫f(t)dt ≈ Σ [0.5 · (f(t_i) + f(t_{i+1})) · (t_{i+1} - t_i)]

    Похибка методу: O(h²) локально, O(h) глобально, де h — крок дискретизації.

    Інтегрування прискорень IMU:
        Прискорення (м/с²) -> інтегрування -> швидкість (м/с)
        Швидкість (м/с) -> інтегрування -> переміщення (м)

    ВАЖЛИВО: Подвійне інтегрування IMU накопичує дрейф через:
    1. Зміщення нуля акселерометра (bias) — призводить до лінійного дрейфу
       швидкості та квадратичного дрейфу позиції
    2. Шум датчика — випадковий дрейф (random walk)
    3. Похибка видалення гравітації — потребує точної орієнтації (кватерніони)

    Тому для реальних метрик швидкості використовується GPS (Spd, VZ),
    а інтегрування IMU демонструється як алгоритмічна вимога завдання.

    Орієнтація та кватерніони:
        Прискорення IMU вимірюються у body frame (система координат апарату).
        Для перетворення у NED/ENU frame потрібна матриця обертання, яка
        будується з кватерніонів орієнтації. Кватерніони q = (w, x, y, z) мають
        перевагу над кутами Ейлера (roll, pitch, yaw), оскільки не страждають
        від gimbal lock — ситуації, коли при pitch = ±90° втрачається один
        ступінь свободи, і два кути стають нерозрізненними.
"""

import numpy as np
import pandas as pd

from .coordinates import haversine, total_distance_haversine


def trapezoidal_integrate(values: np.ndarray, times: np.ndarray) -> np.ndarray:
    """
    Трапецієвидне інтегрування масиву значень за часом.

    Реалізація методу трапецій:
        result[0] = 0
        result[i] = result[i-1] + 0.5 · (values[i] + values[i-1]) · (times[i] - times[i-1])

    Parameters
    ----------
    values : np.ndarray
        Масив значень функції (наприклад, прискорення).
    times : np.ndarray
        Масив відповідних часових міток (секунди).

    Returns
    -------
    np.ndarray
        Масив інтегрованих значень (наприклад, швидкості).
    """
    values = np.asarray(values, dtype=np.float64)
    times = np.asarray(times, dtype=np.float64)

    dt = np.diff(times)
    avg_vals = 0.5 * (values[:-1] + values[1:])
    increments = avg_vals * dt

    result = np.zeros(len(values))
    result[1:] = np.cumsum(increments)
    return result


def velocity_from_imu(imu_df: pd.DataFrame) -> pd.DataFrame:
    """
    Обчислює швидкості з прискорень IMU через трапецієвидне інтегрування.

    УВАГА: Результат містить значний дрейф через накопичення похибок.
    AccZ включає гравітацію (~9.81 м/с² вниз у body frame), яку необхідно
    компенсувати. Спрощено віднімаємо g від вертикальної компоненти.

    Parameters
    ----------
    imu_df : pd.DataFrame
        DataFrame з колонками: time_s, acc_x, acc_y, acc_z

    Returns
    -------
    pd.DataFrame
        DataFrame з колонками: time_s, vel_x, vel_y, vel_z (м/с)
    """
    times = imu_df["time_s"].values

    # Компенсація гравітації з AccZ
    # В body frame Ardupilot: AccZ ~ -9.81 коли апарат нерухомий (Z вниз)
    # Тому додаємо 9.81 для компенсації (або віднімаємо, залежно від конвенції)
    acc_z_compensated = imu_df["acc_z"].values + 9.81  # Ardupilot: Z down, gravity ~ -9.81

    vel_x = trapezoidal_integrate(imu_df["acc_x"].values, times)
    vel_y = trapezoidal_integrate(imu_df["acc_y"].values, times)
    vel_z = trapezoidal_integrate(acc_z_compensated, times)

    return pd.DataFrame({
        "time_s": times,
        "vel_x": vel_x,
        "vel_y": vel_y,
        "vel_z": vel_z,
    })


def compute_flight_metrics(gps_df: pd.DataFrame, imu_df: pd.DataFrame) -> dict:
    """
    Обчислює підсумкові метрики польоту.

    Parameters
    ----------
    gps_df : pd.DataFrame
        GPS дані (time_s, lat, lng, alt, speed, vz, ...)
    imu_df : pd.DataFrame
        IMU дані (time_s, acc_x, acc_y, acc_z, ...)

    Returns
    -------
    dict з ключами:
        max_horizontal_speed : float (м/с)
        max_vertical_speed : float (м/с)
        max_acceleration : float (м/с²)
        max_altitude_gain : float (м)
        total_duration : float (с)
        total_distance : float (м)
        imu_velocities : pd.DataFrame (результат інтегрування IMU)
        max_imu_speed : float (м/с) — максимальна швидкість з IMU інтегрування
    """
    metrics = {}

    # --- GPS-based metrics ---
    if not gps_df.empty:
        speed = gps_df["speed"].values
        # Фільтрація викидів: обмежуємо 99-м перцентилем (GPS-глітчі)
        speed_p99 = float(np.percentile(speed[np.isfinite(speed)], 99))
        speed_clean = speed[speed <= speed_p99]

        metrics["max_horizontal_speed"] = round(float(speed_clean.max()), 2)
        metrics["avg_horizontal_speed"] = round(float(speed_clean.mean()), 2)
        metrics["speed_p50"] = round(float(np.percentile(speed_clean, 50)), 2)
        metrics["speed_p75"] = round(float(np.percentile(speed_clean, 75)), 2)
        metrics["speed_p95"] = round(float(np.percentile(speed_clean, 95)), 2)

        vz_abs = gps_df["vz"].abs()
        vz_p99 = float(np.percentile(vz_abs.values[np.isfinite(vz_abs.values)], 99))
        metrics["max_vertical_speed"] = round(float(vz_abs[vz_abs <= vz_p99].max()), 2)

        metrics["max_altitude_gain"] = round(
            float(gps_df["alt"].max() - gps_df["alt"].iloc[0]), 2
        )
        metrics["alt_min"] = round(float(gps_df["alt"].min()), 2)
        metrics["alt_max"] = round(float(gps_df["alt"].max()), 2)
        metrics["alt_mean"] = round(float(gps_df["alt"].mean()), 2)
        metrics["total_duration"] = round(
            float(gps_df["time_s"].iloc[-1] - gps_df["time_s"].iloc[0]), 2
        )
        metrics["total_distance"] = round(
            total_distance_haversine(gps_df["lat"].values, gps_df["lng"].values), 2
        )
        if "hdop" in gps_df.columns:
            metrics["hdop_mean"] = round(float(gps_df["hdop"].mean()), 2)
        if "nsats" in gps_df.columns:
            metrics["nsats_mean"] = round(float(gps_df["nsats"].mean()), 1)
            metrics["nsats_min"] = int(gps_df["nsats"].min())
    else:
        metrics.update({
            "max_horizontal_speed": 0, "avg_horizontal_speed": 0,
            "speed_p50": 0, "speed_p75": 0, "speed_p95": 0,
            "max_vertical_speed": 0, "max_altitude_gain": 0,
            "alt_min": 0, "alt_max": 0, "alt_mean": 0,
            "total_duration": 0, "total_distance": 0,
        })

    # --- IMU-based metrics ---
    if not imu_df.empty:
        # Максимальне прискорення (повний вектор без гравітації)
        acc_magnitude = np.sqrt(
            imu_df["acc_x"] ** 2 + imu_df["acc_y"] ** 2 + imu_df["acc_z"] ** 2
        )
        # Віднімаємо гравітацію для отримання "чистого" прискорення
        net_acc = np.abs(acc_magnitude - 9.81)
        metrics["max_acceleration"] = round(float(np.percentile(net_acc, 99)), 2)
        metrics["acc_p95"] = round(float(np.percentile(net_acc, 95)), 2)

        # Швидкості з інтегрування IMU (трапецієвидним методом)
        imu_vel = velocity_from_imu(imu_df)
        imu_speed = np.sqrt(imu_vel["vel_x"] ** 2 + imu_vel["vel_y"] ** 2 + imu_vel["vel_z"] ** 2)
        metrics["max_imu_speed"] = round(float(imu_speed.max()), 2)
        metrics["imu_velocities"] = imu_vel
    else:
        metrics["max_acceleration"] = 0
        metrics["acc_p95"] = 0
        metrics["max_imu_speed"] = 0
        metrics["imu_velocities"] = pd.DataFrame()

    return metrics
