"""
Модуль парсингу бінарних лог-файлів Ardupilot (.BIN).

Використовує pymavlink для зчитування MAVLink-повідомлень від датчиків GPS та IMU,
формує структуровані pandas DataFrame для подальшого аналізу.

Підтримує fallback-джерела:
  - GPS → AHR2 (EKF-estimated position) якщо GPS відсутній або без фіксації
  - IMU → ACC + GYR (окремі повідомлення) якщо комбінований IMU відсутній
"""

import pandas as pd
from pymavlink import mavutil


def parse_bin_file(filepath: str) -> dict[str, pd.DataFrame]:
    """
    Парсить бінарний лог-файл Ardupilot та повертає DataFrame для GPS та IMU.

    Parameters
    ----------
    filepath : str
        Шлях до .BIN файлу.

    Returns
    -------
    dict с ключами:
        'gps' : pd.DataFrame
            Колонки: time_s, lat, lng, alt, speed, vz, course, hdop, nsats, status
        'gps_raw' : pd.DataFrame
            Нефільтрований GPS
        'imu' : pd.DataFrame
            Колонки: time_s, acc_x, acc_y, acc_z, gyr_x, gyr_y, gyr_z
        'info' : dict
            Метаінформація: частоти семплювання, одиниці вимірювань, кількість повідомлень.
    """
    mlog = mavutil.mavlink_connection(filepath, dialect="ardupilotmega")

    gps_records = []
    ahr2_records = []
    imu_records = []
    acc_records = []
    gyr_records = []
    bat_records = []

    msg_types = ["GPS", "IMU", "AHR2", "ACC", "GYR", "BAT"]

    while True:
        msg = mlog.recv_match(type=msg_types, blocking=False)
        if msg is None:
            break

        msg_type = msg.get_type()
        try:
            data = msg.to_dict()
        except Exception:
            continue

        if msg_type == "GPS":
            gps_records.append({
                "time_us": data["TimeUS"],
                "instance": data.get("I", 0),
                "status": data.get("Status", 0),
                "lat": data["Lat"],
                "lng": data["Lng"],
                "alt": data["Alt"],
                "speed": data["Spd"],
                "vz": data["VZ"],
                "course": data["GCrs"],
                "hdop": data.get("HDop", 0),
                "nsats": data.get("NSats", 0),
            })
        elif msg_type == "AHR2":
            ahr2_records.append({
                "time_us": data["TimeUS"],
                "lat": data["Lat"],
                "lng": data["Lng"],
                "alt": data["Alt"],
                "roll": data["Roll"],
                "pitch": data["Pitch"],
                "yaw": data["Yaw"],
            })
        elif msg_type == "IMU":
            if data.get("I", 0) != 0:
                continue
            imu_records.append({
                "time_us": data["TimeUS"],
                "acc_x": data["AccX"],
                "acc_y": data["AccY"],
                "acc_z": data["AccZ"],
                "gyr_x": data["GyrX"],
                "gyr_y": data["GyrY"],
                "gyr_z": data["GyrZ"],
            })
        elif msg_type == "ACC":
            if data.get("I", 0) != 0:
                continue
            acc_records.append({
                "time_us": data["TimeUS"],
                "acc_x": data["AccX"],
                "acc_y": data["AccY"],
                "acc_z": data["AccZ"],
            })
        elif msg_type == "GYR":
            if data.get("I", 0) != 0:
                continue
            gyr_records.append({
                "time_us": data["TimeUS"],
                "gyr_x": data["GyrX"],
                "gyr_y": data["GyrY"],
                "gyr_z": data["GyrZ"],
            })
        elif msg_type == "BAT":
            if data.get("Inst", 0) != 0:
                continue
            bat_records.append({
                "time_us": data["TimeUS"],
                "voltage": data.get("Volt", 0.0),
                "current": data.get("Curr", 0.0),
            })

    # --- Побудова GPS DataFrame ---
    gps_df = pd.DataFrame(gps_records)
    gps_source = "GPS"

    # Fallback: якщо GPS порожній або всі записи без фіксації — використовуємо AHR2
    use_ahr2 = False
    if gps_df.empty:
        use_ahr2 = True
    elif not gps_df.empty:
        valid_gps = gps_df[gps_df["status"] >= 3]
        if valid_gps.empty:
            # Перевіряємо чи є хоч якісь ненульові координати
            has_coords = gps_df[(gps_df["lat"] != 0) | (gps_df["lng"] != 0)]
            if has_coords.empty:
                use_ahr2 = True

    if use_ahr2 and ahr2_records:
        ahr2_df = pd.DataFrame(ahr2_records)
        # Фільтруємо нульові координати
        ahr2_valid = ahr2_df[(ahr2_df["lat"] != 0) | (ahr2_df["lng"] != 0)]
        if not ahr2_valid.empty:
            # Конвертуємо AHR2 у GPS-подібний формат
            gps_df = pd.DataFrame({
                "time_us": ahr2_valid["time_us"],
                "instance": 0,
                "status": 6,  # AHR2 — EKF estimated, маркуємо як валідний
                "lat": ahr2_valid["lat"],
                "lng": ahr2_valid["lng"],
                "alt": ahr2_valid["alt"],
                "speed": 0.0,  # AHR2 не містить швидкості — обчислимо нижче
                "vz": 0.0,
                "course": ahr2_valid["yaw"],
                "hdop": 0.0,
                "nsats": 0,
            })
            # Обчислюємо швидкість з координат AHR2
            gps_df = _compute_speed_from_positions(gps_df)
            gps_source = "AHR2"

    # --- Побудова IMU DataFrame ---
    imu_df = pd.DataFrame(imu_records)

    # Fallback: якщо IMU порожній — зливаємо ACC + GYR
    if imu_df.empty and acc_records and gyr_records:
        imu_df = _merge_acc_gyr(acc_records, gyr_records)

    # --- Побудова BAT DataFrame ---
    bat_df = pd.DataFrame(bat_records) if bat_records else pd.DataFrame()

    # --- Нормалізація часу ---
    all_times = []
    if not gps_df.empty:
        all_times.append(gps_df["time_us"].iloc[0])
    if not imu_df.empty:
        all_times.append(imu_df["time_us"].iloc[0])

    if all_times:
        t0 = min(all_times)
        if not gps_df.empty:
            gps_df["time_s"] = (gps_df["time_us"] - t0) / 1e6
        if not imu_df.empty:
            imu_df["time_s"] = (imu_df["time_us"] - t0) / 1e6
        if not bat_df.empty:
            bat_df["time_s"] = (bat_df["time_us"] - t0) / 1e6

    # Фільтрація GPS з 3D-фіксацією (Status >= 3)
    if not gps_df.empty:
        gps_df_filtered = gps_df[gps_df["status"] >= 3].copy()
        if gps_df_filtered.empty:
            gps_df_filtered = gps_df[gps_df["status"] >= 1].copy()
    else:
        gps_df_filtered = gps_df

    # Формування метаінформації
    info = _compute_info(gps_df, imu_df)
    info["gps_source"] = gps_source

    return {
        "gps": gps_df_filtered.reset_index(drop=True),
        "gps_raw": gps_df,
        "imu": imu_df.reset_index(drop=True),
        "bat": bat_df.reset_index(drop=True) if not bat_df.empty else bat_df,
        "info": info,
    }


def _compute_speed_from_positions(gps_df: pd.DataFrame) -> pd.DataFrame:
    """Обчислює швидкість та вертикальну швидкість з послідовних координат."""
    from src.coordinates import haversine

    if len(gps_df) < 2:
        return gps_df

    gps_df = gps_df.copy()
    speeds = [0.0]
    vzs = [0.0]

    lats = gps_df["lat"].values
    lngs = gps_df["lng"].values
    alts = gps_df["alt"].values
    times = gps_df["time_us"].values

    # Згладжуємо висоту медіаною вікном 3 щоб прибрати стрибки EKF конвергенції
    import pandas as _pd
    alts_smooth = _pd.Series(alts).rolling(window=3, center=True, min_periods=1).median().values

    for i in range(1, len(gps_df)):
        dt = (times[i] - times[i - 1]) / 1e6
        if dt <= 0:
            speeds.append(speeds[-1])
            vzs.append(0.0)
            continue
        dist = haversine(lats[i - 1], lngs[i - 1], lats[i], lngs[i])
        speeds.append(dist / dt)
        vzs.append((alts_smooth[i] - alts_smooth[i - 1]) / dt)

    gps_df["speed"] = speeds
    gps_df["vz"] = vzs
    return gps_df


def _merge_acc_gyr(
    acc_records: list[dict], gyr_records: list[dict]
) -> pd.DataFrame:
    """Зливає окремі ACC та GYR повідомлення в єдиний IMU DataFrame."""
    acc_df = pd.DataFrame(acc_records).set_index("time_us").sort_index()
    gyr_df = pd.DataFrame(gyr_records).set_index("time_us").sort_index()

    # merge_asof для з'єднання по найближчому часу
    acc_df = acc_df.reset_index()
    gyr_df = gyr_df.reset_index()

    merged = pd.merge_asof(
        acc_df.sort_values("time_us"),
        gyr_df.sort_values("time_us"),
        on="time_us",
        direction="nearest",
        tolerance=5000,  # 5ms толерантність
    )

    merged = merged.dropna(subset=["gyr_x", "gyr_y", "gyr_z"])
    return merged.reset_index(drop=True)


def _compute_info(gps_df: pd.DataFrame, imu_df: pd.DataFrame) -> dict:
    """Обчислює метаінформацію про частоти семплювання та одиниці."""
    info = {
        "gps_count": len(gps_df),
        "imu_count": len(imu_df),
        "units": {
            "gps": {
                "lat": "degrees (WGS-84)",
                "lng": "degrees (WGS-84)",
                "alt": "meters (AMSL)",
                "speed": "m/s (ground speed)",
                "vz": "m/s (vertical velocity, down positive)",
                "course": "degrees",
            },
            "imu": {
                "acc_x/y/z": "m/s^2 (body frame)",
                "gyr_x/y/z": "rad/s (body frame)",
            },
        },
    }

    if not gps_df.empty and len(gps_df) > 1 and "time_s" in gps_df.columns:
        dt_gps = gps_df["time_s"].diff().dropna()
        info["gps_sampling_hz"] = round(1.0 / dt_gps.mean(), 1)
        info["gps_duration_s"] = round(gps_df["time_s"].iloc[-1] - gps_df["time_s"].iloc[0], 2)

    if not imu_df.empty and len(imu_df) > 1 and "time_s" in imu_df.columns:
        dt_imu = imu_df["time_s"].diff().dropna()
        info["imu_sampling_hz"] = round(1.0 / dt_imu.mean(), 1)
        info["imu_duration_s"] = round(imu_df["time_s"].iloc[-1] - imu_df["time_s"].iloc[0], 2)

    return info
