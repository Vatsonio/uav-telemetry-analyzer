"""
Модуль перетворення координат та обчислення відстаней.

Реалізовано:
- Haversine formula для обчислення відстані між двома точками на сфері
- Конвертація WGS-84 (lat/lon/alt) -> локальна декартова система ENU (East-North-Up)

Теоретичне обґрунтування:
    WGS-84 — глобальна геодезична система координат, де положення задається
    широтою (lat), довготою (lon) та висотою (alt) на еліпсоїді.

    Для локального аналізу польоту зручніше працювати у метричній системі ENU,
    де осі: East (схід), North (північ), Up (вгору). Точка старту стає початком
    координат (0, 0, 0).

    Haversine formula обчислює відстань по великому колу між двома точками на
    сфері, враховуючи кривизну Землі. Формула стійка до числових похибок навіть
    для малих відстаней (на відміну від формули сферичного закону косинусів).

    Для малих відстаней (<10 км) лінеаризація WGS-84 -> ENU дає похибку < 0.1%,
    що прийнятно для аналізу польотів БПЛА.
"""

import numpy as np


# Середній радіус Землі (м), WGS-84
EARTH_RADIUS = 6_371_000.0


def haversine(lat1: float, lon1: float, lat2: float, lon2: float) -> float:
    """
    Обчислює відстань (м) між двома точками на сфері за формулою Haversine.

    Формула Haversine:
        a = sin²(Δlat/2) + cos(lat1) · cos(lat2) · sin²(Δlon/2)
        c = 2 · atan2(√a, √(1-a))
        d = R · c

    Parameters
    ----------
    lat1, lon1 : float
        Координати першої точки (градуси).
    lat2, lon2 : float
        Координати другої точки (градуси).

    Returns
    -------
    float
        Відстань між точками у метрах.
    """
    lat1_r, lat2_r = np.radians(lat1), np.radians(lat2)
    dlat = np.radians(lat2 - lat1)
    dlon = np.radians(lon2 - lon1)

    a = np.sin(dlat / 2) ** 2 + np.cos(lat1_r) * np.cos(lat2_r) * np.sin(dlon / 2) ** 2
    c = 2 * np.arctan2(np.sqrt(a), np.sqrt(1 - a))

    return EARTH_RADIUS * c


def total_distance_haversine(lats: np.ndarray, lons: np.ndarray) -> float:
    """
    Обчислює загальну пройдену дистанцію (м) як суму haversine-відстаней
    між послідовними GPS-точками.

    Parameters
    ----------
    lats, lons : array-like
        Масиви широт та довгот (градуси).

    Returns
    -------
    float
        Загальна дистанція у метрах.
    """
    lats = np.asarray(lats)
    lons = np.asarray(lons)
    total = 0.0
    for i in range(1, len(lats)):
        total += haversine(lats[i - 1], lons[i - 1], lats[i], lons[i])
    return total


def wgs84_to_enu(
    lat: np.ndarray,
    lon: np.ndarray,
    alt: np.ndarray,
    lat0: float,
    lon0: float,
    alt0: float,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Конвертує координати WGS-84 у локальну систему ENU (East-North-Up).

    Використовується лінеаризована апроксимація (дотична площина), яка є
    достатньо точною для відстаней до ~10 км від точки відліку:
        East  = (lon - lon0) · cos(lat0) · π/180 · R
        North = (lat - lat0) · π/180 · R
        Up    = alt - alt0

    Примітка щодо точнішого підходу:
        Для більших відстаней або вищої точності можна використати проміжну
        конвертацію через ECEF (Earth-Centered, Earth-Fixed):
        1. WGS-84 -> ECEF (з урахуванням еліпсоїдальної форми Землі)
        2. ECEF -> ENU (обертання матрицею, залежною від lat0, lon0)
        Для типових польотів БПЛА (сотні метрів - кілометри) лінеаризація дає
        похибку < 0.01%, тому обрано простіший підхід.

    Parameters
    ----------
    lat, lon, alt : array-like
        Координати у WGS-84 (градуси, градуси, метри AMSL).
    lat0, lon0, alt0 : float
        Координати точки відліку (початок координат ENU).

    Returns
    -------
    east, north, up : np.ndarray
        Координати у метрах відносно точки відліку.
    """
    lat = np.asarray(lat, dtype=np.float64)
    lon = np.asarray(lon, dtype=np.float64)
    alt = np.asarray(alt, dtype=np.float64)

    east = (lon - lon0) * np.cos(np.radians(lat0)) * np.radians(1) * EARTH_RADIUS
    north = (lat - lat0) * np.radians(1) * EARTH_RADIUS
    up = alt - alt0

    return east, north, up
