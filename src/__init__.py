"""
Модулі аналізу телеметрії БПЛА.

Пакет містить:
- parser: парсинг .BIN лог-файлів Ardupilot
- coordinates: перетворення WGS-84 → ENU, Haversine
- metrics: обчислення метрик, трапецієвидне інтегрування, детекція фаз
- visualization: 3D/2D візуалізація, графіки
- ai_report: AI-звіт через Gemini, детекція аномалій
"""

from .parser import parse_bin_file
from .coordinates import haversine, total_distance_haversine, wgs84_to_enu
from .metrics import (
    trapezoidal_integrate,
    velocity_from_imu,
    compute_flight_metrics,
    detect_flight_phases,
)
from .visualization import (
    create_trajectory_figure,
    create_speed_profile,
    create_imu_comparison,
    create_battery_chart,
    create_gps_quality_chart,
    create_acceleration_chart,
    create_2d_map,
)
from .ai_report import generate_flight_report, detect_anomalies

__all__ = [
    "parse_bin_file",
    "haversine",
    "total_distance_haversine",
    "wgs84_to_enu",
    "trapezoidal_integrate",
    "velocity_from_imu",
    "compute_flight_metrics",
    "detect_flight_phases",
    "create_trajectory_figure",
    "create_speed_profile",
    "create_imu_comparison",
    "create_battery_chart",
    "create_gps_quality_chart",
    "create_acceleration_chart",
    "create_2d_map",
    "generate_flight_report",
    "detect_anomalies",
]
