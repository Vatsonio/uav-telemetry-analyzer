"""Тести для модуля metrics."""
import numpy as np
import pandas as pd
import pytest
from src.metrics import trapezoidal_integrate, velocity_from_imu, compute_flight_metrics, detect_flight_phases


class TestTrapezoidalIntegrate:
    def test_constant(self):
        # Інтегрування константи 2 по часу 0-5 = 10
        vals = np.full(6, 2.0)
        times = np.arange(6, dtype=float)
        result = trapezoidal_integrate(vals, times)
        assert abs(result[-1] - 10.0) < 1e-10

    def test_linear(self):
        # Інтегрування f(t)=t від 0 до 4 = 8
        times = np.arange(5, dtype=float)
        vals = times.copy()
        result = trapezoidal_integrate(vals, times)
        assert abs(result[-1] - 8.0) < 1e-10

    def test_zero(self):
        vals = np.zeros(10)
        times = np.arange(10, dtype=float)
        result = trapezoidal_integrate(vals, times)
        assert np.allclose(result, 0.0)

    def test_starts_at_zero(self):
        result = trapezoidal_integrate(np.array([1.0, 2.0, 3.0]), np.array([0.0, 1.0, 2.0]))
        assert result[0] == 0.0


class TestFlightMetrics:
    def _make_gps(self, n=50):
        return pd.DataFrame({
            "time_s": np.linspace(0, 10, n),
            "lat": np.linspace(-35.36, -35.37, n),
            "lng": np.full(n, 149.16),
            "alt": np.linspace(100, 120, n),
            "speed": np.full(n, 5.0),
            "vz": np.full(n, -1.0),
            "hdop": np.full(n, 1.2),
            "nsats": np.full(n, 12),
        })

    def _make_imu(self, n=500):
        return pd.DataFrame({
            "time_s": np.linspace(0, 10, n),
            "acc_x": np.random.normal(0, 0.1, n),
            "acc_y": np.random.normal(0, 0.1, n),
            "acc_z": np.random.normal(-9.81, 0.1, n),
        })

    def test_returns_all_keys(self):
        metrics = compute_flight_metrics(self._make_gps(), self._make_imu())
        required = ["max_horizontal_speed", "max_vertical_speed", "max_acceleration",
                     "max_altitude_gain", "total_duration", "total_distance", "imu_velocities"]
        for key in required:
            assert key in metrics

    def test_duration(self):
        metrics = compute_flight_metrics(self._make_gps(), self._make_imu())
        assert abs(metrics["total_duration"] - 10.0) < 0.5

    def test_altitude_gain(self):
        metrics = compute_flight_metrics(self._make_gps(), self._make_imu())
        assert abs(metrics["max_altitude_gain"] - 20.0) < 1.0

    def test_empty_gps(self):
        metrics = compute_flight_metrics(pd.DataFrame(), self._make_imu())
        assert metrics["total_distance"] == 0


class TestFlightPhases:
    def test_flat_flight(self):
        gps = pd.DataFrame({
            "time_s": np.arange(0, 10, 0.1),
            "speed": np.full(100, 0.5),
            "vz": np.full(100, 0.0),
            "alt": np.full(100, 100.0),
        })
        phases = detect_flight_phases(gps)
        assert len(phases) >= 1

    def test_takeoff_and_landing(self):
        n = 200
        t = np.linspace(0, 40, n)
        alt = np.concatenate([
            np.linspace(0, 50, 50),   # takeoff
            np.full(100, 50),          # cruise
            np.linspace(50, 0, 50),    # landing
        ])
        gps = pd.DataFrame({
            "time_s": t,
            "speed": np.full(n, 5.0),
            "alt": alt,
        })
        phases = detect_flight_phases(gps)
        phase_names = [p["phase"] for p in phases]
        assert "takeoff" in phase_names
        assert "landing" in phase_names

    def test_empty(self):
        assert detect_flight_phases(pd.DataFrame()) == []
