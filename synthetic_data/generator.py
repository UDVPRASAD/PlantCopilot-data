"""
PlantCopilot Synthetic Data Generator — Core Engine
=====================================================
Generates realistic industrial plant sensor data grounded in:
  1. Knowledge Graph equipment definitions & operating envelopes
  2. Physics rules for sensor correlations
  3. Failure mode signatures from FMEA
  4. Cascade failure propagation chains

OUTPUT DATASETS:
  - Time-series sensor data (normal + degradation + failure)
  - Labeled anomaly datasets for ML training
  - Failure event sequences for RCA training
  - Maintenance log synthetic records
  - Operating context annotations

PHYSICS-BASED GENERATION:
  Unlike random noise generators, this engine:
  - Correlates sensors using thermodynamic & mechanical equations
  - Models degradation as gradual parameter drift (not random walks)
  - Generates failure signatures that match real sensor patterns
  - Respects operating envelopes and control loop dynamics
  - Simulates cascade failures across connected equipment
"""

import json
import math
import random
import csv
import uuid
from pathlib import Path
from datetime import datetime, timedelta
from dataclasses import dataclass, field, asdict
from typing import Optional
from enum import Enum
from collections import defaultdict

try:
    import numpy as np
    HAS_NUMPY = True
except ImportError:
    HAS_NUMPY = False

try:
    from loguru import logger
except ImportError:
    import logging
    logger = logging.getLogger(__name__)

from knowledge_graph.ontology import (
    Equipment, Instrument, EquipmentClass, EquipmentSubclass,
    MeasurementType, FailureMode, FailureMechanism,
    OperatingState, CriticalityRating, FluidType,
    EquipmentFailureMode, OperatingEnvelope, PhysicsRule,
    FailureCascadeRule, ProcessConnection,
)
from knowledge_graph.builder import PlantKnowledgeGraph, OilGasDomainBuilder


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
#  CONFIGURATION
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

class ScenarioType(str, Enum):
    NORMAL = "normal"
    DEGRADATION = "degradation"
    FAILURE = "failure"
    CASCADE_FAILURE = "cascade_failure"
    STARTUP = "startup"
    SHUTDOWN = "shutdown"
    LOAD_CHANGE = "load_change"
    PROCESS_UPSET = "process_upset"


@dataclass
class GeneratorConfig:
    """Configuration for synthetic data generation."""
    # Time settings
    start_time: datetime = field(default_factory=lambda: datetime(2025, 1, 1))
    sample_interval_seconds: int = 10          # Sensor scan rate
    duration_hours: float = 24.0               # Total generation duration

    # Normal operating noise
    base_noise_pct: float = 0.015              # ±1.5% of range as Gaussian noise
    drift_amplitude_pct: float = 0.008         # Slow sinusoidal drift amplitude
    drift_period_hours: float = 6.0            # Drift cycle period
    diurnal_amplitude_pct: float = 0.012       # Day/night cycle effect
    process_disturbance_probability: float = 0.002  # Random small disturbances

    # Degradation settings
    degradation_start_pct: float = 0.3         # When in timeline degradation starts (30%)
    degradation_rate: float = 0.0001           # Rate of parameter drift per sample
    degradation_noise_increase: float = 1.5    # Noise multiplier during degradation

    # Failure settings
    failure_onset_pct: float = 0.85            # When failure actually occurs (85% of timeline)
    failure_severity: float = 0.8              # How severe the failure signature is (0-1)
    pre_failure_lead_time_hours: float = 4.0   # Detectable anomaly before failure
    post_failure_duration_hours: float = 2.0   # Data after failure event

    # Cascade settings
    cascade_delay_samples: int = 30            # Samples delay between cascade stages
    cascade_attenuation: float = 0.7           # Signal reduction per cascade hop

    # Data quality
    missing_data_probability: float = 0.001    # Random NaN/missing values
    spike_probability: float = 0.0005          # Random instrument spikes
    stuck_signal_probability: float = 0.0002   # Frozen sensor readings

    # Output
    output_format: str = "csv"                 # csv, parquet, json
    include_labels: bool = True                # Include anomaly/failure labels
    include_context: bool = True               # Include operating context annotations

    seed: Optional[int] = None                 # Random seed for reproducibility


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
#  SIGNAL GENERATORS — Physics-based sensor value generation
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

class SignalGenerator:
    """Generates physics-realistic sensor signals."""

    def __init__(self, config: GeneratorConfig):
        self.config = config
        self.rng = random.Random(config.seed)
        if HAS_NUMPY:
            self.np_rng = np.random.RandomState(config.seed)

    def normal_signal(self, base_value: float, range_min: float, range_max: float,
                      sample_index: int, total_samples: int) -> float:
        """Generate a normal operating signal with realistic noise patterns."""
        value = base_value
        full_range = range_max - range_min
        cfg = self.config

        # 1. Gaussian measurement noise
        noise = self.rng.gauss(0, full_range * cfg.base_noise_pct)
        value += noise

        # 2. Slow process drift (sinusoidal)
        t_hours = sample_index * cfg.sample_interval_seconds / 3600
        drift = full_range * cfg.drift_amplitude_pct * math.sin(2 * math.pi * t_hours / cfg.drift_period_hours)
        value += drift

        # 3. Diurnal cycle (day/night effect on temperatures, ambient-influenced)
        diurnal = full_range * cfg.diurnal_amplitude_pct * math.sin(2 * math.pi * t_hours / 24 - math.pi / 3)
        value += diurnal

        # 4. Random small process disturbances
        if self.rng.random() < cfg.process_disturbance_probability:
            value += self.rng.gauss(0, full_range * 0.03)

        return max(range_min, min(range_max, value))

    def degradation_signal(self, base_value: float, range_min: float, range_max: float,
                           sample_index: int, total_samples: int,
                           degradation_direction: str = "increasing",
                           degradation_rate: Optional[float] = None) -> float:
        """Generate a signal showing gradual degradation pattern."""
        cfg = self.config
        rate = degradation_rate or cfg.degradation_rate
        full_range = range_max - range_min

        # Start with normal signal
        value = self.normal_signal(base_value, range_min, range_max, sample_index, total_samples)

        # Calculate degradation progress
        deg_start = int(total_samples * cfg.degradation_start_pct)
        if sample_index > deg_start:
            progress = (sample_index - deg_start) / (total_samples - deg_start)

            # Gradual drift (exponential acceleration)
            drift_magnitude = full_range * rate * (math.exp(3 * progress) - 1)
            if degradation_direction == "increasing":
                value += drift_magnitude
            else:
                value -= drift_magnitude

            # Increased noise during degradation
            extra_noise = self.rng.gauss(0, full_range * cfg.base_noise_pct * (cfg.degradation_noise_increase - 1) * progress)
            value += extra_noise

            # Occasional micro-spikes as degradation progresses
            if self.rng.random() < 0.005 * progress:
                spike = full_range * 0.05 * progress * (1 if degradation_direction == "increasing" else -1)
                value += spike

        return max(range_min, min(range_max, value))

    def failure_signal(self, base_value: float, range_min: float, range_max: float,
                       sample_index: int, total_samples: int,
                       failure_mode: FailureMode,
                       failure_signature: dict) -> float:
        """Generate a signal with failure mode signature."""
        cfg = self.config
        full_range = range_max - range_min
        failure_start = int(total_samples * cfg.failure_onset_pct)

        if sample_index < failure_start:
            # Pre-failure: show degradation
            return self.degradation_signal(
                base_value, range_min, range_max,
                sample_index, total_samples,
                failure_signature.get("direction", "increasing"),
            )

        # Failure phase
        progress = min(1.0, (sample_index - failure_start) / max(1, total_samples - failure_start))
        pattern = failure_signature.get("pattern", "step_change")

        if pattern == "sudden_drop_to_zero":
            value = base_value * max(0, 1 - progress * 20)
            value += self.rng.gauss(0, full_range * 0.002) if value > 0 else 0

        elif pattern == "severe_spike":
            spike_mag = full_range * cfg.failure_severity
            value = base_value + spike_mag * math.exp(-progress * 3) * (1 + 0.5 * math.sin(progress * 50))
            value += self.rng.gauss(0, full_range * 0.02)

        elif pattern == "step_change":
            step = full_range * cfg.failure_severity * 0.3
            direction = 1 if failure_signature.get("direction", "increasing") == "increasing" else -1
            value = base_value + direction * step
            value += self.rng.gauss(0, full_range * 0.03)

        elif pattern == "oscillation":
            amplitude = full_range * cfg.failure_severity * 0.15 * (1 + progress)
            freq = 0.5 + progress * 5  # Increasing frequency
            value = base_value + amplitude * math.sin(2 * math.pi * freq * progress)
            value += self.rng.gauss(0, full_range * 0.015)

        elif pattern == "gradual_rise":
            value = base_value + full_range * cfg.failure_severity * 0.4 * progress
            value += self.rng.gauss(0, full_range * cfg.base_noise_pct * 2)

        elif pattern == "stuck_signal":
            value = base_value + full_range * 0.02 * (1 if self.rng.random() > 0.5 else -1)
            # Frozen — near-zero noise
            value += self.rng.gauss(0, full_range * 0.0005)

        elif pattern == "erratic":
            value = base_value + self.rng.gauss(0, full_range * 0.08 * (1 + progress * 2))
            if self.rng.random() < 0.1 * progress:
                value += self.rng.gauss(0, full_range * 0.2)

        elif pattern == "increasing_trend":
            value = self.degradation_signal(
                base_value, range_min, range_max,
                sample_index, total_samples, "increasing",
                degradation_rate=cfg.degradation_rate * 3,
            )

        elif pattern == "decreasing":
            value = self.degradation_signal(
                base_value, range_min, range_max,
                sample_index, total_samples, "decreasing",
                degradation_rate=cfg.degradation_rate * 3,
            )

        else:
            value = self.normal_signal(base_value, range_min, range_max, sample_index, total_samples)

        return max(range_min, min(range_max, round(value, 4)))

    def startup_signal(self, base_value: float, range_min: float, range_max: float,
                       sample_index: int, total_samples: int) -> float:
        """Simulate equipment startup ramp-up."""
        progress = min(1.0, sample_index / (total_samples * 0.15))  # 15% of time is startup
        ramp = 0.3 + 0.7 * (1 - math.exp(-4 * progress))  # Exponential ramp
        value = range_min + (base_value - range_min) * ramp
        value += self.rng.gauss(0, (range_max - range_min) * self.config.base_noise_pct * (2 - progress))
        return max(range_min, min(range_max, value))

    def apply_data_quality_issues(self, value: float, range_min: float, range_max: float) -> Optional[float]:
        """Apply realistic data quality issues."""
        cfg = self.config

        # Missing data
        if self.rng.random() < cfg.missing_data_probability:
            return None

        # Instrument spike
        if self.rng.random() < cfg.spike_probability:
            spike_dir = 1 if self.rng.random() > 0.5 else -1
            return value + spike_dir * (range_max - range_min) * self.rng.uniform(0.3, 0.8)

        # Stuck signal (return exact same value — will be detected downstream)
        if self.rng.random() < cfg.stuck_signal_probability:
            return round(value, 1)  # Reduced precision = stuck

        return value


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
#  PHYSICS CORRELATOR — Enforces sensor relationships
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

class PhysicsCorrelator:
    """
    Enforces physics-based correlations between sensors on the same equipment.
    Ensures generated data is thermodynamically and mechanically consistent.
    """

    def __init__(self, kg: PlantKnowledgeGraph, rng: random.Random):
        self.kg = kg
        self.rng = rng
        self._build_correlation_map()

    def _build_correlation_map(self):
        """Build a map of which sensors are correlated and how."""
        self.correlations = defaultdict(list)

        # Pump correlations (from domain knowledge)
        self.pump_rules = [
            # (input_type, output_type, correlation_fn_name)
            ("flow", "pressure", "pump_curve"),
            ("flow", "current", "pump_power"),
            ("flow", "temperature", "pump_thermal"),
            ("vibration", "temperature", "friction_heat"),
            ("speed", "flow", "pump_affinity"),
            ("speed", "vibration", "speed_vibration"),
        ]

        self.compressor_rules = [
            ("pressure_suction", "temperature_discharge", "polytropic_compression"),
            ("speed", "flow", "compressor_affinity"),
            ("speed", "vibration", "compressor_speed_vibration"),
            ("flow", "pressure_discharge", "compressor_curve"),
            ("vibration", "temperature", "friction_heat"),
        ]

        self.exchanger_rules = [
            ("temperature_hot_in", "temperature_hot_out", "hx_duty"),
            ("temperature_cold_in", "temperature_cold_out", "hx_duty_cold"),
            ("flow", "differential_pressure", "hx_dp"),
        ]

    def apply_correlations(self, equipment: Equipment, readings: dict[str, float],
                            instruments: list[Instrument]) -> dict[str, float]:
        """
        Adjust sensor readings to maintain physics consistency.
        readings: {tag: raw_value}
        Returns: {tag: corrected_value}
        """
        corrected = dict(readings)
        eq_class = equipment.equipment_class

        if eq_class == EquipmentClass.PUMP:
            corrected = self._correlate_pump(equipment, corrected, instruments)
        elif eq_class == EquipmentClass.COMPRESSOR:
            corrected = self._correlate_compressor(equipment, corrected, instruments)
        elif eq_class == EquipmentClass.HEAT_EXCHANGER:
            corrected = self._correlate_exchanger(equipment, corrected, instruments)
        elif eq_class in (EquipmentClass.SEPARATOR, EquipmentClass.DRUM):
            corrected = self._correlate_separator(equipment, corrected, instruments)

        return corrected

    def _find_sensor(self, instruments: list[Instrument], mtype: MeasurementType,
                      tag_contains: str = "") -> Optional[Instrument]:
        for inst in instruments:
            if inst.measurement_type == mtype:
                if not tag_contains or tag_contains.lower() in inst.tag.lower():
                    return inst
        return None

    def _correlate_pump(self, eq: Equipment, readings: dict, instruments: list) -> dict:
        """Apply pump physics: Q-H curve, power = ρgQH/η, bearing temp ~ power."""
        flow_inst = self._find_sensor(instruments, MeasurementType.FLOW)
        press_inst = self._find_sensor(instruments, MeasurementType.PRESSURE)
        current_inst = self._find_sensor(instruments, MeasurementType.CURRENT)
        temp_inst = self._find_sensor(instruments, MeasurementType.TEMPERATURE)
        vib_inst = self._find_sensor(instruments, MeasurementType.VIBRATION)
        speed_inst = self._find_sensor(instruments, MeasurementType.SPEED)

        if flow_inst and press_inst and flow_inst.tag in readings and press_inst.tag in readings:
            # Pump curve: higher flow → lower discharge pressure
            flow_pct = (readings[flow_inst.tag] - flow_inst.range_min) / (flow_inst.range_max - flow_inst.range_min + 0.01)
            # Quadratic pump curve relationship
            head_factor = 1.0 - 0.35 * flow_pct ** 2
            target_press = press_inst.range_min + (press_inst.range_max - press_inst.range_min) * 0.5 * head_factor
            # Blend: 70% physics, 30% raw (allow some deviation)
            readings[press_inst.tag] = 0.7 * target_press + 0.3 * readings[press_inst.tag]

        if flow_inst and current_inst and flow_inst.tag in readings and current_inst.tag in readings:
            # Power ~ flow × head
            flow_pct = (readings[flow_inst.tag] - flow_inst.range_min) / (flow_inst.range_max - flow_inst.range_min + 0.01)
            power_factor = 0.3 + 0.6 * flow_pct  # No-load current + load-proportional
            target_current = current_inst.range_min + (current_inst.range_max - current_inst.range_min) * power_factor
            readings[current_inst.tag] = 0.7 * target_current + 0.3 * readings[current_inst.tag]

        if current_inst and temp_inst and current_inst.tag in readings and temp_inst.tag in readings:
            # Temperature ~ current² (I²R losses)
            current_pct = (readings[current_inst.tag] - current_inst.range_min) / (current_inst.range_max - current_inst.range_min + 0.01)
            temp_factor = 0.3 + 0.5 * current_pct ** 1.5
            target_temp = temp_inst.range_min + (temp_inst.range_max - temp_inst.range_min) * temp_factor
            readings[temp_inst.tag] = 0.65 * target_temp + 0.35 * readings[temp_inst.tag]

        return readings

    def _correlate_compressor(self, eq: Equipment, readings: dict, instruments: list) -> dict:
        """Apply compressor physics: polytropic compression, affinity laws."""
        suction_p = self._find_sensor(instruments, MeasurementType.PRESSURE, "S")
        discharge_p = self._find_sensor(instruments, MeasurementType.PRESSURE, "D")
        discharge_t = self._find_sensor(instruments, MeasurementType.TEMPERATURE, "D")
        suction_t = self._find_sensor(instruments, MeasurementType.TEMPERATURE, "S")
        speed = self._find_sensor(instruments, MeasurementType.SPEED)
        vib = self._find_sensor(instruments, MeasurementType.VIBRATION)

        # Polytropic: T2 = T1 × (P2/P1)^((k-1)/(k×ηp))
        if (suction_p and discharge_p and suction_t and discharge_t and
            all(i.tag in readings for i in [suction_p, discharge_p, suction_t])):
            T1 = readings[suction_t.tag] + 273.15  # to Kelvin
            P1 = max(0.1, readings[suction_p.tag])
            P2 = max(0.1, readings[discharge_p.tag])
            k = 1.3  # Ratio of specific heats (natural gas ≈ 1.3)
            eta_p = 0.82  # Polytropic efficiency
            exponent = (k - 1) / (k * eta_p)
            T2_calc = T1 * (P2 / P1) ** exponent - 273.15
            if discharge_t.tag in readings:
                readings[discharge_t.tag] = 0.75 * T2_calc + 0.25 * readings[discharge_t.tag]

        # Speed affects vibration baseline
        if speed and vib and speed.tag in readings and vib.tag in readings:
            speed_pct = (readings[speed.tag] - speed.range_min) / (speed.range_max - speed.range_min + 0.01)
            vib_baseline = vib.range_min + (vib.range_max - vib.range_min) * 0.15 * speed_pct ** 1.5
            readings[vib.tag] = max(readings[vib.tag], vib_baseline)

        return readings

    def _correlate_exchanger(self, eq: Equipment, readings: dict, instruments: list) -> dict:
        """Apply heat exchanger energy balance."""
        hot_in = self._find_sensor(instruments, MeasurementType.TEMPERATURE, "HI")
        hot_out = self._find_sensor(instruments, MeasurementType.TEMPERATURE, "HO")
        cold_in = self._find_sensor(instruments, MeasurementType.TEMPERATURE, "CI")
        cold_out = self._find_sensor(instruments, MeasurementType.TEMPERATURE, "CO")

        # Energy balance: Q = m_h × Cp × (T_hi - T_ho) = m_c × Cp × (T_co - T_ci)
        if all(inst and inst.tag in readings for inst in [hot_in, hot_out, cold_in, cold_out]):
            t_hi = readings[hot_in.tag]
            t_ci = readings[cold_in.tag]
            # Assume effectiveness ε ≈ 0.65
            effectiveness = 0.65 + self.rng.gauss(0, 0.03)
            q_max = t_hi - t_ci
            actual_transfer = q_max * effectiveness
            target_hot_out = t_hi - actual_transfer * 0.55
            target_cold_out = t_ci + actual_transfer * 0.45
            readings[hot_out.tag] = 0.7 * target_hot_out + 0.3 * readings[hot_out.tag]
            readings[cold_out.tag] = 0.7 * target_cold_out + 0.3 * readings[cold_out.tag]

        return readings

    def _correlate_separator(self, eq: Equipment, readings: dict, instruments: list) -> dict:
        """Apply separator mass balance."""
        # Level correlates with inlet flow minus outlet flows
        level = self._find_sensor(instruments, MeasurementType.LEVEL)
        flow = self._find_sensor(instruments, MeasurementType.FLOW)

        if level and flow and level.tag in readings and flow.tag in readings:
            flow_pct = (readings[flow.tag] - flow.range_min) / (flow.range_max - flow.range_min + 0.01)
            # Higher flow → level tends to be higher (with control loop maintaining)
            level_bias = flow_pct * 5  # Small influence
            readings[level.tag] = readings[level.tag] + level_bias

        return readings


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
#  SCENARIO GENERATORS — Failure, cascade, startup patterns
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

@dataclass
class ScenarioDefinition:
    """Defines a specific scenario to generate data for."""
    scenario_type: ScenarioType = ScenarioType.NORMAL
    target_equipment_id: str = ""
    failure_mode: Optional[FailureMode] = None
    failure_mechanism: Optional[FailureMechanism] = None
    cascade_chain: list = field(default_factory=list)  # [equipment_ids in cascade order]
    description: str = ""
    severity: float = 0.7  # 0-1
    label: str = "normal"  # For ML training: normal, anomaly, failure, cascade


@dataclass
class GeneratedSample:
    """A single timestamped data point."""
    timestamp: str
    equipment_id: str
    equipment_tag: str
    sensor_tag: str
    measurement_type: str
    value: Optional[float]
    unit: str
    quality: str  # good, suspect, bad, missing
    label: str  # normal, anomaly, pre_failure, failure, cascade, startup, shutdown
    scenario_id: str
    scenario_type: str
    context: str  # Human-readable context annotation


class ScenarioGenerator:
    """Generates complete scenarios from knowledge graph failure mode data."""

    def __init__(self, kg: PlantKnowledgeGraph):
        self.kg = kg

    def generate_normal_scenarios(self, count: int = 5) -> list[ScenarioDefinition]:
        """Generate normal operating scenarios for various equipment."""
        scenarios = []
        equipment_list = list(self.kg.equipment.values())
        for i in range(min(count, len(equipment_list))):
            eq = equipment_list[i % len(equipment_list)]
            scenarios.append(ScenarioDefinition(
                scenario_type=ScenarioType.NORMAL,
                target_equipment_id=eq.id,
                description=f"Normal operation of {eq.tag} ({eq.name})",
                label="normal",
            ))
        return scenarios

    def generate_degradation_scenarios(self) -> list[ScenarioDefinition]:
        """Generate degradation scenarios from FMEA failure modes."""
        scenarios = []
        # Get equipment with known failure modes
        for eq in self.kg.equipment.values():
            fms = self.kg.get_failure_modes_for_class(eq.equipment_class)
            for fm in fms:
                if fm.failure_rate_per_year > 0.03:  # Only common failures
                    scenarios.append(ScenarioDefinition(
                        scenario_type=ScenarioType.DEGRADATION,
                        target_equipment_id=eq.id,
                        failure_mode=fm.failure_mode,
                        failure_mechanism=fm.failure_mechanism,
                        description=f"Gradual {fm.failure_mode.value} on {eq.tag}: {fm.description}",
                        severity=fm.severity / 10.0,
                        label="anomaly",
                    ))
        return scenarios

    def generate_failure_scenarios(self) -> list[ScenarioDefinition]:
        """Generate complete failure event scenarios."""
        scenarios = []
        for eq in self.kg.equipment.values():
            if eq.criticality in (CriticalityRating.CRITICAL, CriticalityRating.HIGH):
                fms = self.kg.get_failure_modes_for_class(eq.equipment_class)
                for fm in fms[:3]:  # Top 3 failure modes per equipment
                    scenarios.append(ScenarioDefinition(
                        scenario_type=ScenarioType.FAILURE,
                        target_equipment_id=eq.id,
                        failure_mode=fm.failure_mode,
                        failure_mechanism=fm.failure_mechanism,
                        description=f"Failure event: {fm.failure_mode.value} on {eq.tag}",
                        severity=fm.severity / 10.0,
                        label="failure",
                    ))
        return scenarios

    def generate_cascade_scenarios(self) -> list[ScenarioDefinition]:
        """Generate multi-equipment cascade failure scenarios."""
        scenarios = []
        for rule in self.kg.cascade_rules.values():
            # Find specific equipment instances for source and target classes
            source_eqs = self.kg.get_equipment_by_class(rule.source_equipment_class)
            target_eqs = self.kg.get_equipment_by_class(rule.target_equipment_class)
            for src in source_eqs[:2]:
                for tgt in target_eqs[:2]:
                    if src.id != tgt.id:
                        scenarios.append(ScenarioDefinition(
                            scenario_type=ScenarioType.CASCADE_FAILURE,
                            target_equipment_id=src.id,
                            failure_mode=rule.source_failure_mode,
                            cascade_chain=[src.id, tgt.id],
                            description=f"Cascade: {src.tag} {rule.source_failure_mode.value} → {tgt.tag} {rule.cascade_effect}",
                            severity=rule.severity_factor,
                            label="cascade",
                        ))
        return scenarios

    def generate_operational_scenarios(self) -> list[ScenarioDefinition]:
        """Generate startup, shutdown, and load change scenarios."""
        scenarios = []
        for eq in self.kg.equipment.values():
            if eq.equipment_class in (EquipmentClass.COMPRESSOR, EquipmentClass.PUMP):
                scenarios.append(ScenarioDefinition(
                    scenario_type=ScenarioType.STARTUP,
                    target_equipment_id=eq.id,
                    description=f"Startup sequence for {eq.tag}",
                    label="startup",
                ))
                scenarios.append(ScenarioDefinition(
                    scenario_type=ScenarioType.LOAD_CHANGE,
                    target_equipment_id=eq.id,
                    description=f"Load change on {eq.tag}: 80% → 50% → 100%",
                    label="normal",
                ))
        return scenarios


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
#  MAIN GENERATOR — Orchestrates everything
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

class SyntheticDataGenerator:
    """
    Main synthetic data generation engine.

    Pipeline:
    1. Build/load Knowledge Graph
    2. Generate scenario definitions from FMEA + cascade rules
    3. For each scenario:
       a. Generate base signals per instrument
       b. Apply physics correlations
       c. Apply failure/degradation signatures
       d. Apply data quality issues
       e. Label each sample
    4. Export datasets
    """

    def __init__(self, kg: PlantKnowledgeGraph, config: Optional[GeneratorConfig] = None):
        self.kg = kg
        self.config = config or GeneratorConfig()
        self.signal_gen = SignalGenerator(self.config)
        self.physics = PhysicsCorrelator(kg, self.signal_gen.rng)
        self.scenario_gen = ScenarioGenerator(kg)
        self.generated_data: list[GeneratedSample] = []

    def generate_all_datasets(self, output_dir: str) -> dict:
        """Generate complete training datasets for all scenario types."""
        output_path = Path(output_dir)
        output_path.mkdir(parents=True, exist_ok=True)

        logger.info("=== PlantCopilot Synthetic Data Generation ===")
        logger.info(f"Knowledge graph: {self.kg.stats()['equipment']} equipment, {self.kg.stats()['instruments']} instruments")

        all_scenarios = []
        stats = {}

        # 1. Normal operation data (bulk — 60% of dataset)
        normal_scenarios = self.scenario_gen.generate_normal_scenarios(count=len(self.kg.equipment))
        for sc in normal_scenarios:
            sc_config = GeneratorConfig(**{**asdict(self.config), "duration_hours": self.config.duration_hours * 2})
            sc_config.seed = hash(sc.target_equipment_id) % (2**31)
        all_scenarios.extend(normal_scenarios)
        stats["normal"] = len(normal_scenarios)

        # 2. Degradation scenarios
        deg_scenarios = self.scenario_gen.generate_degradation_scenarios()
        all_scenarios.extend(deg_scenarios)
        stats["degradation"] = len(deg_scenarios)

        # 3. Failure scenarios
        fail_scenarios = self.scenario_gen.generate_failure_scenarios()
        all_scenarios.extend(fail_scenarios)
        stats["failure"] = len(fail_scenarios)

        # 4. Cascade failure scenarios
        cascade_scenarios = self.scenario_gen.generate_cascade_scenarios()
        all_scenarios.extend(cascade_scenarios)
        stats["cascade"] = len(cascade_scenarios)

        # 5. Operational scenarios
        op_scenarios = self.scenario_gen.generate_operational_scenarios()
        all_scenarios.extend(op_scenarios)
        stats["operational"] = len(op_scenarios)

        logger.info(f"Total scenarios to generate: {len(all_scenarios)}")
        logger.info(f"  Normal: {stats['normal']}, Degradation: {stats['degradation']}, "
                     f"Failure: {stats['failure']}, Cascade: {stats['cascade']}, "
                     f"Operational: {stats['operational']}")

        # Generate data for each scenario
        all_samples = []
        for i, scenario in enumerate(all_scenarios):
            samples = self._generate_scenario_data(scenario, scenario_index=i)
            all_samples.extend(samples)
            if (i + 1) % 20 == 0:
                logger.info(f"  Generated {i+1}/{len(all_scenarios)} scenarios ({len(all_samples):,} samples)")

        logger.info(f"Total samples generated: {len(all_samples):,}")

        # Export
        self._export_timeseries_csv(all_samples, output_path / "sensor_timeseries.csv")
        self._export_labeled_dataset(all_samples, output_path / "labeled_anomaly_dataset.csv")
        self._export_failure_events(all_samples, output_path / "failure_events.csv")
        self._export_scenario_manifest(all_scenarios, output_path / "scenario_manifest.json")
        self._export_training_splits(all_samples, output_path)

        # Export knowledge graph artifacts
        self.kg.export_json(str(output_path / "knowledge_graph.json"))
        self.kg.export_for_llm(str(output_path / "knowledge_base_for_llm.md"))
        self.kg.export_cypher(str(output_path / "knowledge_graph.cypher"))

        final_stats = {
            "total_samples": len(all_samples),
            "total_scenarios": len(all_scenarios),
            "scenario_breakdown": stats,
            "unique_equipment": len(set(s.equipment_id for s in all_samples)),
            "unique_sensors": len(set(s.sensor_tag for s in all_samples)),
            "label_distribution": self._count_labels(all_samples),
            "output_files": [str(f) for f in output_path.glob("*")],
        }

        (output_path / "generation_stats.json").write_text(json.dumps(final_stats, indent=2), encoding="utf-8")
        logger.info(f"Label distribution: {final_stats['label_distribution']}")
        logger.info(f"Datasets exported to: {output_path}")

        return final_stats

    def _generate_scenario_data(self, scenario: ScenarioDefinition,
                                  scenario_index: int) -> list[GeneratedSample]:
        """Generate timeseries data for a single scenario."""
        samples = []
        cfg = self.config
        scenario_id = f"sc_{scenario_index:04d}_{scenario.scenario_type.value}"

        eq = self.kg.equipment.get(scenario.target_equipment_id)
        if not eq:
            return samples

        instruments = self.kg.get_instruments_for_equipment(eq.id)
        if not instruments:
            return samples

        total_samples = int(cfg.duration_hours * 3600 / cfg.sample_interval_seconds)

        # Get failure mode signature if applicable
        failure_signatures = {}
        if scenario.failure_mode:
            fms = self.kg.get_failure_modes_for_class(eq.equipment_class)
            for fm in fms:
                if fm.failure_mode == scenario.failure_mode:
                    for indicator in fm.sensor_indicators:
                        mtype = indicator.get("measurement", "")
                        pattern = indicator.get("pattern", "increasing_trend")
                        for inst in instruments:
                            if inst.measurement_type.value == mtype or mtype in inst.tag.lower():
                                failure_signatures[inst.tag] = {
                                    "pattern": pattern,
                                    "direction": "increasing" if "increas" in pattern or "rise" in pattern or "spike" in pattern else "decreasing",
                                    "lead_time_hours": indicator.get("lead_time_hours", 4),
                                }

        # Generate timestamped readings
        for idx in range(total_samples):
            timestamp = cfg.start_time + timedelta(seconds=idx * cfg.sample_interval_seconds)
            raw_readings = {}
            labels = {}

            for inst in instruments:
                base_val = self._get_base_value(eq, inst)

                if scenario.scenario_type == ScenarioType.NORMAL:
                    value = self.signal_gen.normal_signal(
                        base_val, inst.range_min, inst.range_max, idx, total_samples)
                    label = "normal"

                elif scenario.scenario_type == ScenarioType.DEGRADATION:
                    deg_dir = failure_signatures.get(inst.tag, {}).get("direction", "increasing")
                    value = self.signal_gen.degradation_signal(
                        base_val, inst.range_min, inst.range_max, idx, total_samples, deg_dir)
                    deg_start = int(total_samples * cfg.degradation_start_pct)
                    label = "anomaly" if idx > deg_start else "normal"

                elif scenario.scenario_type == ScenarioType.FAILURE:
                    sig = failure_signatures.get(inst.tag, {"pattern": "step_change", "direction": "increasing"})
                    value = self.signal_gen.failure_signal(
                        base_val, inst.range_min, inst.range_max, idx, total_samples,
                        scenario.failure_mode, sig)
                    failure_start = int(total_samples * cfg.failure_onset_pct)
                    pre_failure_start = failure_start - int(cfg.pre_failure_lead_time_hours * 3600 / cfg.sample_interval_seconds)
                    if idx >= failure_start:
                        label = "failure"
                    elif idx >= pre_failure_start:
                        label = "pre_failure"
                    elif idx >= int(total_samples * cfg.degradation_start_pct):
                        label = "anomaly"
                    else:
                        label = "normal"

                elif scenario.scenario_type == ScenarioType.CASCADE_FAILURE:
                    # Generate failure for primary, then delayed effect on secondary
                    sig = failure_signatures.get(inst.tag, {"pattern": "step_change", "direction": "increasing"})
                    value = self.signal_gen.failure_signal(
                        base_val, inst.range_min, inst.range_max, idx, total_samples,
                        scenario.failure_mode or FailureMode.BREAKDOWN, sig)
                    failure_start = int(total_samples * cfg.failure_onset_pct)
                    label = "cascade" if idx >= failure_start else ("anomaly" if idx >= int(total_samples * 0.6) else "normal")

                elif scenario.scenario_type == ScenarioType.STARTUP:
                    value = self.signal_gen.startup_signal(
                        base_val, inst.range_min, inst.range_max, idx, total_samples)
                    label = "startup" if idx < int(total_samples * 0.15) else "normal"

                elif scenario.scenario_type == ScenarioType.LOAD_CHANGE:
                    progress = idx / total_samples
                    if progress < 0.33:
                        load_factor = 0.8
                    elif progress < 0.66:
                        load_factor = 0.5
                    else:
                        load_factor = 1.0
                    adjusted_base = inst.range_min + (base_val - inst.range_min) * load_factor
                    value = self.signal_gen.normal_signal(
                        adjusted_base, inst.range_min, inst.range_max, idx, total_samples)
                    label = "normal"

                else:
                    value = self.signal_gen.normal_signal(
                        base_val, inst.range_min, inst.range_max, idx, total_samples)
                    label = "normal"

                raw_readings[inst.tag] = value
                labels[inst.tag] = label

            # Apply physics correlations
            corrected_readings = self.physics.apply_correlations(eq, raw_readings, instruments)

            # Apply data quality issues and create samples
            for inst in instruments:
                value = corrected_readings.get(inst.tag)
                if value is None:
                    continue

                final_value = self.signal_gen.apply_data_quality_issues(
                    value, inst.range_min, inst.range_max)

                quality = "good"
                if final_value is None:
                    quality = "missing"
                elif final_value > inst.range_max * 1.1 or final_value < inst.range_min * 0.9:
                    quality = "suspect"

                # Determine alarm status
                context_parts = [scenario.description]
                if final_value is not None:
                    if inst.high_high_alarm and final_value >= inst.high_high_alarm:
                        context_parts.append(f"ALARM: {inst.tag} HH={inst.high_high_alarm}")
                    elif inst.high_alarm and final_value >= inst.high_alarm:
                        context_parts.append(f"ALARM: {inst.tag} H={inst.high_alarm}")
                    elif inst.low_low_alarm and final_value <= inst.low_low_alarm:
                        context_parts.append(f"ALARM: {inst.tag} LL={inst.low_low_alarm}")
                    elif inst.low_alarm and final_value <= inst.low_alarm:
                        context_parts.append(f"ALARM: {inst.tag} L={inst.low_alarm}")

                sample = GeneratedSample(
                    timestamp=timestamp.isoformat(),
                    equipment_id=eq.id,
                    equipment_tag=eq.tag,
                    sensor_tag=inst.tag,
                    measurement_type=inst.measurement_type.value,
                    value=round(final_value, 4) if final_value is not None else None,
                    unit=inst.unit,
                    quality=quality,
                    label=labels.get(inst.tag, "normal"),
                    scenario_id=scenario_id,
                    scenario_type=scenario.scenario_type.value,
                    context="; ".join(context_parts),
                )
                samples.append(sample)

        # Generate cascade secondary equipment data
        if scenario.scenario_type == ScenarioType.CASCADE_FAILURE and len(scenario.cascade_chain) > 1:
            for cascade_idx, cascade_eq_id in enumerate(scenario.cascade_chain[1:], 1):
                cascade_eq = self.kg.equipment.get(cascade_eq_id)
                if not cascade_eq:
                    continue
                cascade_instruments = self.kg.get_instruments_for_equipment(cascade_eq_id)
                delay_samples = cfg.cascade_delay_samples * cascade_idx
                attenuation = cfg.cascade_attenuation ** cascade_idx

                for idx in range(total_samples):
                    timestamp = cfg.start_time + timedelta(seconds=idx * cfg.sample_interval_seconds)
                    for inst in cascade_instruments:
                        base_val = self._get_base_value(cascade_eq, inst)
                        effective_idx = max(0, idx - delay_samples)

                        if effective_idx < int(total_samples * cfg.failure_onset_pct):
                            value = self.signal_gen.normal_signal(
                                base_val, inst.range_min, inst.range_max, idx, total_samples)
                            label = "normal"
                        else:
                            sig = {"pattern": "gradual_rise", "direction": "increasing"}
                            value = self.signal_gen.failure_signal(
                                base_val, inst.range_min, inst.range_max,
                                effective_idx, total_samples,
                                scenario.failure_mode or FailureMode.BREAKDOWN, sig)
                            # Attenuate cascade effect
                            value = base_val + (value - base_val) * attenuation
                            label = "cascade"

                        sample = GeneratedSample(
                            timestamp=timestamp.isoformat(),
                            equipment_id=cascade_eq.id,
                            equipment_tag=cascade_eq.tag,
                            sensor_tag=inst.tag,
                            measurement_type=inst.measurement_type.value,
                            value=round(value, 4),
                            unit=inst.unit,
                            quality="good",
                            label=label,
                            scenario_id=scenario_id,
                            scenario_type="cascade_secondary",
                            context=f"Cascade effect from {eq.tag} → {cascade_eq.tag} (delay: {delay_samples * cfg.sample_interval_seconds}s, attenuation: {attenuation:.1%})",
                        )
                        samples.append(sample)

        return samples

    def _get_base_value(self, eq: Equipment, inst: Instrument) -> float:
        """Get the normal operating base value for a sensor."""
        # Try to get from operating envelope
        for env in self.kg.operating_envelopes.values():
            if env.equipment_id == eq.id and env.state == "normal":
                for param in env.parameters:
                    if inst.measurement_type.value in param.get("measurement_type", ""):
                        return param.get("optimal", (inst.range_min + inst.range_max) / 2)

        # Fallback: midpoint of alarm range or 50% of full range
        if inst.high_alarm and inst.low_alarm:
            return (inst.high_alarm + inst.low_alarm) / 2
        elif inst.high_alarm:
            return inst.range_min + (inst.high_alarm - inst.range_min) * 0.6
        elif inst.low_alarm:
            return inst.low_alarm + (inst.range_max - inst.low_alarm) * 0.4
        else:
            return (inst.range_min + inst.range_max) * 0.45

    # ── Export Methods ──────────────────────────────

    def _export_timeseries_csv(self, samples: list[GeneratedSample], filepath: Path):
        """Export raw timeseries data."""
        with open(filepath, "w", newline="", encoding="utf-8") as f:
            writer = csv.writer(f)
            writer.writerow([
                "timestamp", "equipment_id", "equipment_tag", "sensor_tag",
                "measurement_type", "value", "unit", "quality",
                "label", "scenario_id", "scenario_type",
            ])
            for s in samples:
                writer.writerow([
                    s.timestamp, s.equipment_id, s.equipment_tag, s.sensor_tag,
                    s.measurement_type, s.value, s.unit, s.quality,
                    s.label, s.scenario_id, s.scenario_type,
                ])
        logger.info(f"Timeseries CSV: {filepath} ({len(samples):,} rows)")

    def _export_labeled_dataset(self, samples: list[GeneratedSample], filepath: Path):
        """Export labeled dataset for anomaly detection ML training."""
        with open(filepath, "w", newline="", encoding="utf-8") as f:
            writer = csv.writer(f)
            writer.writerow([
                "timestamp", "equipment_tag", "sensor_tag", "value", "unit",
                "label", "is_anomaly", "scenario_type", "context",
            ])
            for s in samples:
                is_anomaly = 1 if s.label in ("anomaly", "pre_failure", "failure", "cascade") else 0
                writer.writerow([
                    s.timestamp, s.equipment_tag, s.sensor_tag, s.value, s.unit,
                    s.label, is_anomaly, s.scenario_type, s.context,
                ])
        logger.info(f"Labeled dataset: {filepath}")

    def _export_failure_events(self, samples: list[GeneratedSample], filepath: Path):
        """Export failure event log for RCA training."""
        events = []
        current_event = None

        for s in sorted(samples, key=lambda x: (x.scenario_id, x.timestamp)):
            if s.label in ("failure", "cascade") and current_event is None:
                current_event = {
                    "event_id": f"evt_{len(events):04d}",
                    "start_time": s.timestamp,
                    "equipment_tag": s.equipment_tag,
                    "scenario_id": s.scenario_id,
                    "scenario_type": s.scenario_type,
                    "sensors_affected": set(),
                    "context": s.context,
                }
            elif s.label in ("failure", "cascade") and current_event:
                current_event["end_time"] = s.timestamp
                current_event["sensors_affected"].add(s.sensor_tag)
            elif s.label not in ("failure", "cascade") and current_event:
                current_event["sensors_affected"] = list(current_event["sensors_affected"])
                events.append(current_event)
                current_event = None

        if current_event:
            current_event["sensors_affected"] = list(current_event.get("sensors_affected", set()))
            events.append(current_event)

        with open(filepath, "w", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(f, fieldnames=[
                "event_id", "start_time", "end_time", "equipment_tag",
                "scenario_id", "scenario_type", "sensors_affected", "context",
            ])
            writer.writeheader()
            for evt in events:
                evt["sensors_affected"] = json.dumps(evt.get("sensors_affected", []))
                writer.writerow(evt)
        logger.info(f"Failure events: {filepath} ({len(events)} events)")

    def _export_scenario_manifest(self, scenarios: list[ScenarioDefinition], filepath: Path):
        """Export scenario manifest for reproducibility."""
        manifest = []
        for i, sc in enumerate(scenarios):
            manifest.append({
                "scenario_id": f"sc_{i:04d}_{sc.scenario_type.value}",
                "type": sc.scenario_type.value,
                "equipment_id": sc.target_equipment_id,
                "failure_mode": sc.failure_mode.value if sc.failure_mode else None,
                "failure_mechanism": sc.failure_mechanism.value if sc.failure_mechanism else None,
                "cascade_chain": sc.cascade_chain,
                "severity": sc.severity,
                "label": sc.label,
                "description": sc.description,
            })
        filepath.write_text(json.dumps(manifest, indent=2), encoding="utf-8")
        logger.info(f"Scenario manifest: {filepath} ({len(manifest)} scenarios)")

    def _export_training_splits(self, samples: list[GeneratedSample], output_path: Path):
        """Export train/val/test splits for ML training."""
        # Group by scenario
        by_scenario = defaultdict(list)
        for s in samples:
            by_scenario[s.scenario_id].append(s)

        scenarios = list(by_scenario.keys())
        random.Random(42).shuffle(scenarios)

        n = len(scenarios)
        train_end = int(n * 0.7)
        val_end = int(n * 0.85)

        splits = {
            "train": scenarios[:train_end],
            "val": scenarios[train_end:val_end],
            "test": scenarios[val_end:],
        }

        for split_name, split_scenarios in splits.items():
            split_samples = []
            for sc_id in split_scenarios:
                split_samples.extend(by_scenario[sc_id])

            filepath = output_path / f"{split_name}_dataset.csv"
            with open(filepath, "w", newline="", encoding="utf-8") as f:
                writer = csv.writer(f)
                writer.writerow([
                    "timestamp", "equipment_tag", "sensor_tag", "value",
                    "measurement_type", "unit", "label", "is_anomaly",
                ])
                for s in split_samples:
                    is_anomaly = 1 if s.label in ("anomaly", "pre_failure", "failure", "cascade") else 0
                    writer.writerow([
                        s.timestamp, s.equipment_tag, s.sensor_tag, s.value,
                        s.measurement_type, s.unit, s.label, is_anomaly,
                    ])
            logger.info(f"  {split_name}: {filepath} ({len(split_samples):,} samples, {len(split_scenarios)} scenarios)")

    def _count_labels(self, samples: list[GeneratedSample]) -> dict:
        counts = defaultdict(int)
        for s in samples:
            counts[s.label] += 1
        return dict(counts)
