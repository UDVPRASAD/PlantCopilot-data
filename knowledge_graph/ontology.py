"""
PlantCopilot Knowledge Graph — Ontology & Schema
==================================================
Defines the complete data model for industrial plant knowledge:

ENTITY HIERARCHY:
  Enterprise → Site → Plant → Area → Unit → Equipment → Component → Instrument

RELATIONSHIP TYPES:
  Structural:   contains, installed_on, part_of
  Process:      feeds_to, receives_from, controls, measures
  Causal:       causes, mitigates, triggers, inhibits
  Maintenance:  maintained_by, requires_part, follows_procedure
  Safety:       interlocked_with, trips_on, protected_by

DOMAIN COVERAGE:
  - Process engineering (flow, heat/mass balance)
  - Mechanical (rotating equipment, static equipment)
  - Electrical (power distribution, motor control)
  - Instrumentation (measurement, control loops)
  - Safety (SIL, interlocks, relief systems)
  - Maintenance (preventive, predictive, corrective)
  - Failure analysis (FMEA, fault trees, bow-tie)

This ontology is aligned with:
  - ISO 14224 (Equipment reliability data)
  - ISO 15926 (Industrial data integration)
  - IEC 61511 (Safety instrumented systems)
  - ISA-95 (Enterprise-control integration)
  - ISA-88 (Batch control)
"""

from __future__ import annotations
import uuid
from datetime import datetime
from enum import Enum
from dataclasses import dataclass, field
from typing import Optional


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
#  ENUMERATIONS — Controlled Vocabularies
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

class EquipmentClass(str, Enum):
    """ISO 14224 aligned equipment classification."""
    # Static equipment
    VESSEL = "vessel"
    TANK = "tank"
    COLUMN = "column"
    REACTOR = "reactor"
    HEAT_EXCHANGER = "heat_exchanger"
    BOILER = "boiler"
    FURNACE = "furnace"
    FILTER = "filter"
    SEPARATOR = "separator"
    DRUM = "drum"
    SILO = "silo"
    HOPPER = "hopper"
    # Rotating equipment
    PUMP = "pump"
    COMPRESSOR = "compressor"
    FAN = "fan"
    BLOWER = "blower"
    TURBINE = "turbine"
    MIXER = "mixer"
    AGITATOR = "agitator"
    CONVEYOR = "conveyor"
    CENTRIFUGE = "centrifuge"
    CRUSHER = "crusher"
    # Piping
    PIPE = "pipe"
    VALVE = "valve"
    # Electrical
    MOTOR = "motor"
    TRANSFORMER = "transformer"
    SWITCHGEAR = "switchgear"
    MCC = "motor_control_center"
    VFD = "variable_frequency_drive"
    UPS = "ups"
    GENERATOR = "generator"
    # Instrumentation
    TRANSMITTER = "transmitter"
    CONTROLLER = "controller"
    FINAL_ELEMENT = "final_element"
    ANALYZER = "analyzer"
    # Safety
    SAFETY_VALVE = "safety_valve"
    RUPTURE_DISC = "rupture_disc"
    FIRE_DETECTOR = "fire_detector"
    GAS_DETECTOR = "gas_detector"
    DELUGE_SYSTEM = "deluge_system"
    # Structural
    STRUCTURE = "structure"
    BUILDING = "building"
    PIPE_RACK = "pipe_rack"
    FOUNDATION = "foundation"


class EquipmentSubclass(str, Enum):
    """Detailed sub-classification."""
    # Pumps
    PUMP_CENTRIFUGAL = "centrifugal_pump"
    PUMP_RECIPROCATING = "reciprocating_pump"
    PUMP_GEAR = "gear_pump"
    PUMP_SCREW = "screw_pump"
    PUMP_DIAPHRAGM = "diaphragm_pump"
    PUMP_SUBMERSIBLE = "submersible_pump"
    # Compressors
    COMP_CENTRIFUGAL = "centrifugal_compressor"
    COMP_RECIPROCATING = "reciprocating_compressor"
    COMP_SCREW = "screw_compressor"
    COMP_AXIAL = "axial_compressor"
    # Valves
    VALVE_GATE = "gate_valve"
    VALVE_GLOBE = "globe_valve"
    VALVE_BALL = "ball_valve"
    VALVE_BUTTERFLY = "butterfly_valve"
    VALVE_CHECK = "check_valve"
    VALVE_PLUG = "plug_valve"
    VALVE_NEEDLE = "needle_valve"
    VALVE_DIAPHRAGM = "diaphragm_valve"
    VALVE_RELIEF = "relief_valve"
    VALVE_CONTROL = "control_valve"
    # Heat exchangers
    HX_SHELL_TUBE = "shell_and_tube"
    HX_PLATE = "plate"
    HX_AIR_COOLED = "air_cooled"
    HX_DOUBLE_PIPE = "double_pipe"
    HX_SPIRAL = "spiral"
    # Vessels
    VESSEL_PRESSURE = "pressure_vessel"
    VESSEL_ATMOSPHERIC = "atmospheric_vessel"
    VESSEL_VACUUM = "vacuum_vessel"
    # Tanks
    TANK_FIXED_ROOF = "fixed_roof_tank"
    TANK_FLOATING_ROOF = "floating_roof_tank"
    TANK_CONE_ROOF = "cone_roof_tank"
    TANK_SPHERICAL = "spherical_tank"
    TANK_BULLET = "bullet_tank"
    # Columns
    COLUMN_TRAY = "tray_column"
    COLUMN_PACKED = "packed_column"
    COLUMN_BUBBLE_CAP = "bubble_cap_column"
    # Separators
    SEP_TWO_PHASE = "two_phase_separator"
    SEP_THREE_PHASE = "three_phase_separator"
    SEP_CYCLONE = "cyclone_separator"
    SEP_COALESCER = "coalescer"


class MeasurementType(str, Enum):
    """ISA standard measurement variable types."""
    TEMPERATURE = "temperature"
    PRESSURE = "pressure"
    FLOW = "flow"
    LEVEL = "level"
    VIBRATION = "vibration"
    SPEED = "speed"
    CURRENT = "current"
    VOLTAGE = "voltage"
    POWER = "power"
    TORQUE = "torque"
    POSITION = "position"
    DENSITY = "density"
    VISCOSITY = "viscosity"
    PH = "ph"
    CONDUCTIVITY = "conductivity"
    CONCENTRATION = "concentration"
    MOISTURE = "moisture"
    HUMIDITY = "humidity"
    OXYGEN = "oxygen"
    H2S = "h2s"
    CO = "co"
    LEL = "lel"  # Lower Explosive Limit
    WEIGHT = "weight"
    THICKNESS = "thickness"  # Corrosion monitoring
    NOISE = "noise"
    DISPLACEMENT = "displacement"


class FailureMode(str, Enum):
    """ISO 14224 aligned failure modes."""
    # General
    ABNORMAL_INSTRUMENT_READING = "abnormal_instrument_reading"
    BREAKDOWN = "breakdown"
    CONTROL_FAILURE = "control_failure"
    ERRATIC_OUTPUT = "erratic_output"
    EXTERNAL_LEAKAGE = "external_leakage"
    INTERNAL_LEAKAGE = "internal_leakage"
    FAIL_TO_FUNCTION = "fail_to_function"
    FAIL_TO_START = "fail_to_start"
    FAIL_TO_STOP = "fail_to_stop"
    HIGH_OUTPUT = "high_output"
    LOW_OUTPUT = "low_output"
    NOISE_EXCESSIVE = "noise_excessive"
    OVERHEATING = "overheating"
    PLUGGED = "plugged"
    SPURIOUS_OPERATION = "spurious_operation"
    STRUCTURAL_FAILURE = "structural_failure"
    VIBRATION_EXCESSIVE = "vibration_excessive"
    # Rotating equipment specific
    BEARING_FAILURE = "bearing_failure"
    SEAL_FAILURE = "seal_failure"
    IMPELLER_DAMAGE = "impeller_damage"
    SHAFT_MISALIGNMENT = "shaft_misalignment"
    COUPLING_FAILURE = "coupling_failure"
    CAVITATION = "cavitation"
    SURGE = "surge"
    # Static equipment specific
    CORROSION = "corrosion"
    EROSION = "erosion"
    FATIGUE_CRACKING = "fatigue_cracking"
    FOULING = "fouling"
    BLOCKAGE = "blockage"
    TUBE_FAILURE = "tube_failure"
    # Electrical
    INSULATION_FAILURE = "insulation_failure"
    WINDING_FAILURE = "winding_failure"
    OVERLOAD = "overload"
    SHORT_CIRCUIT = "short_circuit"
    GROUND_FAULT = "ground_fault"
    # Instrument
    DRIFT = "drift"
    CALIBRATION_ERROR = "calibration_error"
    SIGNAL_LOSS = "signal_loss"
    STUCK = "stuck"


class FailureMechanism(str, Enum):
    """Root cause mechanisms."""
    WEAR = "wear"
    CORROSION_UNIFORM = "uniform_corrosion"
    CORROSION_PITTING = "pitting_corrosion"
    CORROSION_GALVANIC = "galvanic_corrosion"
    CORROSION_CREVICE = "crevice_corrosion"
    CORROSION_SCC = "stress_corrosion_cracking"
    EROSION_FLOW = "flow_erosion"
    EROSION_CAVITATION = "cavitation_erosion"
    FATIGUE_MECHANICAL = "mechanical_fatigue"
    FATIGUE_THERMAL = "thermal_fatigue"
    CREEP = "creep"
    HYDROGEN_EMBRITTLEMENT = "hydrogen_embrittlement"
    OVERLOAD_MECHANICAL = "mechanical_overload"
    OVERLOAD_THERMAL = "thermal_overload"
    OVERLOAD_ELECTRICAL = "electrical_overload"
    CONTAMINATION = "contamination"
    FOULING_BIOLOGICAL = "biological_fouling"
    FOULING_SCALING = "scaling"
    FOULING_COKING = "coking"
    MISALIGNMENT = "misalignment"
    IMBALANCE = "imbalance"
    LOOSENESS = "looseness"
    DESIGN_ERROR = "design_error"
    MANUFACTURING_DEFECT = "manufacturing_defect"
    INSTALLATION_ERROR = "installation_error"
    OPERATING_ERROR = "operating_error"
    MAINTENANCE_ERROR = "maintenance_error"
    ENVIRONMENTAL = "environmental"
    AGE_DEGRADATION = "age_degradation"


class OperatingState(str, Enum):
    """Equipment operating states."""
    RUNNING = "running"
    STANDBY = "standby"
    STARTUP = "startup"
    SHUTDOWN = "shutdown"
    TRIP = "trip"
    MAINTENANCE = "maintenance"
    DECOMMISSIONED = "decommissioned"
    IDLE = "idle"


class CriticalityRating(str, Enum):
    """Equipment criticality per API 580/581."""
    CRITICAL = "critical"       # Plant shutdown if fails
    HIGH = "high"               # Significant production impact
    MEDIUM = "medium"           # Partial production impact
    LOW = "low"                 # Minimal impact
    NON_CRITICAL = "non_critical"


class MaintenanceStrategy(str, Enum):
    RUN_TO_FAILURE = "run_to_failure"
    TIME_BASED = "time_based"
    CONDITION_BASED = "condition_based"
    PREDICTIVE = "predictive"
    RISK_BASED = "risk_based"
    RELIABILITY_CENTERED = "reliability_centered"


class FluidType(str, Enum):
    CRUDE_OIL = "crude_oil"
    NATURAL_GAS = "natural_gas"
    PRODUCED_WATER = "produced_water"
    STEAM = "steam"
    COOLING_WATER = "cooling_water"
    INSTRUMENT_AIR = "instrument_air"
    NITROGEN = "nitrogen"
    HYDROGEN = "hydrogen"
    FUEL_GAS = "fuel_gas"
    CONDENSATE = "condensate"
    NGL = "ngl"
    CHEMICAL_INJECTION = "chemical_injection"
    DIESEL = "diesel"
    LUBE_OIL = "lube_oil"
    HYDRAULIC_OIL = "hydraulic_oil"
    GLYCOL = "glycol"
    AMINE = "amine"
    ACID = "acid"
    CAUSTIC = "caustic"
    SLURRY = "slurry"
    COMPRESSED_AIR = "compressed_air"
    FLARE_GAS = "flare_gas"
    DRAIN = "drain"
    VENT = "vent"
    CUSTOM = "custom"


class RelationshipType(str, Enum):
    """All relationship types in the knowledge graph."""
    # Structural hierarchy
    CONTAINS = "contains"
    PART_OF = "part_of"
    INSTALLED_ON = "installed_on"
    LOCATED_IN = "located_in"
    # Process connections
    FEEDS_TO = "feeds_to"
    RECEIVES_FROM = "receives_from"
    BYPASSES = "bypasses"
    RECYCLES_TO = "recycles_to"
    # Control relationships
    CONTROLS = "controls"
    MEASURES = "measures"
    ACTUATES = "actuates"
    INTERLOCKED_WITH = "interlocked_with"
    TRIPS_ON = "trips_on"
    # Causal / failure relationships
    CAUSES = "causes"
    CAUSED_BY = "caused_by"
    MITIGATES = "mitigates"
    TRIGGERS = "triggers"
    INHIBITS = "inhibits"
    DEGRADES = "degrades"
    CASCADES_TO = "cascades_to"
    CORRELATES_WITH = "correlates_with"
    # Maintenance
    MAINTAINED_BY = "maintained_by"
    REQUIRES_PART = "requires_part"
    FOLLOWS_PROCEDURE = "follows_procedure"
    REQUIRES_PERMIT = "requires_permit"
    # Safety
    PROTECTED_BY = "protected_by"
    PROVIDES_PROTECTION = "provides_protection"
    ISOLATES = "isolates"
    # Electrical
    POWERED_BY = "powered_by"
    DRIVES = "drives"


class SILLevel(int, Enum):
    """Safety Integrity Level per IEC 61511."""
    NONE = 0
    SIL_1 = 1
    SIL_2 = 2
    SIL_3 = 3
    SIL_4 = 4


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
#  ENTITY MODELS — Knowledge Graph Nodes
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

def _uid() -> str:
    return uuid.uuid4().hex[:12]


@dataclass
class KGEntity:
    """Base entity for all knowledge graph nodes."""
    id: str = field(default_factory=_uid)
    name: str = ""
    description: str = ""
    metadata: dict = field(default_factory=dict)
    created_at: str = field(default_factory=lambda: datetime.utcnow().isoformat())


# ── Hierarchy Entities ─────────────────────────────────────────

@dataclass
class Site(KGEntity):
    """Physical site / facility."""
    location: str = ""
    latitude: float = 0.0
    longitude: float = 0.0
    country: str = ""
    operator: str = ""
    regulatory_body: str = ""
    commissioning_year: int = 0


@dataclass
class Plant(KGEntity):
    """A plant within a site (e.g., Gas Processing Plant, Crude Oil Terminal)."""
    site_id: str = ""
    plant_type: str = ""  # refinery, gas_processing, petrochemical, power, manufacturing
    design_capacity: str = ""
    operating_license: str = ""


@dataclass
class Area(KGEntity):
    """Process area within a plant (e.g., Area 100 - Inlet Separation)."""
    plant_id: str = ""
    area_number: int = 0
    area_class: str = ""  # process, utility, offsite, safety
    hazardous_area_class: str = ""  # Zone 0, Zone 1, Zone 2


@dataclass
class ProcessUnit(KGEntity):
    """A functional process unit (e.g., Crude Distillation Unit)."""
    area_id: str = ""
    unit_number: str = ""
    unit_type: str = ""
    design_throughput: float = 0.0
    throughput_unit: str = ""


# ── Equipment Entity ───────────────────────────────────────────

@dataclass
class Equipment(KGEntity):
    """Core equipment entity — the heart of the knowledge graph."""
    tag: str = ""  # Plant tag number (e.g., P-201A)
    unit_id: str = ""

    # Classification
    equipment_class: EquipmentClass = EquipmentClass.VESSEL
    equipment_subclass: Optional[EquipmentSubclass] = None
    criticality: CriticalityRating = CriticalityRating.MEDIUM

    # Design data
    design_pressure: Optional[float] = None
    design_pressure_unit: str = "barg"
    design_temperature: Optional[float] = None
    design_temperature_unit: str = "°C"
    material_of_construction: str = ""
    corrosion_allowance_mm: float = 0.0
    design_code: str = ""  # ASME, API, EN, etc.
    year_installed: int = 0
    manufacturer: str = ""
    model_number: str = ""
    serial_number: str = ""
    weight_kg: float = 0.0

    # Operating data
    operating_state: OperatingState = OperatingState.RUNNING
    normal_operating_pressure: Optional[float] = None
    normal_operating_temperature: Optional[float] = None
    rated_capacity: Optional[float] = None
    capacity_unit: str = ""

    # Rotating equipment specific
    rated_power_kw: Optional[float] = None
    rated_speed_rpm: Optional[float] = None
    driver_type: str = ""  # electric_motor, turbine, engine, hydraulic

    # Maintenance
    maintenance_strategy: MaintenanceStrategy = MaintenanceStrategy.TIME_BASED
    mtbf_hours: Optional[float] = None  # Mean Time Between Failures
    mttr_hours: Optional[float] = None  # Mean Time To Repair
    last_maintenance_date: str = ""
    next_maintenance_date: str = ""

    # Safety
    sil_level: SILLevel = SILLevel.NONE
    is_safety_critical: bool = False
    hazop_node: str = ""

    # Position in 3D space (for digital twin)
    position_x: float = 0.0
    position_y: float = 0.0
    position_z: float = 0.0

    # Cost
    replacement_cost_usd: float = 0.0


# ── Instrument / Sensor Entity ─────────────────────────────────

@dataclass
class Instrument(KGEntity):
    """Instrument or sensor measuring a process variable."""
    tag: str = ""  # ISA tag (e.g., TT-101, PT-201, FIC-301)
    equipment_id: str = ""  # Equipment it measures

    measurement_type: MeasurementType = MeasurementType.TEMPERATURE
    unit: str = ""
    range_min: float = 0.0
    range_max: float = 100.0
    accuracy_pct: float = 1.0
    response_time_ms: float = 500

    # Alarm setpoints
    low_low_alarm: Optional[float] = None
    low_alarm: Optional[float] = None
    high_alarm: Optional[float] = None
    high_high_alarm: Optional[float] = None

    # Interlock trips
    low_low_trip: Optional[float] = None
    high_high_trip: Optional[float] = None

    # Calibration
    last_calibration_date: str = ""
    calibration_interval_days: int = 365
    calibration_drift_pct: float = 0.0

    # Signal
    signal_type: str = "4-20mA"  # 4-20mA, HART, Foundation Fieldbus, Profibus, Modbus, OPC-UA
    io_address: str = ""
    scan_rate_ms: int = 1000

    # Safety
    sil_level: SILLevel = SILLevel.NONE
    is_safety_function: bool = False


# ── Process Connection (Pipe / Stream) ─────────────────────────

@dataclass
class ProcessConnection(KGEntity):
    """A process stream connecting two equipment items."""
    line_number: str = ""  # e.g., 6"-HC-101-A1A
    from_equipment_id: str = ""
    to_equipment_id: str = ""

    # Fluid properties
    fluid_type: FluidType = FluidType.CRUDE_OIL
    fluid_name: str = ""
    pipe_size_inches: float = 0.0
    pipe_schedule: str = ""
    pipe_material: str = ""
    insulation_type: str = ""

    # Design conditions
    design_pressure: float = 0.0
    design_temperature: float = 0.0

    # Normal operating conditions
    normal_flow_rate: float = 0.0
    flow_unit: str = "m³/h"
    normal_pressure: float = 0.0
    normal_temperature: float = 0.0
    fluid_density: float = 0.0
    fluid_viscosity: float = 0.0


# ── Failure Mode Entity ────────────────────────────────────────

@dataclass
class EquipmentFailureMode(KGEntity):
    """A specific failure mode for an equipment type (FMEA)."""
    equipment_class: EquipmentClass = EquipmentClass.PUMP
    equipment_subclass: Optional[EquipmentSubclass] = None

    failure_mode: FailureMode = FailureMode.BEARING_FAILURE
    failure_mechanism: FailureMechanism = FailureMechanism.WEAR
    failure_rate_per_year: float = 0.0
    detection_method: str = ""  # vibration_monitoring, visual_inspection, etc.

    # Effects
    local_effect: str = ""
    system_effect: str = ""
    plant_effect: str = ""

    # Risk
    severity: int = 1  # 1-10
    occurrence: int = 1  # 1-10
    detection_difficulty: int = 1  # 1-10
    rpn: int = 0  # Risk Priority Number = S × O × D

    # Symptoms (what sensors will show)
    sensor_indicators: list = field(default_factory=list)
    # e.g., [{"measurement": "vibration", "pattern": "increasing", "lead_time_hours": 168}]

    # Recommended actions
    preventive_actions: list = field(default_factory=list)
    corrective_actions: list = field(default_factory=list)

    # Cascade effects
    cascade_targets: list = field(default_factory=list)
    # e.g., [{"equipment_tag": "HX-301", "effect": "reduced_cooling", "delay_hours": 2}]


# ── Operating Envelope ─────────────────────────────────────────

@dataclass
class OperatingEnvelope(KGEntity):
    """Defines safe / normal / optimal operating ranges for equipment."""
    equipment_id: str = ""
    state: str = "normal"  # normal, reduced, emergency, startup, shutdown

    parameters: list = field(default_factory=list)
    # Each parameter: {
    #   "measurement_type": "temperature",
    #   "unit": "°C",
    #   "min_safe": 20, "min_normal": 40, "optimal": 65,
    #   "max_normal": 80, "max_safe": 95,
    #   "rate_of_change_limit": 5.0,  # per minute
    # }


# ── Maintenance Procedure ──────────────────────────────────────

@dataclass
class MaintenanceProcedure(KGEntity):
    """Standard maintenance procedure for equipment."""
    equipment_class: EquipmentClass = EquipmentClass.PUMP
    procedure_type: str = "preventive"  # preventive, corrective, predictive, overhaul
    interval_hours: int = 0
    interval_calendar_days: int = 0
    estimated_duration_hours: float = 0.0
    required_skills: list = field(default_factory=list)
    required_tools: list = field(default_factory=list)
    required_spares: list = field(default_factory=list)
    steps: list = field(default_factory=list)
    safety_precautions: list = field(default_factory=list)
    permits_required: list = field(default_factory=list)  # hot_work, confined_space, etc.


# ── Safety Interlock ───────────────────────────────────────────

@dataclass
class SafetyInterlock(KGEntity):
    """Safety Instrumented Function (SIF) / interlock."""
    sif_number: str = ""
    sil_level: SILLevel = SILLevel.SIL_1

    # Initiator (sensor that detects)
    initiator_tag: str = ""
    initiator_type: MeasurementType = MeasurementType.PRESSURE
    trip_setpoint: float = 0.0
    trip_direction: str = "high"  # high, low

    # Logic solver
    logic_description: str = ""
    voting: str = "1oo1"  # 1oo1, 1oo2, 2oo3

    # Final element (what trips)
    final_element_tag: str = ""
    final_element_action: str = "close"  # close, open, trip, shutdown

    # Protected equipment
    protected_equipment_id: str = ""
    hazard_description: str = ""
    consequence_of_failure: str = ""

    # Testing
    proof_test_interval_months: int = 12
    last_proof_test_date: str = ""


# ── Control Loop ───────────────────────────────────────────────

@dataclass
class ControlLoop(KGEntity):
    """Process control loop (PID, cascade, ratio, etc.)."""
    loop_number: str = ""
    loop_type: str = "pid"  # pid, cascade, ratio, feedforward, split_range, override

    # Sensor (PV)
    sensor_tag: str = ""
    process_variable: MeasurementType = MeasurementType.FLOW

    # Controller
    controller_tag: str = ""
    setpoint: float = 0.0
    setpoint_unit: str = ""
    kp: float = 1.0  # Proportional gain
    ki: float = 0.0  # Integral gain
    kd: float = 0.0  # Derivative gain

    # Output (MV → Final Element)
    output_tag: str = ""
    output_type: str = "valve_position"
    output_range_min: float = 0.0
    output_range_max: float = 100.0
    fail_action: str = "close"  # close, open, last

    # Performance
    tuning_date: str = ""
    loop_response_time_s: float = 0.0
    control_error_pct: float = 0.0


# ── Relationship ───────────────────────────────────────────────

@dataclass
class Relationship:
    """A directed relationship between two entities in the knowledge graph."""
    id: str = field(default_factory=_uid)
    source_id: str = ""
    target_id: str = ""
    relationship_type: RelationshipType = RelationshipType.CONTAINS
    properties: dict = field(default_factory=dict)
    # Properties can include: weight, delay_hours, probability, confidence, etc.
    confidence: float = 1.0
    source_document: str = ""  # P&ID number, procedure reference, etc.


# ── Event / Incident ───────────────────────────────────────────

@dataclass
class HistoricalEvent(KGEntity):
    """A historical event (failure, maintenance, modification, incident)."""
    event_type: str = ""  # failure, maintenance, modification, process_upset, near_miss, incident
    equipment_id: str = ""
    start_time: str = ""
    end_time: str = ""
    duration_hours: float = 0.0
    failure_mode: Optional[FailureMode] = None
    failure_mechanism: Optional[FailureMechanism] = None
    root_cause: str = ""
    immediate_cause: str = ""
    production_loss_hours: float = 0.0
    repair_cost_usd: float = 0.0
    corrective_actions: list = field(default_factory=list)
    lessons_learned: str = ""


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
#  PHYSICS MODELS — Sensor correlation rules
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

@dataclass
class PhysicsRule(KGEntity):
    """
    Defines physical relationships between sensor readings.
    Used by the synthetic data generator to create realistic correlated data,
    and by the anomaly detector to validate sensor readings.
    """
    equation: str = ""  # e.g., "P2 = P1 - friction_loss(flow, length, diameter)"
    input_variables: list = field(default_factory=list)
    output_variable: str = ""
    equipment_class: EquipmentClass = EquipmentClass.PUMP
    confidence: float = 0.95
    description: str = ""

    # Correlation parameters
    correlation_type: str = "proportional"  # proportional, inverse, nonlinear, threshold
    correlation_strength: float = 0.8  # 0-1
    time_lag_seconds: float = 0.0
    noise_factor: float = 0.02


@dataclass
class FailureCascadeRule(KGEntity):
    """
    Defines how failure in one equipment cascades to others.
    Used for RCA and what-if analysis.
    """
    source_equipment_class: EquipmentClass = EquipmentClass.PUMP
    source_failure_mode: FailureMode = FailureMode.BREAKDOWN
    target_equipment_class: EquipmentClass = EquipmentClass.HEAT_EXCHANGER
    cascade_effect: str = ""  # "loss_of_flow", "pressure_buildup", "temperature_rise"
    probability: float = 0.8
    delay_hours: float = 0.5
    severity_factor: float = 0.7
    sensor_signature: dict = field(default_factory=dict)
    # e.g., {"flow": "sudden_drop", "pressure": "gradual_rise", "temperature": "gradual_rise"}
    description: str = ""
