"""
PlantCopilot Knowledge Graph — Builder
========================================
Populates the knowledge graph with real-world oil & gas domain knowledge.

This module encodes 14+ years of operational experience into structured data:
  - Equipment templates with design data per ISO 14224
  - Failure modes and effects (FMEA) per equipment class
  - Physics-based sensor correlation rules
  - Operating envelopes (normal, degraded, emergency)
  - Maintenance procedures and intervals
  - Safety interlocks and trip logic
  - Failure cascade chains
  - Control loop configurations

The builder creates a complete "reference plant" that serves as:
  1. Training data seed for ML models
  2. Knowledge base for LLM grounding (RAG)
  3. Template for customer plant onboarding
  4. Validation rules for anomaly detection
"""

import json
import copy
from pathlib import Path
from dataclasses import asdict
from collections import defaultdict
from typing import Optional
from loguru import logger

from .ontology import (
    Site, Plant, Area, ProcessUnit, Equipment, Instrument,
    ProcessConnection, EquipmentFailureMode, OperatingEnvelope,
    MaintenanceProcedure, SafetyInterlock, ControlLoop,
    PhysicsRule, FailureCascadeRule, Relationship, HistoricalEvent,
    EquipmentClass, EquipmentSubclass, MeasurementType,
    FailureMode, FailureMechanism, OperatingState,
    CriticalityRating, MaintenanceStrategy, FluidType,
    RelationshipType, SILLevel, _uid,
)


class PlantKnowledgeGraph:
    """
    In-memory knowledge graph for an industrial plant.

    Stores all entities and relationships, supports:
    - Graph traversal (BFS/DFS from any node)
    - Equipment lookup by tag, class, area
    - Failure cascade path finding
    - Sensor-to-equipment-to-failure-mode mapping
    - Export to JSON, Neo4j Cypher, RDF/OWL, NetworkX
    """

    def __init__(self):
        # Entity stores (id → entity)
        self.sites: dict[str, Site] = {}
        self.plants: dict[str, Plant] = {}
        self.areas: dict[str, Area] = {}
        self.units: dict[str, ProcessUnit] = {}
        self.equipment: dict[str, Equipment] = {}
        self.instruments: dict[str, Instrument] = {}
        self.connections: dict[str, ProcessConnection] = {}
        self.failure_modes: dict[str, EquipmentFailureMode] = {}
        self.operating_envelopes: dict[str, OperatingEnvelope] = {}
        self.maintenance_procedures: dict[str, MaintenanceProcedure] = {}
        self.safety_interlocks: dict[str, SafetyInterlock] = {}
        self.control_loops: dict[str, ControlLoop] = {}
        self.physics_rules: dict[str, PhysicsRule] = {}
        self.cascade_rules: dict[str, FailureCascadeRule] = {}
        self.events: dict[str, HistoricalEvent] = {}
        self.relationships: list[Relationship] = []

        # Indexes for fast lookup
        self._tag_index: dict[str, str] = {}  # tag → entity id
        self._class_index: dict[str, list] = defaultdict(list)  # class → [ids]
        self._area_index: dict[str, list] = defaultdict(list)
        self._adj_list: dict[str, list] = defaultdict(list)  # entity_id → [(target_id, rel)]

    # ── Entity Registration ──────────────────────────

    def add_entity(self, entity) -> str:
        """Add any entity to the graph."""
        store_map = {
            Site: self.sites, Plant: self.plants, Area: self.areas,
            ProcessUnit: self.units, Equipment: self.equipment,
            Instrument: self.instruments, ProcessConnection: self.connections,
            EquipmentFailureMode: self.failure_modes,
            OperatingEnvelope: self.operating_envelopes,
            MaintenanceProcedure: self.maintenance_procedures,
            SafetyInterlock: self.safety_interlocks,
            ControlLoop: self.control_loops,
            PhysicsRule: self.physics_rules,
            FailureCascadeRule: self.cascade_rules,
            HistoricalEvent: self.events,
        }
        store = store_map.get(type(entity))
        if store is not None:
            store[entity.id] = entity
        # Index by tag
        if hasattr(entity, "tag") and entity.tag:
            self._tag_index[entity.tag] = entity.id
        if hasattr(entity, "equipment_class"):
            self._class_index[entity.equipment_class.value].append(entity.id)
        return entity.id

    def add_relationship(self, source_id: str, target_id: str,
                          rel_type: RelationshipType, **props) -> str:
        """Add a directed relationship between two entities."""
        rel = Relationship(
            source_id=source_id, target_id=target_id,
            relationship_type=rel_type, properties=props,
        )
        self.relationships.append(rel)
        self._adj_list[source_id].append((target_id, rel))
        return rel.id

    # ── Queries ──────────────────────────────────────

    def get_by_tag(self, tag: str):
        """Look up any entity by its plant tag."""
        eid = self._tag_index.get(tag)
        if not eid:
            return None
        for store in [self.equipment, self.instruments]:
            if eid in store:
                return store[eid]
        return None

    def get_equipment_by_class(self, eq_class: EquipmentClass) -> list[Equipment]:
        ids = self._class_index.get(eq_class.value, [])
        return [self.equipment[eid] for eid in ids if eid in self.equipment]

    def get_instruments_for_equipment(self, equipment_id: str) -> list[Instrument]:
        return [inst for inst in self.instruments.values()
                if inst.equipment_id == equipment_id]

    def get_failure_modes_for_class(self, eq_class: EquipmentClass) -> list[EquipmentFailureMode]:
        return [fm for fm in self.failure_modes.values()
                if fm.equipment_class == eq_class]

    def get_cascade_paths(self, equipment_id: str, max_depth: int = 5) -> list[list]:
        """Find all failure cascade paths from an equipment item (BFS)."""
        paths = []
        visited = set()
        queue = [([equipment_id], 0)]
        while queue:
            path, depth = queue.pop(0)
            if depth >= max_depth:
                continue
            current = path[-1]
            if current in visited and depth > 0:
                continue
            visited.add(current)
            for target_id, rel in self._adj_list.get(current, []):
                if rel.relationship_type in (RelationshipType.CASCADES_TO,
                                              RelationshipType.FEEDS_TO,
                                              RelationshipType.CAUSES):
                    new_path = path + [target_id]
                    paths.append(new_path)
                    queue.append((new_path, depth + 1))
        return paths

    def get_upstream(self, equipment_id: str) -> list[Equipment]:
        """Get all equipment that feeds into this equipment."""
        upstream = []
        for conn in self.connections.values():
            if conn.to_equipment_id == equipment_id:
                eq = self.equipment.get(conn.from_equipment_id)
                if eq:
                    upstream.append(eq)
        return upstream

    def get_downstream(self, equipment_id: str) -> list[Equipment]:
        """Get all equipment that this equipment feeds to."""
        downstream = []
        for conn in self.connections.values():
            if conn.from_equipment_id == equipment_id:
                eq = self.equipment.get(conn.to_equipment_id)
                if eq:
                    downstream.append(eq)
        return downstream

    def get_safety_functions(self, equipment_id: str) -> list[SafetyInterlock]:
        return [sif for sif in self.safety_interlocks.values()
                if sif.protected_equipment_id == equipment_id]

    # ── Statistics ───────────────────────────────────

    def stats(self) -> dict:
        return {
            "sites": len(self.sites),
            "plants": len(self.plants),
            "areas": len(self.areas),
            "units": len(self.units),
            "equipment": len(self.equipment),
            "instruments": len(self.instruments),
            "connections": len(self.connections),
            "failure_modes": len(self.failure_modes),
            "operating_envelopes": len(self.operating_envelopes),
            "maintenance_procedures": len(self.maintenance_procedures),
            "safety_interlocks": len(self.safety_interlocks),
            "control_loops": len(self.control_loops),
            "physics_rules": len(self.physics_rules),
            "cascade_rules": len(self.cascade_rules),
            "historical_events": len(self.events),
            "relationships": len(self.relationships),
            "total_entities": sum([
                len(self.sites), len(self.plants), len(self.areas), len(self.units),
                len(self.equipment), len(self.instruments), len(self.connections),
                len(self.failure_modes), len(self.operating_envelopes),
                len(self.maintenance_procedures), len(self.safety_interlocks),
                len(self.control_loops), len(self.physics_rules),
                len(self.cascade_rules), len(self.events),
            ]),
        }

    # ── Export ───────────────────────────────────────

    def export_json(self, filepath: str):
        """Export entire knowledge graph to JSON."""
        data = {
            "metadata": {"version": "1.0", "generator": "PlantCopilot KG Builder"},
            "statistics": self.stats(),
            "sites": {k: asdict(v) for k, v in self.sites.items()},
            "plants": {k: asdict(v) for k, v in self.plants.items()},
            "areas": {k: asdict(v) for k, v in self.areas.items()},
            "units": {k: asdict(v) for k, v in self.units.items()},
            "equipment": {k: asdict(v) for k, v in self.equipment.items()},
            "instruments": {k: asdict(v) for k, v in self.instruments.items()},
            "connections": {k: asdict(v) for k, v in self.connections.items()},
            "failure_modes": {k: asdict(v) for k, v in self.failure_modes.items()},
            "operating_envelopes": {k: asdict(v) for k, v in self.operating_envelopes.items()},
            "maintenance_procedures": {k: asdict(v) for k, v in self.maintenance_procedures.items()},
            "safety_interlocks": {k: asdict(v) for k, v in self.safety_interlocks.items()},
            "control_loops": {k: asdict(v) for k, v in self.control_loops.items()},
            "physics_rules": {k: asdict(v) for k, v in self.physics_rules.items()},
            "cascade_rules": {k: asdict(v) for k, v in self.cascade_rules.items()},
            "events": {k: asdict(v) for k, v in self.events.items()},
            "relationships": [asdict(r) for r in self.relationships],
        }
        Path(filepath).write_text(json.dumps(data, indent=2, default=str), encoding="utf-8")
        logger.info(f"Knowledge graph exported to {filepath} ({self.stats()['total_entities']} entities)")

    def export_cypher(self, filepath: str):
        """Export as Neo4j Cypher statements."""
        lines = ["// PlantCopilot Knowledge Graph — Neo4j Import\n"]

        # Equipment nodes
        for eq in self.equipment.values():
            props = f'tag: "{eq.tag}", name: "{eq.name}", class: "{eq.equipment_class.value}", '
            props += f'criticality: "{eq.criticality.value}", state: "{eq.operating_state.value}"'
            if eq.design_pressure:
                props += f', design_pressure: {eq.design_pressure}'
            if eq.design_temperature:
                props += f', design_temperature: {eq.design_temperature}'
            lines.append(f'CREATE (:{eq.equipment_class.value.upper()} {{{props}}});')

        # Instrument nodes
        for inst in self.instruments.values():
            props = f'tag: "{inst.tag}", type: "{inst.measurement_type.value}", unit: "{inst.unit}"'
            lines.append(f'CREATE (:INSTRUMENT {{{props}}});')

        # Relationships
        for rel in self.relationships:
            src_tag = self._id_to_tag(rel.source_id)
            tgt_tag = self._id_to_tag(rel.target_id)
            if src_tag and tgt_tag:
                lines.append(
                    f'MATCH (a {{tag: "{src_tag}"}}), (b {{tag: "{tgt_tag}"}}) '
                    f'CREATE (a)-[:{rel.relationship_type.value.upper()}]->(b);'
                )

        Path(filepath).write_text("\n".join(lines), encoding="utf-8")
        logger.info(f"Cypher export: {filepath}")

    def export_for_llm(self, filepath: str):
        """Export knowledge graph as LLM-friendly text for RAG grounding."""
        lines = ["# Plant Knowledge Base\n"]

        for eq in self.equipment.values():
            lines.append(f"\n## Equipment: {eq.tag} — {eq.name}")
            lines.append(f"- Type: {eq.equipment_class.value} ({eq.equipment_subclass.value if eq.equipment_subclass else 'N/A'})")
            lines.append(f"- Criticality: {eq.criticality.value}")
            lines.append(f"- Design: {eq.design_pressure} {eq.design_pressure_unit} / {eq.design_temperature} {eq.design_temperature_unit}")
            lines.append(f"- Material: {eq.material_of_construction}")
            lines.append(f"- Status: {eq.operating_state.value}")

            instruments = self.get_instruments_for_equipment(eq.id)
            if instruments:
                lines.append(f"- Instruments ({len(instruments)}):")
                for inst in instruments:
                    alarm_info = []
                    if inst.high_alarm: alarm_info.append(f"H={inst.high_alarm}")
                    if inst.high_high_alarm: alarm_info.append(f"HH={inst.high_high_alarm}")
                    if inst.low_alarm: alarm_info.append(f"L={inst.low_alarm}")
                    if inst.low_low_alarm: alarm_info.append(f"LL={inst.low_low_alarm}")
                    alarm_str = f" [{', '.join(alarm_info)}]" if alarm_info else ""
                    lines.append(f"  - {inst.tag}: {inst.measurement_type.value} ({inst.range_min}-{inst.range_max} {inst.unit}){alarm_str}")

            fms = self.get_failure_modes_for_class(eq.equipment_class)
            if fms:
                lines.append(f"- Known failure modes:")
                for fm in fms[:5]:
                    lines.append(f"  - {fm.failure_mode.value}: {fm.description} (RPN={fm.rpn})")

            downstream = self.get_downstream(eq.id)
            upstream = self.get_upstream(eq.id)
            if upstream:
                lines.append(f"- Receives from: {', '.join(u.tag for u in upstream)}")
            if downstream:
                lines.append(f"- Feeds to: {', '.join(d.tag for d in downstream)}")

            sifs = self.get_safety_functions(eq.id)
            if sifs:
                lines.append(f"- Safety functions:")
                for sif in sifs:
                    lines.append(f"  - {sif.sif_number}: {sif.logic_description}")

        Path(filepath).write_text("\n".join(lines), encoding="utf-8")
        logger.info(f"LLM knowledge base exported: {filepath}")

    def _id_to_tag(self, entity_id: str) -> Optional[str]:
        for tag, eid in self._tag_index.items():
            if eid == entity_id:
                return tag
        return None


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
#  DOMAIN KNOWLEDGE — Oil & Gas Reference Data
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

class OilGasDomainBuilder:
    """
    Populates a PlantKnowledgeGraph with comprehensive oil & gas domain knowledge.

    Encodes:
    - Typical equipment configurations for a gas processing facility
    - FMEA data per ISO 14224 for each equipment class
    - Physics rules for sensor correlations
    - Operating envelopes
    - Maintenance procedures
    - Safety interlocks
    - Failure cascade chains
    """

    def __init__(self, kg: PlantKnowledgeGraph):
        self.kg = kg

    def build_reference_plant(self) -> PlantKnowledgeGraph:
        """Build a complete reference gas processing plant."""
        logger.info("Building reference oil & gas plant knowledge graph...")

        self._build_hierarchy()
        self._build_equipment()
        self._build_instruments()
        self._build_process_connections()
        self._build_failure_modes()
        self._build_operating_envelopes()
        self._build_physics_rules()
        self._build_maintenance_procedures()
        self._build_safety_interlocks()
        self._build_control_loops()
        self._build_cascade_rules()
        self._build_relationships()

        stats = self.kg.stats()
        logger.info(f"Knowledge graph built: {stats['total_entities']} entities, {stats['relationships']} relationships")
        return self.kg

    # ── Plant Hierarchy ─────────────────────────────

    def _build_hierarchy(self):
        site = Site(id="site_01", name="Offshore Production Facility",
                    location="Arabian Gulf", country="UAE", operator="PlantCopilot Demo")
        self.kg.add_entity(site)

        plant = Plant(id="plant_01", name="Gas Processing Plant", site_id=site.id,
                      plant_type="gas_processing", design_capacity="500 MMSCFD")
        self.kg.add_entity(plant)

        areas = [
            Area(id="area_100", name="Inlet Separation", plant_id=plant.id, area_number=100,
                 area_class="process", hazardous_area_class="Zone 1"),
            Area(id="area_200", name="Compression", plant_id=plant.id, area_number=200,
                 area_class="process", hazardous_area_class="Zone 1"),
            Area(id="area_300", name="Gas Treatment", plant_id=plant.id, area_number=300,
                 area_class="process", hazardous_area_class="Zone 1"),
            Area(id="area_400", name="Dehydration", plant_id=plant.id, area_number=400,
                 area_class="process", hazardous_area_class="Zone 2"),
            Area(id="area_500", name="NGL Recovery", plant_id=plant.id, area_number=500,
                 area_class="process", hazardous_area_class="Zone 1"),
            Area(id="area_600", name="Utilities", plant_id=plant.id, area_number=600,
                 area_class="utility", hazardous_area_class="Zone 2"),
            Area(id="area_700", name="Flare & Relief", plant_id=plant.id, area_number=700,
                 area_class="safety"),
            Area(id="area_800", name="Power Generation", plant_id=plant.id, area_number=800,
                 area_class="utility"),
        ]
        for area in areas:
            self.kg.add_entity(area)

        self.areas = {a.id: a for a in areas}

    # ── Equipment ───────────────────────────────────

    def _build_equipment(self):
        equipment_definitions = [
            # ── Area 100: Inlet Separation ──
            Equipment(
                id="eq_V101", tag="V-101", name="Inlet Separator",
                unit_id="area_100", equipment_class=EquipmentClass.SEPARATOR,
                equipment_subclass=EquipmentSubclass.SEP_THREE_PHASE,
                criticality=CriticalityRating.CRITICAL,
                design_pressure=75.0, design_temperature=120.0,
                material_of_construction="SA-516 Gr.70 + 3mm CA",
                corrosion_allowance_mm=3.0, design_code="ASME VIII Div.1",
                normal_operating_pressure=65.0, normal_operating_temperature=85.0,
                year_installed=2015, manufacturer="BHGE",
                maintenance_strategy=MaintenanceStrategy.CONDITION_BASED,
                mtbf_hours=35000, mttr_hours=24,
                is_safety_critical=True, replacement_cost_usd=2500000,
            ),
            Equipment(
                id="eq_V102", tag="V-102", name="Test Separator",
                unit_id="area_100", equipment_class=EquipmentClass.SEPARATOR,
                equipment_subclass=EquipmentSubclass.SEP_TWO_PHASE,
                criticality=CriticalityRating.MEDIUM,
                design_pressure=75.0, design_temperature=120.0,
                material_of_construction="SA-516 Gr.70",
                normal_operating_pressure=60.0, normal_operating_temperature=80.0,
                mtbf_hours=45000,
            ),
            Equipment(
                id="eq_TK101", tag="TK-101", name="Produced Water Tank",
                unit_id="area_100", equipment_class=EquipmentClass.TANK,
                equipment_subclass=EquipmentSubclass.TANK_FIXED_ROOF,
                criticality=CriticalityRating.HIGH,
                design_pressure=0.07, design_temperature=80.0,
                material_of_construction="A283 Gr.C",
                rated_capacity=5000.0, capacity_unit="m³",
                maintenance_strategy=MaintenanceStrategy.TIME_BASED,
                mtbf_hours=80000,
            ),
            # ── Area 200: Compression ──
            Equipment(
                id="eq_C201", tag="C-201", name="1st Stage Gas Compressor",
                unit_id="area_200", equipment_class=EquipmentClass.COMPRESSOR,
                equipment_subclass=EquipmentSubclass.COMP_CENTRIFUGAL,
                criticality=CriticalityRating.CRITICAL,
                design_pressure=45.0, design_temperature=150.0,
                material_of_construction="AISI 4140",
                rated_power_kw=4500.0, rated_speed_rpm=11500,
                driver_type="gas_turbine",
                normal_operating_pressure=35.0, normal_operating_temperature=120.0,
                maintenance_strategy=MaintenanceStrategy.PREDICTIVE,
                mtbf_hours=25000, mttr_hours=72,
                is_safety_critical=True, replacement_cost_usd=8500000,
            ),
            Equipment(
                id="eq_C202", tag="C-202", name="2nd Stage Gas Compressor",
                unit_id="area_200", equipment_class=EquipmentClass.COMPRESSOR,
                equipment_subclass=EquipmentSubclass.COMP_CENTRIFUGAL,
                criticality=CriticalityRating.CRITICAL,
                design_pressure=85.0, design_temperature=160.0,
                rated_power_kw=5200.0, rated_speed_rpm=12000,
                driver_type="gas_turbine",
                normal_operating_pressure=72.0, normal_operating_temperature=135.0,
                maintenance_strategy=MaintenanceStrategy.PREDICTIVE,
                mtbf_hours=22000, mttr_hours=96,
                is_safety_critical=True, replacement_cost_usd=9200000,
            ),
            Equipment(
                id="eq_E201", tag="E-201", name="1st Stage Aftercooler",
                unit_id="area_200", equipment_class=EquipmentClass.HEAT_EXCHANGER,
                equipment_subclass=EquipmentSubclass.HX_SHELL_TUBE,
                criticality=CriticalityRating.HIGH,
                design_pressure=45.0, design_temperature=150.0,
                material_of_construction="SA-516 Gr.70 / Admiralty Brass tubes",
                normal_operating_pressure=35.0, normal_operating_temperature=95.0,
                maintenance_strategy=MaintenanceStrategy.CONDITION_BASED,
                mtbf_hours=50000,
            ),
            Equipment(
                id="eq_V201", tag="V-201", name="1st Stage Suction Scrubber",
                unit_id="area_200", equipment_class=EquipmentClass.SEPARATOR,
                equipment_subclass=EquipmentSubclass.SEP_TWO_PHASE,
                criticality=CriticalityRating.HIGH,
                design_pressure=20.0, design_temperature=100.0,
            ),
            Equipment(
                id="eq_V202", tag="V-202", name="2nd Stage Suction Scrubber",
                unit_id="area_200", equipment_class=EquipmentClass.SEPARATOR,
                equipment_subclass=EquipmentSubclass.SEP_TWO_PHASE,
                criticality=CriticalityRating.HIGH,
                design_pressure=50.0, design_temperature=120.0,
            ),
            # ── Area 300: Gas Treatment ──
            Equipment(
                id="eq_T301", tag="T-301", name="Amine Contactor",
                unit_id="area_300", equipment_class=EquipmentClass.COLUMN,
                equipment_subclass=EquipmentSubclass.COLUMN_TRAY,
                criticality=CriticalityRating.CRITICAL,
                design_pressure=85.0, design_temperature=80.0,
                material_of_construction="SA-516 Gr.70 + SS316L cladding",
                maintenance_strategy=MaintenanceStrategy.CONDITION_BASED,
                mtbf_hours=60000,
            ),
            Equipment(
                id="eq_T302", tag="T-302", name="Amine Regenerator",
                unit_id="area_300", equipment_class=EquipmentClass.COLUMN,
                equipment_subclass=EquipmentSubclass.COLUMN_TRAY,
                criticality=CriticalityRating.CRITICAL,
                design_pressure=3.5, design_temperature=130.0,
                material_of_construction="SA-516 Gr.70 + SS316L cladding",
            ),
            Equipment(
                id="eq_P301", tag="P-301A", name="Lean Amine Pump (A)",
                unit_id="area_300", equipment_class=EquipmentClass.PUMP,
                equipment_subclass=EquipmentSubclass.PUMP_CENTRIFUGAL,
                criticality=CriticalityRating.HIGH,
                rated_power_kw=250.0, rated_speed_rpm=2960,
                driver_type="electric_motor",
                maintenance_strategy=MaintenanceStrategy.PREDICTIVE,
                mtbf_hours=30000, mttr_hours=12,
            ),
            Equipment(
                id="eq_P301B", tag="P-301B", name="Lean Amine Pump (B) [Standby]",
                unit_id="area_300", equipment_class=EquipmentClass.PUMP,
                equipment_subclass=EquipmentSubclass.PUMP_CENTRIFUGAL,
                criticality=CriticalityRating.HIGH,
                rated_power_kw=250.0, rated_speed_rpm=2960,
                driver_type="electric_motor",
                operating_state=OperatingState.STANDBY,
            ),
            Equipment(
                id="eq_E301", tag="E-301", name="Lean/Rich Amine Exchanger",
                unit_id="area_300", equipment_class=EquipmentClass.HEAT_EXCHANGER,
                equipment_subclass=EquipmentSubclass.HX_PLATE,
                criticality=CriticalityRating.HIGH,
                design_pressure=85.0, design_temperature=130.0,
            ),
            Equipment(
                id="eq_E302", tag="E-302", name="Lean Amine Cooler",
                unit_id="area_300", equipment_class=EquipmentClass.HEAT_EXCHANGER,
                equipment_subclass=EquipmentSubclass.HX_AIR_COOLED,
                criticality=CriticalityRating.MEDIUM,
            ),
            # ── Area 400: Dehydration ──
            Equipment(
                id="eq_T401", tag="T-401", name="Glycol Contactor",
                unit_id="area_400", equipment_class=EquipmentClass.COLUMN,
                equipment_subclass=EquipmentSubclass.COLUMN_PACKED,
                criticality=CriticalityRating.HIGH,
                design_pressure=85.0, design_temperature=60.0,
            ),
            Equipment(
                id="eq_E401", tag="E-401", name="Glycol Reboiler",
                unit_id="area_400", equipment_class=EquipmentClass.HEAT_EXCHANGER,
                equipment_subclass=EquipmentSubclass.HX_SHELL_TUBE,
                criticality=CriticalityRating.HIGH,
                design_temperature=204.0,
            ),
            # ── Area 500: NGL Recovery ──
            Equipment(
                id="eq_T501", tag="T-501", name="Demethanizer",
                unit_id="area_500", equipment_class=EquipmentClass.COLUMN,
                equipment_subclass=EquipmentSubclass.COLUMN_TRAY,
                criticality=CriticalityRating.CRITICAL,
                design_pressure=35.0, design_temperature=-50.0,
            ),
            Equipment(
                id="eq_E501", tag="E-501", name="Gas-Gas Exchanger",
                unit_id="area_500", equipment_class=EquipmentClass.HEAT_EXCHANGER,
                equipment_subclass=EquipmentSubclass.HX_PLATE,
                criticality=CriticalityRating.HIGH,
            ),
            # ── Area 600: Utilities ──
            Equipment(
                id="eq_P601", tag="P-601A", name="Cooling Water Pump (A)",
                unit_id="area_600", equipment_class=EquipmentClass.PUMP,
                equipment_subclass=EquipmentSubclass.PUMP_CENTRIFUGAL,
                criticality=CriticalityRating.HIGH,
                rated_power_kw=350.0, rated_speed_rpm=1480,
                driver_type="electric_motor",
                mtbf_hours=35000, mttr_hours=8,
            ),
            Equipment(
                id="eq_C601", tag="C-601", name="Instrument Air Compressor",
                unit_id="area_600", equipment_class=EquipmentClass.COMPRESSOR,
                equipment_subclass=EquipmentSubclass.COMP_SCREW,
                criticality=CriticalityRating.CRITICAL,
                rated_power_kw=200.0, rated_speed_rpm=3000,
                design_pressure=8.0,
            ),
            # ── Area 700: Flare ──
            Equipment(
                id="eq_V701", tag="V-701", name="Flare KO Drum",
                unit_id="area_700", equipment_class=EquipmentClass.DRUM,
                criticality=CriticalityRating.CRITICAL,
                design_pressure=3.5, is_safety_critical=True,
            ),
            # ── Valves (key process valves) ──
            Equipment(
                id="eq_XV101", tag="XV-101", name="Inlet ESD Valve",
                unit_id="area_100", equipment_class=EquipmentClass.VALVE,
                equipment_subclass=EquipmentSubclass.VALVE_BALL,
                criticality=CriticalityRating.CRITICAL,
                sil_level=SILLevel.SIL_2, is_safety_critical=True,
            ),
            Equipment(
                id="eq_FV301", tag="FV-301", name="Amine Flow Control Valve",
                unit_id="area_300", equipment_class=EquipmentClass.VALVE,
                equipment_subclass=EquipmentSubclass.VALVE_CONTROL,
                criticality=CriticalityRating.HIGH,
            ),
        ]

        for eq in equipment_definitions:
            self.kg.add_entity(eq)

    # ── Instruments / Sensors ──────────────────────

    def _build_instruments(self):
        """Build comprehensive instrument list for each equipment."""
        instrument_templates = {
            # V-101 Inlet Separator
            "eq_V101": [
                ("PT-101", MeasurementType.PRESSURE, "barg", 0, 100, None, None, 70, 75),
                ("TT-101", MeasurementType.TEMPERATURE, "°C", 0, 150, None, 30, 100, 110),
                ("LT-101A", MeasurementType.LEVEL, "%", 0, 100, 10, 15, 85, 90),
                ("LT-101B", MeasurementType.LEVEL, "%", 0, 100, 10, 15, 85, 90),  # Redundant
                ("FT-101", MeasurementType.FLOW, "m³/h", 0, 5000, 50, 200, None, None),
            ],
            # C-201 Compressor
            "eq_C201": [
                ("PT-201S", MeasurementType.PRESSURE, "barg", 0, 30, 3, 5, None, None),
                ("PT-201D", MeasurementType.PRESSURE, "barg", 0, 50, None, None, 42, 45),
                ("TT-201S", MeasurementType.TEMPERATURE, "°C", 0, 100, None, None, None, None),
                ("TT-201D", MeasurementType.TEMPERATURE, "°C", 0, 180, None, None, 140, 150),
                ("VT-201DE", MeasurementType.VIBRATION, "mm/s", 0, 25, None, None, 12, 18),
                ("VT-201NDE", MeasurementType.VIBRATION, "mm/s", 0, 25, None, None, 12, 18),
                ("ST-201", MeasurementType.SPEED, "RPM", 0, 13000, 8000, 9000, 12000, 12500),
                ("TT-201B1", MeasurementType.TEMPERATURE, "°C", 0, 130, None, None, 95, 110),
                ("TT-201B2", MeasurementType.TEMPERATURE, "°C", 0, 130, None, None, 95, 110),
                ("YT-201AX", MeasurementType.DISPLACEMENT, "μm", 0, 150, None, None, 75, 100),
                ("YT-201AY", MeasurementType.DISPLACEMENT, "μm", 0, 150, None, None, 75, 100),
                ("FT-201", MeasurementType.FLOW, "m³/h", 0, 50000, None, None, None, None),
                ("IT-201", MeasurementType.CURRENT, "A", 0, 500, None, None, 400, 450),
            ],
            # C-202 Second stage compressor (similar)
            "eq_C202": [
                ("PT-202S", MeasurementType.PRESSURE, "barg", 0, 50, 30, 33, None, None),
                ("PT-202D", MeasurementType.PRESSURE, "barg", 0, 90, None, None, 80, 85),
                ("TT-202D", MeasurementType.TEMPERATURE, "°C", 0, 200, None, None, 150, 160),
                ("VT-202DE", MeasurementType.VIBRATION, "mm/s", 0, 25, None, None, 12, 18),
                ("ST-202", MeasurementType.SPEED, "RPM", 0, 14000, 9000, 10000, 12500, 13000),
            ],
            # T-301 Amine Contactor
            "eq_T301": [
                ("PT-301T", MeasurementType.PRESSURE, "barg", 0, 90, None, None, 80, 85),
                ("PT-301B", MeasurementType.PRESSURE, "barg", 0, 90, None, None, 82, 85),
                ("TT-301T", MeasurementType.TEMPERATURE, "°C", 0, 80, None, None, 60, 70),
                ("TT-301B", MeasurementType.TEMPERATURE, "°C", 0, 80, None, None, 55, 65),
                ("LT-301", MeasurementType.LEVEL, "%", 0, 100, 10, 20, 80, 90),
                ("AT-301", MeasurementType.CONCENTRATION, "ppm", 0, 100, None, None, 4, 10),
            ],
            # P-301A Amine Pump
            "eq_P301": [
                ("PT-301S", MeasurementType.PRESSURE, "barg", 0, 5, 0.5, 1.0, None, None),
                ("PT-301D", MeasurementType.PRESSURE, "barg", 0, 15, None, None, 12, 14),
                ("FT-301", MeasurementType.FLOW, "m³/h", 0, 500, 30, 50, 400, 450),
                ("TT-301P", MeasurementType.TEMPERATURE, "°C", 0, 100, None, None, 75, 85),
                ("VT-301", MeasurementType.VIBRATION, "mm/s", 0, 15, None, None, 8, 12),
                ("IT-301", MeasurementType.CURRENT, "A", 0, 200, None, 20, 160, 180),
            ],
            # E-201 Aftercooler
            "eq_E201": [
                ("TT-201HI", MeasurementType.TEMPERATURE, "°C", 0, 180, None, None, None, None),
                ("TT-201HO", MeasurementType.TEMPERATURE, "°C", 0, 100, None, None, 50, 55),
                ("TT-201CI", MeasurementType.TEMPERATURE, "°C", 0, 50, None, None, None, None),
                ("TT-201CO", MeasurementType.TEMPERATURE, "°C", 0, 80, None, None, 50, 55),
                ("PT-201E", MeasurementType.PRESSURE, "barg", 0, 50, None, None, 42, 45),
                ("PDT-201", MeasurementType.PRESSURE, "bar", 0, 5, None, None, 1.5, 2.0),
            ],
            # P-601A Cooling Water Pump
            "eq_P601": [
                ("PT-601D", MeasurementType.PRESSURE, "barg", 0, 10, None, None, 7, 8),
                ("FT-601", MeasurementType.FLOW, "m³/h", 0, 2000, 200, 400, 1800, None),
                ("TT-601", MeasurementType.TEMPERATURE, "°C", 0, 50, None, None, 38, 42),
                ("VT-601", MeasurementType.VIBRATION, "mm/s", 0, 15, None, None, 8, 12),
                ("IT-601", MeasurementType.CURRENT, "A", 0, 400, None, 30, 300, 350),
            ],
            # C-601 Instrument Air
            "eq_C601": [
                ("PT-601IA", MeasurementType.PRESSURE, "barg", 0, 10, 5, 6, 8, 9),
                ("TT-601IA", MeasurementType.TEMPERATURE, "°C", 0, 120, None, None, 90, 100),
            ],
            # TK-101 Produced Water
            "eq_TK101": [
                ("LT-101T", MeasurementType.LEVEL, "%", 0, 100, 5, 10, 90, 95),
                ("TT-101T", MeasurementType.TEMPERATURE, "°C", 0, 80, None, None, 65, 70),
            ],
        }

        for eq_id, sensors in instrument_templates.items():
            for tag, mtype, unit, rmin, rmax, ll, la, ha, hh in sensors:
                inst = Instrument(
                    id=f"inst_{tag.replace('-', '_').lower()}",
                    tag=tag, equipment_id=eq_id,
                    measurement_type=mtype, unit=unit,
                    range_min=rmin, range_max=rmax,
                    low_low_alarm=ll, low_alarm=la,
                    high_alarm=ha, high_high_alarm=hh,
                    signal_type="4-20mA HART", scan_rate_ms=1000,
                )
                self.kg.add_entity(inst)

    # ── Process Connections ─────────────────────────

    def _build_process_connections(self):
        connections = [
            # Inlet → Separator
            ("eq_V101", "eq_C201", "Well Fluid Inlet", FluidType.CRUDE_OIL, 12, 65, 85),
            ("eq_V101", "eq_TK101", "Produced Water", FluidType.PRODUCED_WATER, 6, 2, 75),
            # Compression train
            ("eq_V201", "eq_C201", "1st Stage Suction", FluidType.NATURAL_GAS, 24, 8, 60),
            ("eq_C201", "eq_E201", "1st Stage Discharge", FluidType.NATURAL_GAS, 16, 35, 120),
            ("eq_E201", "eq_V202", "Cooled Gas", FluidType.NATURAL_GAS, 16, 34, 45),
            ("eq_V202", "eq_C202", "2nd Stage Suction", FluidType.NATURAL_GAS, 16, 33, 42),
            ("eq_C202", "eq_T301", "HP Gas to Treatment", FluidType.NATURAL_GAS, 12, 72, 135),
            # Gas treatment
            ("eq_T301", "eq_T401", "Sweet Gas", FluidType.NATURAL_GAS, 12, 70, 45),
            ("eq_T301", "eq_E301", "Rich Amine", FluidType.AMINE, 8, 78, 70),
            ("eq_E301", "eq_T302", "Hot Rich Amine", FluidType.AMINE, 8, 3, 98),
            ("eq_T302", "eq_E301", "Hot Lean Amine", FluidType.AMINE, 8, 2.5, 120),
            ("eq_E301", "eq_E302", "Warm Lean Amine", FluidType.AMINE, 8, 80, 75),
            ("eq_E302", "eq_P301", "Cool Lean Amine", FluidType.AMINE, 8, 78, 45),
            ("eq_P301", "eq_T301", "Lean Amine to Contactor", FluidType.AMINE, 8, 82, 42),
            # Dehydration
            ("eq_T401", "eq_T501", "Dry Gas", FluidType.NATURAL_GAS, 12, 68, 40),
            # NGL Recovery
            ("eq_T501", "eq_E501", "Sales Gas", FluidType.NATURAL_GAS, 12, 65, -20),
            # Utilities
            ("eq_P601", "eq_E201", "Cooling Water Supply", FluidType.COOLING_WATER, 10, 4, 30),
            ("eq_P601", "eq_E302", "Cooling Water to Cooler", FluidType.COOLING_WATER, 8, 4, 30),
            # Flare header
            ("eq_V101", "eq_V701", "HP Flare", FluidType.FLARE_GAS, 8, 0.5, 80),
        ]

        for from_id, to_id, name, fluid, size, pressure, temp in connections:
            conn = ProcessConnection(
                name=name, from_equipment_id=from_id, to_equipment_id=to_id,
                fluid_type=fluid, pipe_size_inches=size,
                normal_pressure=pressure, normal_temperature=temp,
            )
            self.kg.add_entity(conn)

    # ── Failure Modes (FMEA) ───────────────────────

    def _build_failure_modes(self):
        """Build comprehensive FMEA database per equipment class."""
        fmea_data = [
            # ── Centrifugal Pump Failure Modes ──
            (EquipmentClass.PUMP, FailureMode.BEARING_FAILURE, FailureMechanism.WEAR,
             0.08, 8, 6, 3, "Increased vibration → temperature rise → seizure",
             "Vibration increase 2-4 weeks before failure; bearing temperature rise 1-2 weeks",
             [{"measurement": "vibration", "pattern": "increasing_trend", "lead_time_hours": 336},
              {"measurement": "temperature", "pattern": "increasing_trend", "lead_time_hours": 168}],
             [{"equipment_class": "heat_exchanger", "effect": "reduced_cooling_flow", "delay_hours": 0.5}]),

            (EquipmentClass.PUMP, FailureMode.SEAL_FAILURE, FailureMechanism.WEAR,
             0.12, 6, 7, 4, "External leakage, reduced discharge pressure",
             "Pressure fluctuation; may detect leak via gas/liquid detectors",
             [{"measurement": "pressure", "pattern": "decreasing_with_fluctuation", "lead_time_hours": 48},
              {"measurement": "flow", "pattern": "decreasing", "lead_time_hours": 24}],
             []),

            (EquipmentClass.PUMP, FailureMode.CAVITATION, FailureMechanism.EROSION_CAVITATION,
             0.06, 7, 5, 5, "Erosion damage, vibration, noise, reduced performance",
             "High-frequency vibration signature; suction pressure drops below NPSHr",
             [{"measurement": "vibration", "pattern": "high_frequency_spike", "lead_time_hours": 1},
              {"measurement": "pressure", "pattern": "suction_drop", "lead_time_hours": 0.5}],
             []),

            (EquipmentClass.PUMP, FailureMode.SHAFT_MISALIGNMENT, FailureMechanism.MISALIGNMENT,
             0.04, 5, 4, 6, "Increased vibration at 1x and 2x RPM, coupling wear",
             "Axial vibration increase at 2x RPM; temperature rise on coupling side bearing",
             [{"measurement": "vibration", "pattern": "2x_rpm_dominant", "lead_time_hours": 720}],
             []),

            # ── Centrifugal Compressor Failure Modes ──
            (EquipmentClass.COMPRESSOR, FailureMode.SURGE, FailureMechanism.OVERLOAD_MECHANICAL,
             0.03, 9, 3, 2, "Rapid flow reversal, severe vibration, potential mechanical damage",
             "Sudden flow drop with pressure oscillation; anti-surge valve should open",
             [{"measurement": "flow", "pattern": "sudden_drop_oscillation", "lead_time_hours": 0.01},
              {"measurement": "vibration", "pattern": "severe_spike", "lead_time_hours": 0.01},
              {"measurement": "pressure", "pattern": "rapid_oscillation", "lead_time_hours": 0.01}],
             [{"equipment_class": "heat_exchanger", "effect": "flow_reversal_damage", "delay_hours": 0.01}]),

            (EquipmentClass.COMPRESSOR, FailureMode.BEARING_FAILURE, FailureMechanism.WEAR,
             0.05, 9, 5, 3, "Vibration increase, temperature rise, potential catastrophic failure",
             "Radial vibration trend increase over weeks; bearing metal temperature rise",
             [{"measurement": "vibration", "pattern": "increasing_trend", "lead_time_hours": 504},
              {"measurement": "temperature", "pattern": "increasing_trend", "lead_time_hours": 168},
              {"measurement": "displacement", "pattern": "increasing_orbit", "lead_time_hours": 336}],
             [{"equipment_class": "separator", "effect": "loss_of_compression_no_gas_processing", "delay_hours": 0.1}]),

            (EquipmentClass.COMPRESSOR, FailureMode.FOULING, FailureMechanism.FOULING_COKING,
             0.07, 6, 7, 5, "Reduced efficiency, higher discharge temperature, reduced capacity",
             "Gradual polytropic efficiency drop; discharge temperature creep over months",
             [{"measurement": "temperature", "pattern": "slow_drift_up", "lead_time_hours": 2160},
              {"measurement": "power", "pattern": "increasing_for_same_throughput", "lead_time_hours": 1440}],
             []),

            # ── Heat Exchanger Failure Modes ──
            (EquipmentClass.HEAT_EXCHANGER, FailureMode.FOULING, FailureMechanism.FOULING_SCALING,
             0.15, 5, 8, 5, "Reduced heat transfer, increased pressure drop, higher outlet temperatures",
             "Outlet temperature drift; differential pressure increase over weeks/months",
             [{"measurement": "temperature", "pattern": "outlet_drift_up", "lead_time_hours": 720},
              {"measurement": "pressure", "pattern": "differential_increase", "lead_time_hours": 480}],
             [{"equipment_class": "compressor", "effect": "higher_suction_temperature", "delay_hours": 1}]),

            (EquipmentClass.HEAT_EXCHANGER, FailureMode.TUBE_FAILURE, FailureMechanism.CORROSION_PITTING,
             0.04, 8, 4, 3, "Cross-contamination between shell and tube fluids",
             "Product quality change; pressure differential change; may see in downstream analysis",
             [{"measurement": "pressure", "pattern": "shell_tube_equalization", "lead_time_hours": 0.5},
              {"measurement": "concentration", "pattern": "contamination_spike", "lead_time_hours": 1}],
             [{"equipment_class": "column", "effect": "feed_contamination", "delay_hours": 2}]),

            (EquipmentClass.HEAT_EXCHANGER, FailureMode.EXTERNAL_LEAKAGE, FailureMechanism.CORROSION_UNIFORM,
             0.03, 7, 5, 4, "Loss of containment, fire/explosion risk if hydrocarbon",
             "Visual inspection or gas detection",
             [], []),

            # ── Separator/Vessel Failure Modes ──
            (EquipmentClass.SEPARATOR, FailureMode.HIGH_OUTPUT, FailureMechanism.OPERATING_ERROR,
             0.06, 5, 5, 3, "Liquid carryover to gas outlet, floods downstream equipment",
             "Level rising; gas outlet quality deteriorating",
             [{"measurement": "level", "pattern": "rising_above_normal", "lead_time_hours": 0.5},
              {"measurement": "concentration", "pattern": "liquid_in_gas", "lead_time_hours": 1}],
             [{"equipment_class": "compressor", "effect": "liquid_ingestion_damage", "delay_hours": 0.5}]),

            (EquipmentClass.SEPARATOR, FailureMode.CORROSION, FailureMechanism.CORROSION_UNIFORM,
             0.02, 8, 3, 7, "Wall thinning, potential loss of containment",
             "Thickness monitoring via UT inspection; corrosion coupon trends",
             [{"measurement": "thickness", "pattern": "wall_thinning_trend", "lead_time_hours": 8760}],
             []),

            # ── Column Failure Modes ──
            (EquipmentClass.COLUMN, FailureMode.FOULING, FailureMechanism.FOULING_COKING,
             0.08, 6, 7, 5, "Increased pressure drop, reduced separation efficiency, flooding",
             "Differential pressure increase; product quality degradation",
             [{"measurement": "pressure", "pattern": "differential_increase", "lead_time_hours": 720},
              {"measurement": "concentration", "pattern": "off_spec_product", "lead_time_hours": 168}],
             []),

            (EquipmentClass.COLUMN, FailureMode.PLUGGED, FailureMechanism.FOULING_SCALING,
             0.04, 7, 5, 4, "Tray/packing blockage, loss of mass transfer",
             "Sudden pressure drop increase; level disturbance",
             [{"measurement": "pressure", "pattern": "sudden_dp_increase", "lead_time_hours": 2}],
             []),

            # ── Valve Failure Modes ──
            (EquipmentClass.VALVE, FailureMode.STUCK, FailureMechanism.CORROSION_UNIFORM,
             0.05, 6, 5, 4, "Valve fails to move to required position",
             "Stem travel feedback differs from signal; flow control loop oscillation",
             [{"measurement": "position", "pattern": "stuck_no_response", "lead_time_hours": 0.01}],
             []),

            (EquipmentClass.VALVE, FailureMode.INTERNAL_LEAKAGE, FailureMechanism.EROSION_FLOW,
             0.06, 5, 6, 5, "Passing when closed, cannot achieve tight shutoff",
             "Downstream pressure/flow when valve commanded closed",
             [{"measurement": "pressure", "pattern": "downstream_pressure_when_closed", "lead_time_hours": 0.1}],
             []),

            # ── Motor Failure Modes ──
            (EquipmentClass.MOTOR, FailureMode.INSULATION_FAILURE, FailureMechanism.OVERLOAD_THERMAL,
             0.03, 8, 4, 5, "Winding short circuit, ground fault, motor failure",
             "Current imbalance between phases; winding temperature rise",
             [{"measurement": "current", "pattern": "phase_imbalance", "lead_time_hours": 168},
              {"measurement": "temperature", "pattern": "winding_hotspot", "lead_time_hours": 72}],
             [{"equipment_class": "pump", "effect": "loss_of_drive", "delay_hours": 0.01}]),

            (EquipmentClass.MOTOR, FailureMode.OVERHEATING, FailureMechanism.OVERLOAD_ELECTRICAL,
             0.05, 6, 5, 3, "Reduced insulation life, potential winding failure",
             "Winding temperature trending up; ambient temperature contribution",
             [{"measurement": "temperature", "pattern": "increasing_trend", "lead_time_hours": 48},
              {"measurement": "current", "pattern": "above_rated", "lead_time_hours": 24}],
             []),

            # ── Tank Failure Modes ──
            (EquipmentClass.TANK, FailureMode.CORROSION, FailureMechanism.CORROSION_UNIFORM,
             0.03, 7, 4, 7, "Floor/shell thinning, potential leak to containment",
             "API 653 inspection findings; UT thickness readings; leak detection",
             [{"measurement": "thickness", "pattern": "wall_thinning", "lead_time_hours": 8760}],
             []),

            (EquipmentClass.TANK, FailureMode.HIGH_OUTPUT, FailureMechanism.OPERATING_ERROR,
             0.04, 7, 4, 2, "Overflow, environmental spill",
             "Level alarm activation; overflow detection",
             [{"measurement": "level", "pattern": "rising_towards_overflow", "lead_time_hours": 1}],
             []),
        ]

        for (eq_class, fm, mechanism, rate, severity, occurrence, detection,
             desc, sensor_desc, indicators, cascades) in fmea_data:
            rpn = severity * occurrence * detection
            fme = EquipmentFailureMode(
                name=f"{eq_class.value}_{fm.value}",
                equipment_class=eq_class,
                failure_mode=fm, failure_mechanism=mechanism,
                failure_rate_per_year=rate,
                severity=severity, occurrence=occurrence,
                detection_difficulty=detection, rpn=rpn,
                description=desc,
                detection_method=sensor_desc,
                sensor_indicators=indicators,
                cascade_targets=cascades,
                preventive_actions=[f"Regular inspection per {eq_class.value} maintenance plan"],
                corrective_actions=[f"Replace/repair affected {fm.value} component"],
            )
            self.kg.add_entity(fme)

    # ── Operating Envelopes ─────────────────────────

    def _build_operating_envelopes(self):
        envelopes = [
            ("eq_V101", "normal", [
                {"measurement_type": "pressure", "unit": "barg", "min_safe": 5, "min_normal": 55, "optimal": 65, "max_normal": 72, "max_safe": 75, "rate_of_change_limit": 2.0},
                {"measurement_type": "temperature", "unit": "°C", "min_safe": 30, "min_normal": 70, "optimal": 85, "max_normal": 100, "max_safe": 120, "rate_of_change_limit": 5.0},
                {"measurement_type": "level", "unit": "%", "min_safe": 10, "min_normal": 30, "optimal": 50, "max_normal": 75, "max_safe": 90},
            ]),
            ("eq_C201", "normal", [
                {"measurement_type": "pressure_suction", "unit": "barg", "min_safe": 3, "min_normal": 6, "optimal": 8, "max_normal": 12, "max_safe": 20},
                {"measurement_type": "pressure_discharge", "unit": "barg", "min_safe": 20, "min_normal": 30, "optimal": 35, "max_normal": 40, "max_safe": 45},
                {"measurement_type": "temperature_discharge", "unit": "°C", "min_safe": 40, "min_normal": 90, "optimal": 120, "max_normal": 140, "max_safe": 150},
                {"measurement_type": "vibration", "unit": "mm/s", "min_safe": 0, "min_normal": 0, "optimal": 3, "max_normal": 10, "max_safe": 18},
                {"measurement_type": "speed", "unit": "RPM", "min_safe": 8000, "min_normal": 10000, "optimal": 11500, "max_normal": 12000, "max_safe": 12500},
            ]),
            ("eq_C201", "startup", [
                {"measurement_type": "speed", "unit": "RPM", "min_safe": 0, "min_normal": 0, "optimal": 5000, "max_normal": 11500, "max_safe": 12000, "rate_of_change_limit": 500},
                {"measurement_type": "vibration", "unit": "mm/s", "min_safe": 0, "min_normal": 0, "optimal": 5, "max_normal": 15, "max_safe": 20},
            ]),
            ("eq_P301", "normal", [
                {"measurement_type": "flow", "unit": "m³/h", "min_safe": 30, "min_normal": 100, "optimal": 250, "max_normal": 400, "max_safe": 450},
                {"measurement_type": "pressure_discharge", "unit": "barg", "min_safe": 5, "min_normal": 8, "optimal": 10, "max_normal": 12, "max_safe": 14},
                {"measurement_type": "vibration", "unit": "mm/s", "min_safe": 0, "min_normal": 0, "optimal": 2, "max_normal": 6, "max_safe": 12},
                {"measurement_type": "temperature_bearing", "unit": "°C", "min_safe": 20, "min_normal": 30, "optimal": 50, "max_normal": 70, "max_safe": 85},
            ]),
            ("eq_T301", "normal", [
                {"measurement_type": "pressure", "unit": "barg", "min_safe": 50, "min_normal": 65, "optimal": 72, "max_normal": 78, "max_safe": 85},
                {"measurement_type": "temperature_top", "unit": "°C", "min_safe": 25, "min_normal": 35, "optimal": 45, "max_normal": 55, "max_safe": 70},
                {"measurement_type": "level", "unit": "%", "min_safe": 10, "min_normal": 30, "optimal": 50, "max_normal": 75, "max_safe": 90},
                {"measurement_type": "h2s_outlet", "unit": "ppm", "min_safe": 0, "min_normal": 0, "optimal": 1, "max_normal": 3, "max_safe": 10},
            ]),
        ]

        for eq_id, state, params in envelopes:
            env = OperatingEnvelope(
                name=f"{eq_id}_{state}", equipment_id=eq_id, state=state, parameters=params,
            )
            self.kg.add_entity(env)

    # ── Physics Rules ──────────────────────────────

    def _build_physics_rules(self):
        rules = [
            PhysicsRule(
                name="pump_head_vs_flow",
                equation="H = H_shutoff - k * Q^2",
                description="Centrifugal pump head decreases with flow (pump curve). As flow increases, discharge pressure drops.",
                equipment_class=EquipmentClass.PUMP,
                input_variables=["flow_rate"],
                output_variable="discharge_pressure",
                correlation_type="inverse_quadratic",
                correlation_strength=0.95,
            ),
            PhysicsRule(
                name="pump_power_vs_flow",
                equation="P = rho * g * Q * H / (eta_pump * eta_motor)",
                description="Pump power consumption proportional to flow × head. Motor current tracks power demand.",
                equipment_class=EquipmentClass.PUMP,
                input_variables=["flow_rate", "discharge_pressure"],
                output_variable="power_consumption",
                correlation_type="proportional",
                correlation_strength=0.92,
            ),
            PhysicsRule(
                name="pump_temperature_vs_power",
                description="Bearing temperature rises with power load and reduces with cooling. Thermal lag of ~15 minutes.",
                equipment_class=EquipmentClass.PUMP,
                input_variables=["power_consumption", "ambient_temperature"],
                output_variable="bearing_temperature",
                correlation_type="proportional",
                correlation_strength=0.85,
                time_lag_seconds=900,
            ),
            PhysicsRule(
                name="compressor_discharge_temp",
                equation="T2 = T1 * (P2/P1)^((k-1)/(k*eta_p))",
                description="Polytropic compression: discharge temperature determined by pressure ratio, suction temperature, and efficiency.",
                equipment_class=EquipmentClass.COMPRESSOR,
                input_variables=["suction_temperature", "suction_pressure", "discharge_pressure"],
                output_variable="discharge_temperature",
                correlation_type="nonlinear",
                correlation_strength=0.97,
            ),
            PhysicsRule(
                name="compressor_surge_proximity",
                equation="surge_margin = (Q - Q_surge) / Q_surge * 100",
                description="Compressor approaches surge when flow drops below surge line. Anti-surge valve opens to protect.",
                equipment_class=EquipmentClass.COMPRESSOR,
                input_variables=["flow_rate", "discharge_pressure"],
                output_variable="surge_margin",
                correlation_type="threshold",
                correlation_strength=0.99,
            ),
            PhysicsRule(
                name="compressor_power_vs_flow_speed",
                equation="P = m_dot * delta_h = f(N^3, rho)",
                description="Compressor power follows fan laws: proportional to speed cubed. Gas density affects loading.",
                equipment_class=EquipmentClass.COMPRESSOR,
                input_variables=["speed", "flow_rate", "gas_density"],
                output_variable="power_consumption",
                correlation_type="nonlinear",
                correlation_strength=0.93,
            ),
            PhysicsRule(
                name="hx_lmtd_fouling",
                equation="Q = U * A * LMTD; U decreases as fouling increases",
                description="Heat exchanger duty drops with fouling. Outlet temp rises on hot side, drops on cold side.",
                equipment_class=EquipmentClass.HEAT_EXCHANGER,
                input_variables=["hot_inlet_temp", "cold_inlet_temp", "fouling_factor"],
                output_variable="hot_outlet_temp",
                correlation_type="proportional",
                correlation_strength=0.9,
                time_lag_seconds=300,
            ),
            PhysicsRule(
                name="hx_dp_vs_fouling",
                description="Pressure drop across exchanger increases as fouling builds up. Exponential relationship.",
                equipment_class=EquipmentClass.HEAT_EXCHANGER,
                input_variables=["flow_rate", "fouling_factor"],
                output_variable="differential_pressure",
                correlation_type="proportional",
                correlation_strength=0.88,
            ),
            PhysicsRule(
                name="separator_level_balance",
                equation="dL/dt = (Q_in - Q_oil - Q_water - Q_gas) / A_cross_section",
                description="Separator level is mass balance: inflow minus outflows. Level rises if outlets restricted.",
                equipment_class=EquipmentClass.SEPARATOR,
                input_variables=["inlet_flow", "oil_outlet_flow", "water_outlet_flow", "gas_outlet_flow"],
                output_variable="level",
                correlation_type="integral",
                correlation_strength=0.98,
            ),
            PhysicsRule(
                name="column_dp_vs_loading",
                equation="DP = k * (vapor_rate)^2 * (liquid_rate)^0.5",
                description="Column differential pressure increases with vapor and liquid loading. Flooding if DP exceeds limit.",
                equipment_class=EquipmentClass.COLUMN,
                input_variables=["vapor_flow", "liquid_flow"],
                output_variable="differential_pressure",
                correlation_type="nonlinear",
                correlation_strength=0.91,
            ),
            PhysicsRule(
                name="pipe_pressure_drop",
                equation="dP = f * (L/D) * (rho * v^2 / 2)",
                description="Darcy-Weisbach: pressure drop along pipe proportional to velocity squared, length, and friction.",
                equipment_class=EquipmentClass.PIPE,
                input_variables=["flow_rate", "pipe_diameter", "pipe_length", "fluid_viscosity"],
                output_variable="pressure_drop",
                correlation_type="proportional",
                correlation_strength=0.94,
            ),
            PhysicsRule(
                name="vibration_vs_imbalance",
                description="Vibration at 1x RPM frequency increases with rotor imbalance. Phase angle indicates heavy spot location.",
                equipment_class=EquipmentClass.PUMP,
                input_variables=["rotor_imbalance", "speed"],
                output_variable="vibration_1x",
                correlation_type="proportional",
                correlation_strength=0.92,
            ),
            PhysicsRule(
                name="motor_current_vs_load",
                description="Motor current proportional to shaft load. Current spike during startup (6-8x rated).",
                equipment_class=EquipmentClass.MOTOR,
                input_variables=["shaft_load", "power_factor"],
                output_variable="current",
                correlation_type="proportional",
                correlation_strength=0.95,
            ),
        ]

        for rule in rules:
            self.kg.add_entity(rule)

    # ── Maintenance Procedures ─────────────────────

    def _build_maintenance_procedures(self):
        procedures = [
            MaintenanceProcedure(
                name="Centrifugal Pump PM — Quarterly",
                equipment_class=EquipmentClass.PUMP,
                procedure_type="preventive",
                interval_calendar_days=90,
                estimated_duration_hours=4,
                required_skills=["mechanical_technician", "instrument_technician"],
                required_tools=["vibration_analyzer", "alignment_laser", "torque_wrench"],
                required_spares=["mechanical_seal_kit", "bearing_set", "coupling_element"],
                steps=[
                    "1. Isolate pump (LOTO procedure, blind flanges if required)",
                    "2. Record baseline vibration readings (overall + spectrum)",
                    "3. Check mechanical seal for leakage (visual + drip rate)",
                    "4. Record bearing temperatures (DE and NDE)",
                    "5. Check coupling alignment (laser alignment tool)",
                    "6. Inspect coupling element for wear/cracking",
                    "7. Check foundation bolts for tightness",
                    "8. Verify lube oil level and quality (sample for lab analysis)",
                    "9. Check suction strainer differential pressure",
                    "10. Verify instrument readings (pressure gauges, temperature)",
                    "11. Run performance test: record head, flow, power at 3 operating points",
                    "12. Update maintenance records in CMMS",
                ],
                safety_precautions=[
                    "LOTO before any intrusive work",
                    "Verify zero energy state (pressure, electrical, stored energy)",
                    "Wear appropriate PPE (face shield if seal area work)",
                    "Fire watch if in hazardous area",
                ],
                permits_required=["cold_work_permit"],
            ),
            MaintenanceProcedure(
                name="Compressor Predictive Monitoring — Monthly",
                equipment_class=EquipmentClass.COMPRESSOR,
                procedure_type="predictive",
                interval_calendar_days=30,
                estimated_duration_hours=2,
                required_skills=["vibration_analyst_cat_II", "instrument_technician"],
                required_tools=["vibration_analyzer", "infrared_camera", "ultrasonic_detector"],
                steps=[
                    "1. Collect vibration data: DE bearing X-Y, NDE bearing X-Y, axial",
                    "2. Collect vibration spectrum (0-1000 Hz, 8192 lines resolution)",
                    "3. Collect shaft orbit plots from proximity probes",
                    "4. Record bearing metal temperatures",
                    "5. Perform infrared thermography scan (bearings, coupling, casing)",
                    "6. Ultrasonic scan for internal defects / valve leakage",
                    "7. Record operating parameters (P, T, flow, speed, power)",
                    "8. Calculate polytropic efficiency and compare to baseline",
                    "9. Check anti-surge valve stroke and response time",
                    "10. Analyze trends and update predictive maintenance forecast",
                ],
                safety_precautions=["No LOTO required — online monitoring", "Maintain safe distance from rotating parts"],
            ),
            MaintenanceProcedure(
                name="Heat Exchanger Inspection — Annual",
                equipment_class=EquipmentClass.HEAT_EXCHANGER,
                procedure_type="preventive",
                interval_calendar_days=365,
                estimated_duration_hours=48,
                required_skills=["inspection_engineer", "mechanical_fitter", "nde_technician"],
                required_tools=["ut_thickness_gauge", "eddy_current_tester", "borescope"],
                required_spares=["gasket_set", "tube_plugs", "bolting_set"],
                steps=[
                    "1. Isolate, drain, and gas-free exchanger (confined space entry procedures)",
                    "2. Remove channel head covers / floating head",
                    "3. Visual inspection of tubes, tubesheet, baffles",
                    "4. UT thickness measurements on shell and nozzles (grid pattern)",
                    "5. Eddy current testing on tube bundle (100% if critical, 20% sample otherwise)",
                    "6. Measure tube-to-tubesheet joint integrity",
                    "7. Hydrostatic test if tube plugging or repairs performed",
                    "8. Clean tube bundle (hydro-blasting or chemical cleaning)",
                    "9. Reassemble with new gaskets, torque to specification",
                    "10. Pressure test and reinstate",
                ],
                safety_precautions=[
                    "Confined space entry permit required",
                    "Gas test before entry (O2, LEL, H2S, CO)",
                    "Continuous gas monitoring during work",
                    "Rescue team on standby",
                ],
                permits_required=["confined_space_entry", "cold_work_permit"],
            ),
            MaintenanceProcedure(
                name="Safety Valve Testing — Annual Proof Test",
                equipment_class=EquipmentClass.VALVE,
                procedure_type="preventive",
                interval_calendar_days=365,
                estimated_duration_hours=6,
                required_skills=["instrument_technician", "process_engineer"],
                steps=[
                    "1. Plan test window with operations (may require online or offline test)",
                    "2. If offline: isolate PSV, remove for bench testing",
                    "3. Test pop pressure on test bench (must be within ±3% of set pressure)",
                    "4. Test reseat pressure (must close cleanly, no chatter)",
                    "5. Check for seat leakage (bubble test)",
                    "6. Recondition if required (lap seats, replace springs)",
                    "7. Reset and certify",
                    "8. Reinstall with new gaskets",
                    "9. Update SIF proof test records",
                ],
                permits_required=["safety_system_bypass_permit"],
            ),
        ]

        for proc in procedures:
            self.kg.add_entity(proc)

    # ── Safety Interlocks ──────────────────────────

    def _build_safety_interlocks(self):
        interlocks = [
            SafetyInterlock(
                name="V-101 High Level Trip", sif_number="SIF-101-01",
                sil_level=SILLevel.SIL_2,
                initiator_tag="LT-101A", initiator_type=MeasurementType.LEVEL,
                trip_setpoint=90, trip_direction="high", voting="1oo2",
                final_element_tag="XV-101", final_element_action="close",
                protected_equipment_id="eq_V101",
                hazard_description="Liquid carryover to compressor causing mechanical damage",
                consequence_of_failure="Compressor damage, potential gas release, $2M+ repair cost",
                logic_description="IF LT-101A OR LT-101B > 90% THEN close XV-101 inlet valve",
                proof_test_interval_months=12,
            ),
            SafetyInterlock(
                name="C-201 High Vibration Trip", sif_number="SIF-201-01",
                sil_level=SILLevel.SIL_1,
                initiator_tag="VT-201DE", initiator_type=MeasurementType.VIBRATION,
                trip_setpoint=18, trip_direction="high", voting="2oo3",
                final_element_tag="C-201", final_element_action="trip",
                protected_equipment_id="eq_C201",
                hazard_description="Catastrophic bearing/shaft failure during high vibration",
                consequence_of_failure="Compressor destruction, potential casing breach, fire risk",
                logic_description="IF VT-201DE AND VT-201NDE > 18 mm/s THEN trip compressor, close suction/discharge valves",
            ),
            SafetyInterlock(
                name="C-201 Anti-Surge", sif_number="SIF-201-02",
                sil_level=SILLevel.SIL_1,
                initiator_tag="FT-201", initiator_type=MeasurementType.FLOW,
                trip_setpoint=0, trip_direction="low", voting="1oo1",
                final_element_tag="FV-201ASV", final_element_action="open",
                protected_equipment_id="eq_C201",
                hazard_description="Compressor surge causing mechanical damage",
                logic_description="Anti-surge controller opens recycle valve when operating point approaches surge line",
            ),
            SafetyInterlock(
                name="C-201 High Discharge Temperature", sif_number="SIF-201-03",
                sil_level=SILLevel.SIL_1,
                initiator_tag="TT-201D", initiator_type=MeasurementType.TEMPERATURE,
                trip_setpoint=150, trip_direction="high", voting="1oo1",
                final_element_tag="C-201", final_element_action="trip",
                protected_equipment_id="eq_C201",
                hazard_description="Overheating damages seals, bearings, and may ignite lube oil",
            ),
            SafetyInterlock(
                name="T-301 High Pressure", sif_number="SIF-301-01",
                sil_level=SILLevel.SIL_2,
                initiator_tag="PT-301T", initiator_type=MeasurementType.PRESSURE,
                trip_setpoint=85, trip_direction="high", voting="1oo2",
                final_element_tag="XV-101", final_element_action="close",
                protected_equipment_id="eq_T301",
                hazard_description="Column overpressure leading to loss of containment",
                logic_description="IF PT-301 > 85 barg THEN close inlet, open relief to flare",
            ),
            SafetyInterlock(
                name="TK-101 High Level", sif_number="SIF-101-02",
                sil_level=SILLevel.SIL_1,
                initiator_tag="LT-101T", initiator_type=MeasurementType.LEVEL,
                trip_setpoint=95, trip_direction="high", voting="1oo1",
                final_element_tag="XV-TK101", final_element_action="close",
                protected_equipment_id="eq_TK101",
                hazard_description="Tank overflow, environmental spill",
            ),
        ]

        for interlock in interlocks:
            self.kg.add_entity(interlock)

    # ── Control Loops ──────────────────────────────

    def _build_control_loops(self):
        loops = [
            ControlLoop(
                name="V-101 Level Control", loop_number="LIC-101",
                sensor_tag="LT-101A", process_variable=MeasurementType.LEVEL,
                controller_tag="LIC-101", setpoint=50, setpoint_unit="%",
                output_tag="LV-101", output_type="valve_position",
                fail_action="open", kp=2.0, ki=0.5, kd=0.1,
            ),
            ControlLoop(
                name="V-101 Pressure Control", loop_number="PIC-101",
                sensor_tag="PT-101", process_variable=MeasurementType.PRESSURE,
                controller_tag="PIC-101", setpoint=65, setpoint_unit="barg",
                output_tag="PV-101", fail_action="open", kp=1.5, ki=0.3,
            ),
            ControlLoop(
                name="Amine Flow Control", loop_number="FIC-301",
                sensor_tag="FT-301", process_variable=MeasurementType.FLOW,
                controller_tag="FIC-301", setpoint=250, setpoint_unit="m³/h",
                output_tag="FV-301", fail_action="open", kp=1.0, ki=0.2,
            ),
            ControlLoop(
                name="Compressor Speed Control", loop_number="SIC-201",
                sensor_tag="PT-201D", process_variable=MeasurementType.PRESSURE,
                controller_tag="SIC-201", setpoint=35, setpoint_unit="barg",
                output_tag="C-201_SPEED", output_type="speed_setpoint",
                fail_action="last", kp=0.5, ki=0.05,
            ),
            ControlLoop(
                name="Aftercooler Temperature Control", loop_number="TIC-201",
                sensor_tag="TT-201HO", process_variable=MeasurementType.TEMPERATURE,
                controller_tag="TIC-201", setpoint=45, setpoint_unit="°C",
                output_tag="TV-201", fail_action="open", kp=1.2, ki=0.4,
            ),
        ]

        for loop in loops:
            self.kg.add_entity(loop)

    # ── Cascade Rules ──────────────────────────────

    def _build_cascade_rules(self):
        cascades = [
            FailureCascadeRule(
                name="Pump failure → Loss of cooling → Exchanger overtemperature",
                source_equipment_class=EquipmentClass.PUMP,
                source_failure_mode=FailureMode.BREAKDOWN,
                target_equipment_class=EquipmentClass.HEAT_EXCHANGER,
                cascade_effect="loss_of_cooling_flow",
                probability=0.9, delay_hours=0.25, severity_factor=0.7,
                sensor_signature={"flow": "sudden_drop_to_zero", "temperature": "gradual_rise"},
                description="CW pump trip causes loss of cooling water flow. Exchanger hot-side outlet temperature rises within 15 minutes.",
            ),
            FailureCascadeRule(
                name="Exchanger fouling → High suction temperature → Compressor surge",
                source_equipment_class=EquipmentClass.HEAT_EXCHANGER,
                source_failure_mode=FailureMode.FOULING,
                target_equipment_class=EquipmentClass.COMPRESSOR,
                cascade_effect="higher_suction_temperature_reduced_capacity",
                probability=0.6, delay_hours=48, severity_factor=0.5,
                sensor_signature={"temperature": "gradual_rise", "flow": "gradual_decrease"},
                description="Aftercooler fouling increases suction temperature, reducing gas density and compressor capacity. May push operating point toward surge.",
            ),
            FailureCascadeRule(
                name="Separator liquid carryover → Compressor liquid damage",
                source_equipment_class=EquipmentClass.SEPARATOR,
                source_failure_mode=FailureMode.HIGH_OUTPUT,
                target_equipment_class=EquipmentClass.COMPRESSOR,
                cascade_effect="liquid_ingestion",
                probability=0.95, delay_hours=0.01, severity_factor=1.0,
                sensor_signature={"level": "rising_above_alarm", "vibration": "severe_spike"},
                description="Liquid carryover from separator enters compressor suction. Immediate severe vibration and potential mechanical damage.",
            ),
            FailureCascadeRule(
                name="Amine pump failure → Loss of sweetening → Off-spec gas",
                source_equipment_class=EquipmentClass.PUMP,
                source_failure_mode=FailureMode.BREAKDOWN,
                target_equipment_class=EquipmentClass.COLUMN,
                cascade_effect="loss_of_amine_circulation",
                probability=0.95, delay_hours=0.1, severity_factor=0.8,
                sensor_signature={"flow": "drop_to_zero", "concentration": "h2s_rising"},
                description="Amine pump trip stops lean amine flow to contactor. H2S in sweet gas rises within minutes, off-spec gas to pipeline.",
            ),
            FailureCascadeRule(
                name="Instrument air compressor failure → Control valve failure",
                source_equipment_class=EquipmentClass.COMPRESSOR,
                source_failure_mode=FailureMode.BREAKDOWN,
                target_equipment_class=EquipmentClass.VALVE,
                cascade_effect="loss_of_instrument_air",
                probability=0.85, delay_hours=0.5, severity_factor=0.9,
                sensor_signature={"pressure": "instrument_air_dropping"},
                description="Loss of instrument air causes control valves to fail to their safe position (open/close). Plant-wide control disruption.",
            ),
            FailureCascadeRule(
                name="Cooling water pump trip → Multiple exchanger impact",
                source_equipment_class=EquipmentClass.PUMP,
                source_failure_mode=FailureMode.FAIL_TO_START,
                target_equipment_class=EquipmentClass.HEAT_EXCHANGER,
                cascade_effect="loss_of_cooling_multiple_exchangers",
                probability=0.95, delay_hours=0.08, severity_factor=0.8,
                sensor_signature={"flow": "cw_flow_drop", "temperature": "multiple_exchangers_rising"},
                description="Single cooling water pump serves multiple exchangers. Trip cascades to all downstream process temperatures.",
            ),
        ]

        for rule in cascades:
            self.kg.add_entity(rule)

    # ── Build All Relationships ────────────────────

    def _build_relationships(self):
        kg = self.kg

        # Structural: Area contains equipment
        area_equipment = {
            "area_100": ["eq_V101", "eq_V102", "eq_TK101", "eq_XV101"],
            "area_200": ["eq_C201", "eq_C202", "eq_E201", "eq_V201", "eq_V202"],
            "area_300": ["eq_T301", "eq_T302", "eq_P301", "eq_P301B", "eq_E301", "eq_E302", "eq_FV301"],
            "area_400": ["eq_T401", "eq_E401"],
            "area_500": ["eq_T501", "eq_E501"],
            "area_600": ["eq_P601", "eq_C601"],
            "area_700": ["eq_V701"],
        }

        for area_id, eq_ids in area_equipment.items():
            for eq_id in eq_ids:
                kg.add_relationship(area_id, eq_id, RelationshipType.CONTAINS)
                kg.add_relationship(eq_id, area_id, RelationshipType.LOCATED_IN)

        # Process: Equipment feeds/receives
        for conn in kg.connections.values():
            if conn.from_equipment_id and conn.to_equipment_id:
                kg.add_relationship(conn.from_equipment_id, conn.to_equipment_id,
                                    RelationshipType.FEEDS_TO, fluid=conn.fluid_type.value)
                kg.add_relationship(conn.to_equipment_id, conn.from_equipment_id,
                                    RelationshipType.RECEIVES_FROM)

        # Instruments → Equipment
        for inst in kg.instruments.values():
            if inst.equipment_id:
                kg.add_relationship(inst.id, inst.equipment_id, RelationshipType.MEASURES)
                kg.add_relationship(inst.equipment_id, inst.id, RelationshipType.PART_OF)

        # Safety interlocks → Equipment
        for sif in kg.safety_interlocks.values():
            if sif.protected_equipment_id:
                kg.add_relationship(sif.id, sif.protected_equipment_id, RelationshipType.PROVIDES_PROTECTION)
                kg.add_relationship(sif.protected_equipment_id, sif.id, RelationshipType.PROTECTED_BY)

        # Cascade relationships between equipment
        for conn in kg.connections.values():
            if conn.from_equipment_id and conn.to_equipment_id:
                kg.add_relationship(conn.from_equipment_id, conn.to_equipment_id,
                                    RelationshipType.CASCADES_TO, probability=0.7)

        # Standby relationship
        kg.add_relationship("eq_P301", "eq_P301B", RelationshipType.INTERLOCKED_WITH,
                            relationship="active_standby")

        # Utility relationships
        kg.add_relationship("eq_P601", "eq_E201", RelationshipType.FEEDS_TO, service="cooling_water")
        kg.add_relationship("eq_C601", "eq_FV301", RelationshipType.FEEDS_TO, service="instrument_air")

        logger.info(f"Built {len(kg.relationships)} relationships")
