#!/usr/bin/env python3
"""
PlantCopilot Data Foundation — Practical Usage Guide
=====================================================

This guide shows you exactly how to use every component with real code.

TABLE OF CONTENTS:
  1. Quick Start — Build KG + Generate Data (5 minutes)
  2. Knowledge Graph Queries — Find equipment, sensors, failure modes
  3. Failure Cascade Analysis — Trace failure propagation paths
  4. Synthetic Data Generation — Normal, degradation, failure datasets
  5. ML Model Training — Anomaly detection with generated data
  6. LLM Integration (RAG) — Ground PlantCopilot chat with KG
  7. Root Cause Analysis — Use KG for automated RCA
  8. Custom Plant Setup — Add your own equipment
  9. Production Deployment — API integration patterns

Run each section independently:
  python usage_guide.py --section 1
  python usage_guide.py --section 2
  ... etc.

Or run all:
  python usage_guide.py --all
"""

import sys
import json
import csv
import time
from pathlib import Path
from datetime import datetime, timedelta
from collections import defaultdict

sys.path.insert(0, str(Path(__file__).parent))

from knowledge_graph.ontology import *
from knowledge_graph.builder import PlantKnowledgeGraph, OilGasDomainBuilder
from synthetic_data.generator import (
    SyntheticDataGenerator, GeneratorConfig, ScenarioType,
    ScenarioGenerator, ScenarioDefinition,
)


def print_header(title):
    print(f"\n{'='*70}")
    print(f"  {title}")
    print(f"{'='*70}\n")


def print_section(title):
    print(f"\n--- {title} ---\n")


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
#  SECTION 1: Quick Start
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

def section_1_quick_start():
    print_header("SECTION 1: Quick Start — Build KG + Generate Data")

    # Step 1: Build Knowledge Graph (takes <1 second)
    print("Step 1: Building Knowledge Graph...")
    kg = PlantKnowledgeGraph()
    builder = OilGasDomainBuilder(kg)
    builder.build_reference_plant()

    stats = kg.stats()
    print(f"  ✓ Built: {stats['total_entities']} entities, {stats['relationships']} relationships")
    print(f"  ✓ Equipment: {stats['equipment']}")
    print(f"  ✓ Sensors: {stats['instruments']}")
    print(f"  ✓ Failure modes: {stats['failure_modes']}")
    print(f"  ✓ Physics rules: {stats['physics_rules']}")

    # Step 2: Generate synthetic data
    print("\nStep 2: Generating synthetic data (1 hour, quick test)...")
    config = GeneratorConfig(
        duration_hours=1.0,
        sample_interval_seconds=30,
        seed=42,
    )
    generator = SyntheticDataGenerator(kg, config)
    output_dir = "./output_guide"
    gen_stats = generator.generate_all_datasets(output_dir)

    print(f"  ✓ Generated: {gen_stats['total_samples']:,} samples")
    print(f"  ✓ Scenarios: {gen_stats['total_scenarios']}")
    print(f"  ✓ Labels: {gen_stats['label_distribution']}")

    # Step 3: Check output files
    print("\nStep 3: Output files created:")
    for f in sorted(Path(output_dir).glob("*")):
        size = f.stat().st_size
        size_str = f"{size/1024:.1f} KB" if size < 1_000_000 else f"{size/1_000_000:.1f} MB"
        print(f"  {f.name:<42} {size_str:>10}")

    print("\n✅ Quick start complete! You now have:")
    print("   - A knowledge graph with full oil & gas domain knowledge")
    print("   - Labeled sensor data for ML training")
    print("   - Train/val/test splits ready to go")

    return kg


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
#  SECTION 2: Knowledge Graph Queries
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

def section_2_kg_queries(kg=None):
    print_header("SECTION 2: Knowledge Graph Queries")

    if kg is None:
        kg = PlantKnowledgeGraph()
        OilGasDomainBuilder(kg).build_reference_plant()

    # ── Query 1: Look up equipment by tag ──
    print_section("Query 1: Look up equipment by tag")
    print("  Code: kg.get_by_tag('C-201')")
    eq = kg.get_by_tag("C-201")
    if eq:
        print(f"  Result:")
        print(f"    Tag:          {eq.tag}")
        print(f"    Name:         {eq.name}")
        print(f"    Class:        {eq.equipment_class.value}")
        print(f"    Subclass:     {eq.equipment_subclass.value if eq.equipment_subclass else 'N/A'}")
        print(f"    Criticality:  {eq.criticality.value}")
        print(f"    Design P/T:   {eq.design_pressure} {eq.design_pressure_unit} / {eq.design_temperature} {eq.design_temperature_unit}")
        print(f"    Rated Power:  {eq.rated_power_kw} kW")
        print(f"    Rated Speed:  {eq.rated_speed_rpm} RPM")
        print(f"    Driver:       {eq.driver_type}")
        print(f"    MTBF:         {eq.mtbf_hours} hours")
        print(f"    Material:     {eq.material_of_construction}")
        print(f"    Safety:       SIL {eq.sil_level.value}, Critical={eq.is_safety_critical}")
        print(f"    Strategy:     {eq.maintenance_strategy.value}")

    # ── Query 2: Get all instruments for an equipment ──
    print_section("Query 2: Get all instruments for C-201")
    print("  Code: kg.get_instruments_for_equipment(eq.id)")
    instruments = kg.get_instruments_for_equipment(eq.id)
    print(f"  Found {len(instruments)} instruments:")
    for inst in instruments:
        alarms = []
        if inst.low_low_alarm is not None: alarms.append(f"LL={inst.low_low_alarm}")
        if inst.low_alarm is not None: alarms.append(f"L={inst.low_alarm}")
        if inst.high_alarm is not None: alarms.append(f"H={inst.high_alarm}")
        if inst.high_high_alarm is not None: alarms.append(f"HH={inst.high_high_alarm}")
        alarm_str = f" Alarms: [{', '.join(alarms)}]" if alarms else ""
        print(f"    {inst.tag:<12} {inst.measurement_type.value:<14} {inst.range_min}-{inst.range_max} {inst.unit:<6}{alarm_str}")

    # ── Query 3: Get all equipment by class ──
    print_section("Query 3: Get all pumps in the plant")
    print("  Code: kg.get_equipment_by_class(EquipmentClass.PUMP)")
    pumps = kg.get_equipment_by_class(EquipmentClass.PUMP)
    for p in pumps:
        print(f"    {p.tag:<10} {p.name:<35} {p.criticality.value:<10} {p.operating_state.value}")

    # ── Query 4: Get failure modes for equipment class ──
    print_section("Query 4: Get failure modes for pumps (FMEA)")
    print("  Code: kg.get_failure_modes_for_class(EquipmentClass.PUMP)")
    fms = kg.get_failure_modes_for_class(EquipmentClass.PUMP)
    for fm in fms:
        print(f"    {fm.failure_mode.value:<25} RPN={fm.rpn:<5} Rate={fm.failure_rate_per_year}/yr")
        print(f"      Mechanism: {fm.failure_mechanism.value}")
        print(f"      Effect:    {fm.description}")
        if fm.sensor_indicators:
            for ind in fm.sensor_indicators:
                print(f"      Sensor:    {ind['measurement']} → {ind['pattern']} (lead time: {ind['lead_time_hours']}h)")
        print()

    # ── Query 5: Get upstream/downstream equipment ──
    print_section("Query 5: Process flow — what connects to C-201?")
    print("  Code: kg.get_upstream(eq.id) / kg.get_downstream(eq.id)")
    upstream = kg.get_upstream(eq.id)
    downstream = kg.get_downstream(eq.id)
    print(f"  Upstream (feeds INTO C-201):")
    for u in upstream:
        print(f"    ← {u.tag}: {u.name}")
    print(f"  Downstream (C-201 feeds TO):")
    for d in downstream:
        print(f"    → {d.tag}: {d.name}")

    # ── Query 6: Safety functions ──
    print_section("Query 6: Safety interlocks protecting C-201")
    print("  Code: kg.get_safety_functions(eq.id)")
    sifs = kg.get_safety_functions(eq.id)
    for sif in sifs:
        print(f"    {sif.sif_number} (SIL {sif.sil_level.value})")
        print(f"      Initiator:  {sif.initiator_tag} {sif.trip_direction} > {sif.trip_setpoint}")
        print(f"      Action:     {sif.final_element_tag} → {sif.final_element_action}")
        print(f"      Logic:      {sif.logic_description}")
        print(f"      Hazard:     {sif.hazard_description}")
        print()

    # ── Query 7: Iterate all equipment with health context ──
    print_section("Query 7: Plant overview — all equipment with sensor count")
    print("  Code: iterate kg.equipment.values()")
    print(f"  {'Tag':<10} {'Name':<35} {'Class':<18} {'Crit.':<10} {'Sensors':<8} {'SIFs'}")
    print(f"  {'-'*10} {'-'*35} {'-'*18} {'-'*10} {'-'*8} {'-'*5}")
    for eq_item in kg.equipment.values():
        n_sensors = len(kg.get_instruments_for_equipment(eq_item.id))
        n_sifs = len(kg.get_safety_functions(eq_item.id))
        print(f"  {eq_item.tag:<10} {eq_item.name:<35} {eq_item.equipment_class.value:<18} {eq_item.criticality.value:<10} {n_sensors:<8} {n_sifs}")

    return kg


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
#  SECTION 3: Failure Cascade Analysis
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

def section_3_cascade_analysis(kg=None):
    print_header("SECTION 3: Failure Cascade Analysis")

    if kg is None:
        kg = PlantKnowledgeGraph()
        OilGasDomainBuilder(kg).build_reference_plant()

    # ── Cascade paths from a critical equipment ──
    print_section("Cascade paths from V-101 (Inlet Separator)")
    print("  Code: kg.get_cascade_paths('eq_V101', max_depth=4)")
    print("  This traces: 'If V-101 fails, what else could be affected?'\n")

    paths = kg.get_cascade_paths("eq_V101", max_depth=4)
    for i, path in enumerate(paths[:10]):
        tags = []
        for eid in path:
            eq = kg.equipment.get(eid)
            if eq:
                tags.append(eq.tag)
            else:
                # Check if it's an area or other entity
                for store in [kg.areas, kg.connections]:
                    if eid in store:
                        tags.append(store[eid].name[:20])
                        break
        if tags:
            print(f"  Path {i+1}: {' → '.join(tags)}")

    # ── Cascade rules with details ──
    print_section("Detailed cascade failure rules")
    print("  These encode real-world failure propagation knowledge:\n")
    for rule in kg.cascade_rules.values():
        print(f"  📍 {rule.name}")
        print(f"     Source:      {rule.source_equipment_class.value} / {rule.source_failure_mode.value}")
        print(f"     Target:      {rule.target_equipment_class.value}")
        print(f"     Effect:      {rule.cascade_effect}")
        print(f"     Probability: {rule.probability:.0%}")
        print(f"     Delay:       {rule.delay_hours} hours")
        print(f"     Severity:    {rule.severity_factor:.0%}")
        if rule.sensor_signature:
            print(f"     Sensor signs: {rule.sensor_signature}")
        print(f"     Detail:      {rule.description}")
        print()

    # ── Practical RCA use case ──
    print_section("Practical: 'Compressor C-201 vibration is high — what could cause it?'")

    eq = kg.get_by_tag("C-201")
    fms = kg.get_failure_modes_for_class(EquipmentClass.COMPRESSOR)
    vibration_causes = []
    for fm in fms:
        for ind in fm.sensor_indicators:
            if "vibration" in ind.get("measurement", ""):
                vibration_causes.append((fm, ind))

    print("  Knowledge Graph answers:")
    for fm, ind in vibration_causes:
        print(f"  ▸ {fm.failure_mode.value} ({fm.failure_mechanism.value})")
        print(f"    Pattern: {ind['pattern']}")
        print(f"    Lead time: {ind['lead_time_hours']} hours before failure")
        print(f"    Description: {fm.description}")
        print()

    # Also check upstream
    print("  Also check upstream equipment:")
    upstream = kg.get_upstream(eq.id)
    for u in upstream:
        print(f"  ▸ {u.tag} ({u.name}) — could be sending bad feed")

    return kg


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
#  SECTION 4: Synthetic Data Generation — Detailed Control
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

def section_4_data_generation(kg=None):
    print_header("SECTION 4: Synthetic Data Generation — Fine Control")

    if kg is None:
        kg = PlantKnowledgeGraph()
        OilGasDomainBuilder(kg).build_reference_plant()

    # ── Option A: Generate everything (bulk) ──
    print_section("Option A: Generate ALL scenarios (normal + failures + cascades)")
    print("""
  Code:
    config = GeneratorConfig(
        duration_hours=24,           # 24 hours of data
        sample_interval_seconds=10,  # 10-second readings
        seed=42,                     # Reproducible
    )
    generator = SyntheticDataGenerator(kg, config)
    stats = generator.generate_all_datasets('./my_output')

  This creates:
    sensor_timeseries.csv        — Raw time-series (timestamp, tag, value, unit)
    labeled_anomaly_dataset.csv  — Same + labels (normal/anomaly/failure/cascade)
    failure_events.csv           — Failure event log (start, end, sensors affected)
    train_dataset.csv            — 70% training split
    val_dataset.csv              — 15% validation split
    test_dataset.csv             — 15% test split
    knowledge_graph.json         — Full KG export
    knowledge_base_for_llm.md   — LLM-ready knowledge text
    scenario_manifest.json       — All scenarios for reproducibility
""")

    # ── Option B: Generate specific scenarios ──
    print_section("Option B: Generate SPECIFIC scenario (e.g., pump bearing failure)")
    print("  Useful when you want targeted training data.\n")

    config = GeneratorConfig(
        duration_hours=0.5,
        sample_interval_seconds=10,
        seed=42,
    )
    generator = SyntheticDataGenerator(kg, config)

    # Create a specific scenario
    scenario = ScenarioDefinition(
        scenario_type=ScenarioType.FAILURE,
        target_equipment_id="eq_P301",
        failure_mode=FailureMode.BEARING_FAILURE,
        failure_mechanism=FailureMechanism.WEAR,
        description="P-301A bearing failure — gradual vibration increase leading to seizure",
        severity=0.8,
        label="failure",
    )

    samples = generator._generate_scenario_data(scenario, scenario_index=0)
    print(f"  Generated {len(samples)} samples for P-301A bearing failure scenario")
    print(f"\n  Sample data (first 20 readings of VT-301 vibration sensor):")
    print(f"  {'Timestamp':<22} {'Tag':<10} {'Value':<10} {'Label':<12} {'Quality'}")
    print(f"  {'-'*22} {'-'*10} {'-'*10} {'-'*12} {'-'*7}")

    vib_samples = [s for s in samples if s.sensor_tag == "VT-301"][:20]
    for s in vib_samples:
        val_str = f"{s.value:.2f}" if s.value else "NaN"
        print(f"  {s.timestamp:<22} {s.sensor_tag:<10} {val_str:<10} {s.label:<12} {s.quality}")

    print(f"\n  Notice: values start normal, then gradually increase (degradation),")
    print(f"  then spike at failure onset — matching real bearing failure signature.")

    # ── Option C: Custom configuration ──
    print_section("Option C: Tune generation parameters")
    print("""
  Key parameters you can adjust:

    config = GeneratorConfig(
        # Time
        start_time=datetime(2024, 6, 1),  # Custom start date
        duration_hours=168,                # 7 days for serious ML training
        sample_interval_seconds=5,         # 5-second high-frequency data

        # Noise & realism
        base_noise_pct=0.02,               # 2% measurement noise (default 1.5%)
        drift_amplitude_pct=0.01,          # Larger process drift
        diurnal_amplitude_pct=0.015,       # Stronger day/night effect
        process_disturbance_probability=0.005,  # More random disturbances

        # Failure timing
        degradation_start_pct=0.4,         # Degradation starts at 40% of timeline
        failure_onset_pct=0.8,             # Failure at 80%
        pre_failure_lead_time_hours=8.0,   # 8 hours of detectable anomaly before fail

        # Data quality issues (realistic imperfections)
        missing_data_probability=0.002,    # 0.2% missing values
        spike_probability=0.001,           # 0.1% instrument spikes
        stuck_signal_probability=0.0005,   # 0.05% frozen readings

        # ML splits
        seed=42,                           # Always set for reproducibility
    )
""")

    # ── Option D: Scale for production ML ──
    print_section("Option D: Production ML dataset sizes")
    print("""
  Recommended sizes for different use cases:

  ┌─────────────────────────┬──────────┬──────────┬────────────────┐
  │ Use Case                │ Duration │ Interval │ ~Samples       │
  ├─────────────────────────┼──────────┼──────────┼────────────────┤
  │ Quick prototype/test    │ 1 hour   │ 30s      │ ~60,000        │
  │ Model development       │ 24 hours │ 10s      │ ~1.2 million   │
  │ Serious ML training     │ 7 days   │ 10s      │ ~8.5 million   │
  │ Production baseline     │ 30 days  │ 10s      │ ~36 million    │
  │ Full simulation         │ 90 days  │ 5s       │ ~200 million   │
  └─────────────────────────┴──────────┴──────────┴────────────────┘

  CLI commands:
    python main.py -o ./output -d 1 -i 30 --quick    # Prototype
    python main.py -o ./output -d 24 -i 10            # Development
    python main.py -o ./output -d 168 -i 10           # Training
""")

    return kg


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
#  SECTION 5: ML Model Training
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

def section_5_ml_training(kg=None):
    print_header("SECTION 5: ML Model Training with Generated Data")

    print("""
  Your generated data is ready for ML training. Here's how to use it:

  ═══════════════════════════════════════════════════════════════════
  USE CASE 1: Anomaly Detection (Binary Classification)
  ═══════════════════════════════════════════════════════════════════

  The labeled_anomaly_dataset.csv has columns:
    timestamp, equipment_tag, sensor_tag, value, unit,
    label, is_anomaly, scenario_type, context

  'is_anomaly' is your binary target: 0=normal, 1=anomaly

  Code (scikit-learn):
  ─────────────────────
    import pandas as pd
    from sklearn.ensemble import IsolationForest, RandomForestClassifier
    from sklearn.metrics import classification_report

    # Load splits
    train = pd.read_csv('output/train_dataset.csv')
    test = pd.read_csv('output/test_dataset.csv')

    # Feature engineering: pivot sensors to columns per equipment
    def make_features(df):
        # Group by timestamp + equipment, pivot sensor values to columns
        pivoted = df.pivot_table(
            index=['timestamp', 'equipment_tag'],
            columns='sensor_tag',
            values='value',
            aggfunc='first'
        ).reset_index()
        # Add rolling statistics
        for col in pivoted.select_dtypes(include='number').columns:
            pivoted[f'{col}_rolling_mean'] = pivoted[col].rolling(10).mean()
            pivoted[f'{col}_rolling_std'] = pivoted[col].rolling(10).std()
        return pivoted.dropna()

    X_train = make_features(train)
    X_test = make_features(test)

    # Supervised approach (we have labels!)
    feature_cols = [c for c in X_train.columns if c not in
                    ['timestamp', 'equipment_tag', 'label', 'is_anomaly']]
    clf = RandomForestClassifier(n_estimators=100, class_weight='balanced')
    clf.fit(X_train[feature_cols], X_train['is_anomaly'])
    preds = clf.predict(X_test[feature_cols])
    print(classification_report(X_test['is_anomaly'], preds))


  ═══════════════════════════════════════════════════════════════════
  USE CASE 2: Multiclass Failure Classification
  ═══════════════════════════════════════════════════════════════════

  The 'label' column has: normal, anomaly, pre_failure, failure, cascade, startup

  Code:
  ─────
    from sklearn.preprocessing import LabelEncoder

    le = LabelEncoder()
    y_train = le.fit_transform(train['label'])
    y_test = le.transform(test['label'])

    # Train multiclass model
    clf = RandomForestClassifier(n_estimators=200, class_weight='balanced')
    clf.fit(X_train[feature_cols], y_train)

    # This model can tell you: "Is this normal, degrading, or failing?"
    # AND what type of failure pattern it matches


  ═══════════════════════════════════════════════════════════════════
  USE CASE 3: Time-Series Forecasting (Predict Future Values)
  ═══════════════════════════════════════════════════════════════════

  Code (PyTorch LSTM):
  ─────────────────────
    import torch
    import torch.nn as nn

    class SensorLSTM(nn.Module):
        def __init__(self, n_features, hidden_size=64):
            super().__init__()
            self.lstm = nn.LSTM(n_features, hidden_size, batch_first=True)
            self.fc = nn.Linear(hidden_size, n_features)

        def forward(self, x):
            out, _ = self.lstm(x)
            return self.fc(out[:, -1, :])

    # Create sequences: use past 60 readings to predict next 1
    # Train on normal data, then reconstruction error = anomaly score


  ═══════════════════════════════════════════════════════════════════
  USE CASE 4: Equipment-Specific Models
  ═══════════════════════════════════════════════════════════════════

  Filter data for specific equipment:

    compressor_data = train[train['equipment_tag'] == 'C-201']
    pump_data = train[train['equipment_tag'] == 'P-301A']

  Train separate models per equipment class for better accuracy.


  ═══════════════════════════════════════════════════════════════════
  USE CASE 5: Remaining Useful Life (RUL) Prediction
  ═══════════════════════════════════════════════════════════════════

  From degradation scenarios, the label transitions:
    normal → anomaly → pre_failure → failure

  Each sample's position in this sequence = proxy for remaining life.

  Code:
  ─────
    # Filter degradation + failure scenarios
    deg_data = train[train['scenario_type'].isin(['degradation', 'failure'])]

    # Add RUL column: distance from current sample to failure onset
    # (in the generated data, failure_onset_pct = 85% of timeline)
    # So samples before 85% have RUL > 0, samples after have RUL = 0
""")


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
#  SECTION 6: LLM Integration (RAG)
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

def section_6_llm_integration(kg=None):
    print_header("SECTION 6: LLM Integration — Ground PlantCopilot with KG")

    if kg is None:
        kg = PlantKnowledgeGraph()
        OilGasDomainBuilder(kg).build_reference_plant()

    # ── Method 1: Pre-exported knowledge base for RAG ──
    print_section("Method 1: Use knowledge_base_for_llm.md as RAG context")
    print("""
  The exported knowledge_base_for_llm.md is already formatted for LLM consumption.
  Chunk it and put it in your vector database:

  Code (LangChain):
  ──────────────────
    from langchain.document_loaders import TextLoader
    from langchain.text_splitter import RecursiveCharacterTextSplitter
    from langchain.vectorstores import Chroma
    from langchain.embeddings import OpenAIEmbeddings  # or any embedding model

    # Load the KG knowledge base
    loader = TextLoader('output/knowledge_base_for_llm.md')
    docs = loader.load()

    # Chunk by equipment section (## headers)
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=1000,
        chunk_overlap=100,
        separators=["## Equipment:", "## ", "\\n\\n"]
    )
    chunks = splitter.split_documents(docs)

    # Store in vector DB
    vectorstore = Chroma.from_documents(chunks, OpenAIEmbeddings())

    # Query: "What are the failure modes for the gas compressor?"
    results = vectorstore.similarity_search("compressor failure modes", k=3)
""")

    # ── Method 2: Live KG queries as tool calls ──
    print_section("Method 2: KG as LLM Tool (Function Calling)")
    print("  Give the LLM tools to query the KG directly:\n")

    # Simulate what the LLM would call
    example_queries = [
        ("What sensors monitor compressor C-201?", "get_instruments_for_equipment"),
        ("What could cause high vibration on C-201?", "get_failure_modes_for_class"),
        ("If P-301A pump fails, what gets affected?", "get_downstream + get_cascade_paths"),
        ("What safety systems protect V-101?", "get_safety_functions"),
        ("What connects to the Amine Contactor?", "get_upstream + get_downstream"),
    ]

    print(f"  {'User Question':<55} {'KG Tool Call'}")
    print(f"  {'-'*55} {'-'*35}")
    for question, tool in example_queries:
        print(f"  {question:<55} {tool}")

    print("""
  Implementation (FastAPI + LangChain):
  ──────────────────────────────────────
    from langchain.tools import tool

    @tool
    def lookup_equipment(tag: str) -> str:
        '''Look up equipment details by plant tag number.'''
        eq = kg.get_by_tag(tag)
        if not eq:
            return f"No equipment found with tag {tag}"
        instruments = kg.get_instruments_for_equipment(eq.id)
        return json.dumps({
            "tag": eq.tag, "name": eq.name,
            "class": eq.equipment_class.value,
            "criticality": eq.criticality.value,
            "sensors": [{"tag": i.tag, "type": i.measurement_type.value,
                         "range": f"{i.range_min}-{i.range_max} {i.unit}"}
                        for i in instruments],
        })

    @tool
    def get_failure_causes(equipment_tag: str, symptom: str) -> str:
        '''Find possible failure causes for an equipment symptom.'''
        eq = kg.get_by_tag(equipment_tag)
        fms = kg.get_failure_modes_for_class(eq.equipment_class)
        relevant = []
        for fm in fms:
            for ind in fm.sensor_indicators:
                if symptom.lower() in ind.get("measurement", "").lower():
                    relevant.append({
                        "failure_mode": fm.failure_mode.value,
                        "mechanism": fm.failure_mechanism.value,
                        "description": fm.description,
                        "rpn": fm.rpn,
                        "lead_time_hours": ind.get("lead_time_hours"),
                    })
        return json.dumps(relevant)

    @tool
    def trace_cascade_effects(equipment_tag: str) -> str:
        '''Trace what downstream equipment could be affected if this fails.'''
        eq = kg.get_by_tag(equipment_tag)
        paths = kg.get_cascade_paths(eq.id, max_depth=3)
        results = []
        for path in paths:
            tags = [kg.equipment[eid].tag for eid in path if eid in kg.equipment]
            results.append(" → ".join(tags))
        return json.dumps(results)

    # Give these tools to your LLM agent
    tools = [lookup_equipment, get_failure_causes, trace_cascade_effects]
""")

    # ── Method 3: Structured prompt building ──
    print_section("Method 3: Build structured prompts from KG")
    print("  For a specific question, build context from multiple KG queries:\n")

    # Example: operator asks about C-201 vibration
    eq = kg.get_by_tag("C-201")
    instruments = kg.get_instruments_for_equipment(eq.id)
    fms = kg.get_failure_modes_for_class(EquipmentClass.COMPRESSOR)
    sifs = kg.get_safety_functions(eq.id)
    upstream = kg.get_upstream(eq.id)

    context = f"""EQUIPMENT CONTEXT for {eq.tag} ({eq.name}):
- Type: {eq.equipment_class.value}, Criticality: {eq.criticality.value}
- Design: {eq.design_pressure} barg / {eq.design_temperature}°C
- Sensors: {', '.join(f'{i.tag}({i.measurement_type.value})' for i in instruments)}
- Upstream: {', '.join(f'{u.tag}' for u in upstream)}
- Known failure modes: {', '.join(f'{fm.failure_mode.value}(RPN={fm.rpn})' for fm in fms)}
- Safety: {', '.join(f'{s.sif_number}: {s.logic_description}' for s in sifs)}
"""
    print(f"  Generated context for LLM prompt:\n")
    for line in context.strip().split('\n'):
        print(f"    {line}")

    print(f"\n  This context goes into the LLM system prompt for grounded answers.")


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
#  SECTION 7: Root Cause Analysis
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

def section_7_rca(kg=None):
    print_header("SECTION 7: Root Cause Analysis with Knowledge Graph")

    if kg is None:
        kg = PlantKnowledgeGraph()
        OilGasDomainBuilder(kg).build_reference_plant()

    print_section("RCA Workflow: Operator reports 'C-201 discharge temperature rising'")

    eq = kg.get_by_tag("C-201")

    print("  Step 1: Identify affected sensor")
    instruments = kg.get_instruments_for_equipment(eq.id)
    discharge_t = [i for i in instruments if "201D" in i.tag and i.measurement_type == MeasurementType.TEMPERATURE]
    if discharge_t:
        inst = discharge_t[0]
        print(f"    Sensor: {inst.tag} ({inst.measurement_type.value})")
        print(f"    Range: {inst.range_min}-{inst.range_max} {inst.unit}")
        print(f"    Alarms: H={inst.high_alarm}, HH={inst.high_high_alarm}")

    print("\n  Step 2: Check physics rules — what CAUSES discharge temp to rise?")
    for rule in kg.physics_rules.values():
        if rule.equipment_class == EquipmentClass.COMPRESSOR and "discharge_temp" in rule.name:
            print(f"    Physics: {rule.name}")
            print(f"    Equation: {rule.equation}")
            print(f"    Means: Higher suction temp OR higher pressure ratio OR lower efficiency")

    print("\n  Step 3: Check failure modes that match this symptom")
    fms = kg.get_failure_modes_for_class(EquipmentClass.COMPRESSOR)
    matching = []
    for fm in fms:
        for ind in fm.sensor_indicators:
            if "temperature" in ind.get("measurement", ""):
                matching.append((fm, ind))

    for fm, ind in matching:
        print(f"    ▸ {fm.failure_mode.value}: {fm.description}")
        print(f"      Pattern: {ind['pattern']}, Lead time: {ind['lead_time_hours']}h")
        if fm.cascade_targets:
            print(f"      Cascade: {fm.cascade_targets}")

    print("\n  Step 4: Check upstream — could the cause be external?")
    upstream = kg.get_upstream(eq.id)
    for u in upstream:
        print(f"    ▸ Check {u.tag} ({u.name})")
        u_fms = kg.get_failure_modes_for_class(u.equipment_class)
        for ufm in u_fms[:2]:
            if ufm.cascade_targets:
                for ct in ufm.cascade_targets:
                    if "temperature" in ct.get("effect", "").lower() or "compressor" in ct.get("equipment_class", ""):
                        print(f"      Could cause: {ufm.failure_mode.value} → {ct['effect']}")

    print("\n  Step 5: Check operating envelope — is it within limits?")
    for env in kg.operating_envelopes.values():
        if env.equipment_id == eq.id and env.state == "normal":
            for param in env.parameters:
                if "temperature" in param.get("measurement_type", ""):
                    print(f"    {param['measurement_type']}:")
                    print(f"      Safe: {param.get('min_safe')}-{param.get('max_safe')} {param.get('unit')}")
                    print(f"      Normal: {param.get('min_normal')}-{param.get('max_normal')}")
                    print(f"      Optimal: {param.get('optimal')}")

    print("""
  Step 6: Automated RCA conclusion

    The KG gives you a ranked list of probable causes:
    1. FOULING (most likely) — gradual efficiency drop raises discharge temp
       → Check: polytropic efficiency trend over past weeks
       → Action: Schedule online wash or shutdown for cleaning

    2. HIGHER SUCTION TEMPERATURE — upstream cooler issue
       → Check: E-201 aftercooler outlet temperature
       → Action: Check cooling water flow, inspect for fouling

    3. HIGHER PRESSURE RATIO — increased system backpressure
       → Check: downstream pressure trending up
       → Action: Check downstream vessels, valves for restriction

    4. BEARING DEGRADATION — internal friction adding heat
       → Check: vibration trends, bearing temperature
       → Action: Schedule bearing inspection within MTTR window
""")


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
#  SECTION 8: Custom Plant Setup
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

def section_8_custom_plant():
    print_header("SECTION 8: Add Your Own Equipment to the Knowledge Graph")

    print("""
  The reference plant is a TEMPLATE. Here's how to customize it:

  ═══════════════════════════════════════════════════════════════════
  Step 1: Create a new KG (or extend the reference)
  ═══════════════════════════════════════════════════════════════════
""")

    kg = PlantKnowledgeGraph()

    # Add your own site
    site = Site(id="my_site", name="My Processing Plant",
                location="Gujarat, India", country="India")
    kg.add_entity(site)

    # Add an area
    area = Area(id="area_100", name="Feed Processing",
                plant_id="my_plant", area_number=100)
    kg.add_entity(area)

    # Add YOUR equipment
    my_pump = Equipment(
        id="eq_my_pump", tag="P-101A",
        name="Feed Water Pump",
        unit_id="area_100",
        equipment_class=EquipmentClass.PUMP,
        equipment_subclass=EquipmentSubclass.PUMP_CENTRIFUGAL,
        criticality=CriticalityRating.HIGH,
        design_pressure=15.0,
        design_temperature=80.0,
        material_of_construction="SS316L",
        rated_power_kw=75.0,
        rated_speed_rpm=2960,
        driver_type="electric_motor",
        maintenance_strategy=MaintenanceStrategy.PREDICTIVE,
        mtbf_hours=25000,
        mttr_hours=8,
    )
    kg.add_entity(my_pump)

    # Add sensors for YOUR equipment
    my_sensors = [
        Instrument(id="i1", tag="PT-101D", equipment_id="eq_my_pump",
                   measurement_type=MeasurementType.PRESSURE, unit="barg",
                   range_min=0, range_max=15,
                   high_alarm=12, high_high_alarm=14),
        Instrument(id="i2", tag="FT-101", equipment_id="eq_my_pump",
                   measurement_type=MeasurementType.FLOW, unit="m³/h",
                   range_min=0, range_max=100,
                   low_alarm=10, high_alarm=90),
        Instrument(id="i3", tag="VT-101", equipment_id="eq_my_pump",
                   measurement_type=MeasurementType.VIBRATION, unit="mm/s",
                   range_min=0, range_max=15,
                   high_alarm=7, high_high_alarm=11),
        Instrument(id="i4", tag="TT-101B", equipment_id="eq_my_pump",
                   measurement_type=MeasurementType.TEMPERATURE, unit="°C",
                   range_min=0, range_max=100,
                   high_alarm=70, high_high_alarm=85),
        Instrument(id="i5", tag="IT-101", equipment_id="eq_my_pump",
                   measurement_type=MeasurementType.CURRENT, unit="A",
                   range_min=0, range_max=100,
                   high_alarm=75, high_high_alarm=90),
    ]
    for s in my_sensors:
        kg.add_entity(s)

    # Verify
    eq = kg.get_by_tag("P-101A")
    instruments = kg.get_instruments_for_equipment(eq.id)
    print(f"  Added: {eq.tag} ({eq.name}) with {len(instruments)} sensors")
    for i in instruments:
        print(f"    {i.tag}: {i.measurement_type.value} ({i.range_min}-{i.range_max} {i.unit})")

    print("""
  ═══════════════════════════════════════════════════════════════════
  Step 2: Generate synthetic data for YOUR equipment
  ═══════════════════════════════════════════════════════════════════

    # The generator automatically discovers your equipment + sensors
    config = GeneratorConfig(duration_hours=24, sample_interval_seconds=10)
    generator = SyntheticDataGenerator(kg, config)
    stats = generator.generate_all_datasets('./my_plant_output')

    # It will:
    # - Generate normal operating data using your sensor ranges
    # - Apply pump physics rules (Q-H curve, power correlation)
    # - Generate failure scenarios using FMEA database
    # - Create labeled training data

  ═══════════════════════════════════════════════════════════════════
  Step 3: Add custom failure modes (optional)
  ═══════════════════════════════════════════════════════════════════

    my_failure = EquipmentFailureMode(
        name="my_pump_impeller_erosion",
        equipment_class=EquipmentClass.PUMP,
        failure_mode=FailureMode.IMPELLER_DAMAGE,
        failure_mechanism=FailureMechanism.EROSION_CAVITATION,
        failure_rate_per_year=0.05,
        severity=7, occurrence=4, detection_difficulty=5,
        rpn=140,
        description="Impeller erosion from sand in feed water",
        sensor_indicators=[
            {"measurement": "flow", "pattern": "decreasing", "lead_time_hours": 720},
            {"measurement": "vibration", "pattern": "increasing_trend", "lead_time_hours": 480},
            {"measurement": "current", "pattern": "increasing_for_same_throughput", "lead_time_hours": 336},
        ],
    )
    kg.add_entity(my_failure)

  ═══════════════════════════════════════════════════════════════════
  Step 4: Add process connections (how equipment links together)
  ═══════════════════════════════════════════════════════════════════

    conn = ProcessConnection(
        name="Feed Water to HX",
        from_equipment_id="eq_my_pump",
        to_equipment_id="eq_my_hx",
        fluid_type=FluidType.COOLING_WATER,
        pipe_size_inches=6,
        normal_flow_rate=80,
        normal_pressure=10,
    )
    kg.add_entity(conn)

    # Now the cascade analysis knows: if pump fails → HX loses cooling
""")


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
#  SECTION 9: Production Integration
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

def section_9_production():
    print_header("SECTION 9: Production Deployment Patterns")

    print("""
  ═══════════════════════════════════════════════════════════════════
  Architecture: How KG + Synthetic Data fits in PlantCopilot
  ═══════════════════════════════════════════════════════════════════

    ┌─────────────────────────────────────────────────────────────┐
    │                    PlantCopilot Architecture                 │
    │                                                             │
    │  ┌──────────┐  ┌──────────┐  ┌──────────────────────────┐  │
    │  │ CAD/DXF  │  │  P&ID    │  │   Sensor Data (OPC-UA/   │  │
    │  │ Parser   │  │ Detector │  │   MQTT/Modbus)           │  │
    │  └────┬─────┘  └────┬─────┘  └───────────┬──────────────┘  │
    │       │              │                     │                 │
    │       ▼              ▼                     ▼                 │
    │  ┌──────────────────────────────────────────────────────┐   │
    │  │          KNOWLEDGE GRAPH  ◄── YOU ARE HERE           │   │
    │  │   Equipment │ Sensors │ Failure Modes │ Physics      │   │
    │  │   Cascades  │ SIFs    │ Maintenance   │ Envelopes    │   │
    │  └──────────────────┬───────────────────────────────────┘   │
    │                     │                                       │
    │       ┌─────────────┼──────────────┐                        │
    │       ▼             ▼              ▼                        │
    │  ┌─────────┐  ┌──────────┐  ┌───────────┐                  │
    │  │ Anomaly │  │   LLM    │  │   RCA     │                  │
    │  │ Detect  │  │  (RAG)   │  │  Engine   │                  │
    │  │ (ML)    │  │          │  │           │                  │
    │  └────┬────┘  └────┬─────┘  └─────┬─────┘                  │
    │       │             │              │                        │
    │       ▼             ▼              ▼                        │
    │  ┌──────────────────────────────────────────────────────┐   │
    │  │              3D Visualization + Dashboard            │   │
    │  │         Real-time sensor overlay + alerts            │   │
    │  └──────────────────────────────────────────────────────┘   │
    └─────────────────────────────────────────────────────────────┘

  ═══════════════════════════════════════════════════════════════════
  Integration Pattern: FastAPI endpoints for KG
  ═══════════════════════════════════════════════════════════════════

    from fastapi import FastAPI
    app = FastAPI()

    # Initialize KG once at startup
    kg = PlantKnowledgeGraph()
    OilGasDomainBuilder(kg).build_reference_plant()

    @app.get("/api/v1/equipment/{tag}")
    def get_equipment(tag: str):
        eq = kg.get_by_tag(tag)
        instruments = kg.get_instruments_for_equipment(eq.id)
        fms = kg.get_failure_modes_for_class(eq.equipment_class)
        return {
            "equipment": asdict(eq),
            "instruments": [asdict(i) for i in instruments],
            "failure_modes": [asdict(fm) for fm in fms],
            "upstream": [asdict(u) for u in kg.get_upstream(eq.id)],
            "downstream": [asdict(d) for d in kg.get_downstream(eq.id)],
            "safety_functions": [asdict(s) for s in kg.get_safety_functions(eq.id)],
        }

    @app.get("/api/v1/rca/{tag}")
    def root_cause_analysis(tag: str, symptom: str):
        eq = kg.get_by_tag(tag)
        fms = kg.get_failure_modes_for_class(eq.equipment_class)
        causes = []
        for fm in fms:
            for ind in fm.sensor_indicators:
                if symptom.lower() in ind.get("measurement", ""):
                    causes.append({
                        "failure_mode": fm.failure_mode.value,
                        "rpn": fm.rpn,
                        "lead_time_hours": ind.get("lead_time_hours"),
                        "description": fm.description,
                    })
        return {"causes": sorted(causes, key=lambda x: x["rpn"], reverse=True)}

    @app.get("/api/v1/cascade/{tag}")
    def cascade_analysis(tag: str):
        eq = kg.get_by_tag(tag)
        paths = kg.get_cascade_paths(eq.id)
        return {"cascade_paths": [
            [kg.equipment[eid].tag for eid in path if eid in kg.equipment]
            for path in paths
        ]}


  ═══════════════════════════════════════════════════════════════════
  Next Steps — Build Order
  ═══════════════════════════════════════════════════════════════════

    DONE:
    ✅ 1. Plant Knowledge Graph (ontology + builder + export)
    ✅ 2. Synthetic Data Generator (physics-based, labeled, ML-ready)
    ✅ 3. 3D Visualization frontend (Three.js + React)
    ✅ 4. Sensor pipeline (OPC-UA/MQTT/Modbus + anomaly detection)
    ✅ 5. FastAPI backend (REST + WebSocket)

    NEXT (in order of value):
    ⬜ 6. ML Anomaly Detection model (train on synthetic data)
    ⬜ 7. LLM RAG integration (PlantCopilot chat grounded with KG)
    ⬜ 8. RCA Engine (automated root cause analysis using KG)
    ⬜ 9. Report Generator (auto-generate PDF maintenance reports)
    ⬜ 10. Customer onboarding (import real plant data into KG)
""")


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
#  MAIN
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="PlantCopilot Usage Guide")
    parser.add_argument("--section", "-s", type=int, help="Run specific section (1-9)")
    parser.add_argument("--all", action="store_true", help="Run all sections")
    args = parser.parse_args()

    sections = {
        1: section_1_quick_start,
        2: section_2_kg_queries,
        3: section_3_cascade_analysis,
        4: section_4_data_generation,
        5: section_5_ml_training,
        6: section_6_llm_integration,
        7: section_7_rca,
        8: section_8_custom_plant,
        9: section_9_production,
    }

    if args.section:
        if args.section in sections:
            sections[args.section]()
        else:
            print(f"Section {args.section} not found. Available: 1-9")
    elif args.all:
        kg = None
        for num, func in sorted(sections.items()):
            result = func(kg)
            if isinstance(result, PlantKnowledgeGraph):
                kg = result
    else:
        print("PlantCopilot Data Foundation — Usage Guide")
        print("="*50)
        print("\nAvailable sections:")
        print("  1. Quick Start — Build KG + Generate Data")
        print("  2. Knowledge Graph Queries")
        print("  3. Failure Cascade Analysis")
        print("  4. Synthetic Data Generation (detailed)")
        print("  5. ML Model Training patterns")
        print("  6. LLM Integration (RAG)")
        print("  7. Root Cause Analysis")
        print("  8. Custom Plant Setup")
        print("  9. Production Deployment")
        print("\nRun:  python usage_guide.py --section 1")
        print("      python usage_guide.py --all")
