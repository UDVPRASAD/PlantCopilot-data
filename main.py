#!/usr/bin/env python3
"""
PlantCopilot Data Foundation — CLI Runner
==========================================
Build knowledge graph → Generate synthetic data → Export all artifacts.

Usage:
  # Full pipeline: build KG + generate data
  python main.py --output ./output --duration 24 --interval 10

  # Quick test (1 hour of data, 30s interval)
  python main.py --output ./output --duration 1 --interval 30 --quick

  # Large dataset for ML training (7 days)
  python main.py --output ./output --duration 168 --interval 10

  # Only build knowledge graph (no synthetic data)
  python main.py --output ./output --kg-only

  # Custom seed for reproducibility
  python main.py --output ./output --seed 42
"""

import sys
import json
import time
import argparse
from pathlib import Path
from datetime import datetime

# Add parent to path for imports
sys.path.insert(0, str(Path(__file__).parent))

from knowledge_graph.ontology import *
from knowledge_graph.builder import PlantKnowledgeGraph, OilGasDomainBuilder
from synthetic_data.generator import SyntheticDataGenerator, GeneratorConfig

try:
    from loguru import logger
    logger.remove()
    logger.add(sys.stderr, level="INFO", format="<green>{time:HH:mm:ss}</green> | <level>{level: <8}</level> | {message}")
except ImportError:
    import logging
    logging.basicConfig(level=logging.INFO, format="%(asctime)s | %(levelname)-8s | %(message)s")
    logger = logging.getLogger(__name__)


def build_knowledge_graph() -> PlantKnowledgeGraph:
    """Build the complete oil & gas reference plant knowledge graph."""
    logger.info("━" * 60)
    logger.info("PHASE 1: Building Plant Knowledge Graph")
    logger.info("━" * 60)

    kg = PlantKnowledgeGraph()
    builder = OilGasDomainBuilder(kg)
    builder.build_reference_plant()

    stats = kg.stats()
    logger.info("")
    logger.info("Knowledge Graph Statistics:")
    logger.info(f"  Sites:                {stats['sites']}")
    logger.info(f"  Plants:               {stats['plants']}")
    logger.info(f"  Areas:                {stats['areas']}")
    logger.info(f"  Equipment:            {stats['equipment']}")
    logger.info(f"  Instruments/Sensors:  {stats['instruments']}")
    logger.info(f"  Process Connections:  {stats['connections']}")
    logger.info(f"  Failure Modes (FMEA): {stats['failure_modes']}")
    logger.info(f"  Operating Envelopes:  {stats['operating_envelopes']}")
    logger.info(f"  Maintenance Procs:    {stats['maintenance_procedures']}")
    logger.info(f"  Safety Interlocks:    {stats['safety_interlocks']}")
    logger.info(f"  Control Loops:        {stats['control_loops']}")
    logger.info(f"  Physics Rules:        {stats['physics_rules']}")
    logger.info(f"  Cascade Rules:        {stats['cascade_rules']}")
    logger.info(f"  Relationships:        {stats['relationships']}")
    logger.info(f"  ─────────────────────────")
    logger.info(f"  TOTAL ENTITIES:       {stats['total_entities']}")
    logger.info("")

    return kg


def generate_synthetic_data(kg: PlantKnowledgeGraph, config: GeneratorConfig,
                             output_dir: str) -> dict:
    """Generate synthetic data using knowledge graph."""
    logger.info("━" * 60)
    logger.info("PHASE 2: Generating Synthetic Data")
    logger.info("━" * 60)
    logger.info(f"  Duration:     {config.duration_hours} hours")
    logger.info(f"  Interval:     {config.sample_interval_seconds} seconds")
    logger.info(f"  Est. samples: ~{int(config.duration_hours * 3600 / config.sample_interval_seconds)} per sensor per scenario")
    logger.info("")

    generator = SyntheticDataGenerator(kg, config)
    stats = generator.generate_all_datasets(output_dir)

    return stats


def main():
    parser = argparse.ArgumentParser(
        description="PlantCopilot Data Foundation: Knowledge Graph + Synthetic Data Generator",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python main.py --output ./output                    # Default 24h at 10s
  python main.py --output ./output --quick            # Quick 1h test
  python main.py --output ./output --duration 168     # 7 days for ML
  python main.py --output ./output --kg-only          # KG only, no data
        """,
    )

    parser.add_argument("--output", "-o", type=str, default="./output",
                        help="Output directory for all generated artifacts (default: ./output)")
    parser.add_argument("--duration", "-d", type=float, default=24.0,
                        help="Duration of synthetic data in hours (default: 24)")
    parser.add_argument("--interval", "-i", type=int, default=10,
                        help="Sample interval in seconds (default: 10)")
    parser.add_argument("--seed", "-s", type=int, default=42,
                        help="Random seed for reproducibility (default: 42)")
    parser.add_argument("--kg-only", action="store_true",
                        help="Only build knowledge graph, skip data generation")
    parser.add_argument("--quick", action="store_true",
                        help="Quick mode: 1 hour duration, 30s interval")
    parser.add_argument("--noise", type=float, default=0.015,
                        help="Base noise percentage (default: 0.015 = 1.5%%)")

    args = parser.parse_args()

    # Quick mode overrides
    if args.quick:
        args.duration = 1.0
        args.interval = 30

    output_path = Path(args.output)
    output_path.mkdir(parents=True, exist_ok=True)

    logger.info("╔════════════════════════════════════════════════════════╗")
    logger.info("║      PlantCopilot Data Foundation Generator           ║")
    logger.info("║      Knowledge Graph + Synthetic Data Pipeline        ║")
    logger.info("╚════════════════════════════════════════════════════════╝")
    logger.info("")

    start_time = time.time()

    # Phase 1: Knowledge Graph
    kg = build_knowledge_graph()

    # Export KG regardless
    kg.export_json(str(output_path / "knowledge_graph.json"))
    kg.export_for_llm(str(output_path / "knowledge_base_for_llm.md"))
    kg.export_cypher(str(output_path / "knowledge_graph.cypher"))
    logger.info("Knowledge graph exported (JSON + LLM + Neo4j Cypher)")

    if args.kg_only:
        elapsed = time.time() - start_time
        logger.info(f"\nCompleted (KG only) in {elapsed:.1f}s")
        return

    # Phase 2: Synthetic Data
    config = GeneratorConfig(
        start_time=datetime(2025, 1, 1),
        sample_interval_seconds=args.interval,
        duration_hours=args.duration,
        base_noise_pct=args.noise,
        seed=args.seed,
    )

    stats = generate_synthetic_data(kg, config, str(output_path))

    elapsed = time.time() - start_time

    # Summary
    logger.info("")
    logger.info("━" * 60)
    logger.info("GENERATION COMPLETE")
    logger.info("━" * 60)
    logger.info(f"  Time elapsed:     {elapsed:.1f}s")
    logger.info(f"  Total samples:    {stats['total_samples']:,}")
    logger.info(f"  Total scenarios:  {stats['total_scenarios']}")
    logger.info(f"  Equipment:        {stats['unique_equipment']}")
    logger.info(f"  Sensors:          {stats['unique_sensors']}")
    logger.info(f"  Output directory:  {output_path.absolute()}")
    logger.info("")
    logger.info("Output files:")
    for f in sorted(output_path.glob("*")):
        size = f.stat().st_size
        if size > 1_000_000:
            size_str = f"{size / 1_000_000:.1f} MB"
        elif size > 1_000:
            size_str = f"{size / 1_000:.1f} KB"
        else:
            size_str = f"{size} B"
        logger.info(f"  {f.name: <40} {size_str: >10}")


if __name__ == "__main__":
    main()
