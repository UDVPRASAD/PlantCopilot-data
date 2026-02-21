# Plant Knowledge Base


## Equipment: V-101 — Inlet Separator
- Type: separator (three_phase_separator)
- Criticality: critical
- Design: 75.0 barg / 120.0 °C
- Material: SA-516 Gr.70 + 3mm CA
- Status: running
- Instruments (5):
  - PT-101: pressure (0-100 barg) [H=70, HH=75]
  - TT-101: temperature (0-150 °C) [H=100, HH=110, L=30]
  - LT-101A: level (0-100 %) [H=85, HH=90, L=15, LL=10]
  - LT-101B: level (0-100 %) [H=85, HH=90, L=15, LL=10]
  - FT-101: flow (0-5000 m³/h) [L=200, LL=50]
- Known failure modes:
  - high_output: Liquid carryover to gas outlet, floods downstream equipment (RPN=75)
  - corrosion: Wall thinning, potential loss of containment (RPN=168)
- Feeds to: C-201, TK-101, V-701
- Safety functions:
  - SIF-101-01: IF LT-101A OR LT-101B > 90% THEN close XV-101 inlet valve

## Equipment: V-102 — Test Separator
- Type: separator (two_phase_separator)
- Criticality: medium
- Design: 75.0 barg / 120.0 °C
- Material: SA-516 Gr.70
- Status: running
- Known failure modes:
  - high_output: Liquid carryover to gas outlet, floods downstream equipment (RPN=75)
  - corrosion: Wall thinning, potential loss of containment (RPN=168)

## Equipment: TK-101 — Produced Water Tank
- Type: tank (fixed_roof_tank)
- Criticality: high
- Design: 0.07 barg / 80.0 °C
- Material: A283 Gr.C
- Status: running
- Instruments (2):
  - LT-101T: level (0-100 %) [H=90, HH=95, L=10, LL=5]
  - TT-101T: temperature (0-80 °C) [H=65, HH=70]
- Known failure modes:
  - corrosion: Floor/shell thinning, potential leak to containment (RPN=196)
  - high_output: Overflow, environmental spill (RPN=56)
- Receives from: V-101
- Safety functions:
  - SIF-101-02: 

## Equipment: C-201 — 1st Stage Gas Compressor
- Type: compressor (centrifugal_compressor)
- Criticality: critical
- Design: 45.0 barg / 150.0 °C
- Material: AISI 4140
- Status: running
- Instruments (13):
  - PT-201S: pressure (0-30 barg) [L=5, LL=3]
  - PT-201D: pressure (0-50 barg) [H=42, HH=45]
  - TT-201S: temperature (0-100 °C)
  - TT-201D: temperature (0-180 °C) [H=140, HH=150]
  - VT-201DE: vibration (0-25 mm/s) [H=12, HH=18]
  - VT-201NDE: vibration (0-25 mm/s) [H=12, HH=18]
  - ST-201: speed (0-13000 RPM) [H=12000, HH=12500, L=9000, LL=8000]
  - TT-201B1: temperature (0-130 °C) [H=95, HH=110]
  - TT-201B2: temperature (0-130 °C) [H=95, HH=110]
  - YT-201AX: displacement (0-150 μm) [H=75, HH=100]
  - YT-201AY: displacement (0-150 μm) [H=75, HH=100]
  - FT-201: flow (0-50000 m³/h)
  - IT-201: current (0-500 A) [H=400, HH=450]
- Known failure modes:
  - surge: Rapid flow reversal, severe vibration, potential mechanical damage (RPN=54)
  - bearing_failure: Vibration increase, temperature rise, potential catastrophic failure (RPN=135)
  - fouling: Reduced efficiency, higher discharge temperature, reduced capacity (RPN=210)
- Receives from: V-101, V-201
- Feeds to: E-201
- Safety functions:
  - SIF-201-01: IF VT-201DE AND VT-201NDE > 18 mm/s THEN trip compressor, close suction/discharge valves
  - SIF-201-02: Anti-surge controller opens recycle valve when operating point approaches surge line
  - SIF-201-03: 

## Equipment: C-202 — 2nd Stage Gas Compressor
- Type: compressor (centrifugal_compressor)
- Criticality: critical
- Design: 85.0 barg / 160.0 °C
- Material: 
- Status: running
- Instruments (5):
  - PT-202S: pressure (0-50 barg) [L=33, LL=30]
  - PT-202D: pressure (0-90 barg) [H=80, HH=85]
  - TT-202D: temperature (0-200 °C) [H=150, HH=160]
  - VT-202DE: vibration (0-25 mm/s) [H=12, HH=18]
  - ST-202: speed (0-14000 RPM) [H=12500, HH=13000, L=10000, LL=9000]
- Known failure modes:
  - surge: Rapid flow reversal, severe vibration, potential mechanical damage (RPN=54)
  - bearing_failure: Vibration increase, temperature rise, potential catastrophic failure (RPN=135)
  - fouling: Reduced efficiency, higher discharge temperature, reduced capacity (RPN=210)
- Receives from: V-202
- Feeds to: T-301

## Equipment: E-201 — 1st Stage Aftercooler
- Type: heat_exchanger (shell_and_tube)
- Criticality: high
- Design: 45.0 barg / 150.0 °C
- Material: SA-516 Gr.70 / Admiralty Brass tubes
- Status: running
- Instruments (6):
  - TT-201HI: temperature (0-180 °C)
  - TT-201HO: temperature (0-100 °C) [H=50, HH=55]
  - TT-201CI: temperature (0-50 °C)
  - TT-201CO: temperature (0-80 °C) [H=50, HH=55]
  - PT-201E: pressure (0-50 barg) [H=42, HH=45]
  - PDT-201: pressure (0-5 bar) [H=1.5, HH=2.0]
- Known failure modes:
  - fouling: Reduced heat transfer, increased pressure drop, higher outlet temperatures (RPN=200)
  - tube_failure: Cross-contamination between shell and tube fluids (RPN=96)
  - external_leakage: Loss of containment, fire/explosion risk if hydrocarbon (RPN=140)
- Receives from: C-201, P-601A
- Feeds to: V-202

## Equipment: V-201 — 1st Stage Suction Scrubber
- Type: separator (two_phase_separator)
- Criticality: high
- Design: 20.0 barg / 100.0 °C
- Material: 
- Status: running
- Known failure modes:
  - high_output: Liquid carryover to gas outlet, floods downstream equipment (RPN=75)
  - corrosion: Wall thinning, potential loss of containment (RPN=168)
- Feeds to: C-201

## Equipment: V-202 — 2nd Stage Suction Scrubber
- Type: separator (two_phase_separator)
- Criticality: high
- Design: 50.0 barg / 120.0 °C
- Material: 
- Status: running
- Known failure modes:
  - high_output: Liquid carryover to gas outlet, floods downstream equipment (RPN=75)
  - corrosion: Wall thinning, potential loss of containment (RPN=168)
- Receives from: E-201
- Feeds to: C-202

## Equipment: T-301 — Amine Contactor
- Type: column (tray_column)
- Criticality: critical
- Design: 85.0 barg / 80.0 °C
- Material: SA-516 Gr.70 + SS316L cladding
- Status: running
- Instruments (6):
  - PT-301T: pressure (0-90 barg) [H=80, HH=85]
  - PT-301B: pressure (0-90 barg) [H=82, HH=85]
  - TT-301T: temperature (0-80 °C) [H=60, HH=70]
  - TT-301B: temperature (0-80 °C) [H=55, HH=65]
  - LT-301: level (0-100 %) [H=80, HH=90, L=20, LL=10]
  - AT-301: concentration (0-100 ppm) [H=4, HH=10]
- Known failure modes:
  - fouling: Increased pressure drop, reduced separation efficiency, flooding (RPN=210)
  - plugged: Tray/packing blockage, loss of mass transfer (RPN=140)
- Receives from: C-202, P-301A
- Feeds to: T-401, E-301
- Safety functions:
  - SIF-301-01: IF PT-301 > 85 barg THEN close inlet, open relief to flare

## Equipment: T-302 — Amine Regenerator
- Type: column (tray_column)
- Criticality: critical
- Design: 3.5 barg / 130.0 °C
- Material: SA-516 Gr.70 + SS316L cladding
- Status: running
- Known failure modes:
  - fouling: Increased pressure drop, reduced separation efficiency, flooding (RPN=210)
  - plugged: Tray/packing blockage, loss of mass transfer (RPN=140)
- Receives from: E-301
- Feeds to: E-301

## Equipment: P-301A — Lean Amine Pump (A)
- Type: pump (centrifugal_pump)
- Criticality: high
- Design: None barg / None °C
- Material: 
- Status: running
- Instruments (6):
  - PT-301S: pressure (0-5 barg) [L=1.0, LL=0.5]
  - PT-301D: pressure (0-15 barg) [H=12, HH=14]
  - FT-301: flow (0-500 m³/h) [H=400, HH=450, L=50, LL=30]
  - TT-301P: temperature (0-100 °C) [H=75, HH=85]
  - VT-301: vibration (0-15 mm/s) [H=8, HH=12]
  - IT-301: current (0-200 A) [H=160, HH=180, L=20]
- Known failure modes:
  - bearing_failure: Increased vibration → temperature rise → seizure (RPN=144)
  - seal_failure: External leakage, reduced discharge pressure (RPN=168)
  - cavitation: Erosion damage, vibration, noise, reduced performance (RPN=175)
  - shaft_misalignment: Increased vibration at 1x and 2x RPM, coupling wear (RPN=120)
- Receives from: E-302
- Feeds to: T-301

## Equipment: P-301B — Lean Amine Pump (B) [Standby]
- Type: pump (centrifugal_pump)
- Criticality: high
- Design: None barg / None °C
- Material: 
- Status: standby
- Known failure modes:
  - bearing_failure: Increased vibration → temperature rise → seizure (RPN=144)
  - seal_failure: External leakage, reduced discharge pressure (RPN=168)
  - cavitation: Erosion damage, vibration, noise, reduced performance (RPN=175)
  - shaft_misalignment: Increased vibration at 1x and 2x RPM, coupling wear (RPN=120)

## Equipment: E-301 — Lean/Rich Amine Exchanger
- Type: heat_exchanger (plate)
- Criticality: high
- Design: 85.0 barg / 130.0 °C
- Material: 
- Status: running
- Known failure modes:
  - fouling: Reduced heat transfer, increased pressure drop, higher outlet temperatures (RPN=200)
  - tube_failure: Cross-contamination between shell and tube fluids (RPN=96)
  - external_leakage: Loss of containment, fire/explosion risk if hydrocarbon (RPN=140)
- Receives from: T-301, T-302
- Feeds to: T-302, E-302

## Equipment: E-302 — Lean Amine Cooler
- Type: heat_exchanger (air_cooled)
- Criticality: medium
- Design: None barg / None °C
- Material: 
- Status: running
- Known failure modes:
  - fouling: Reduced heat transfer, increased pressure drop, higher outlet temperatures (RPN=200)
  - tube_failure: Cross-contamination between shell and tube fluids (RPN=96)
  - external_leakage: Loss of containment, fire/explosion risk if hydrocarbon (RPN=140)
- Receives from: E-301, P-601A
- Feeds to: P-301A

## Equipment: T-401 — Glycol Contactor
- Type: column (packed_column)
- Criticality: high
- Design: 85.0 barg / 60.0 °C
- Material: 
- Status: running
- Known failure modes:
  - fouling: Increased pressure drop, reduced separation efficiency, flooding (RPN=210)
  - plugged: Tray/packing blockage, loss of mass transfer (RPN=140)
- Receives from: T-301
- Feeds to: T-501

## Equipment: E-401 — Glycol Reboiler
- Type: heat_exchanger (shell_and_tube)
- Criticality: high
- Design: None barg / 204.0 °C
- Material: 
- Status: running
- Known failure modes:
  - fouling: Reduced heat transfer, increased pressure drop, higher outlet temperatures (RPN=200)
  - tube_failure: Cross-contamination between shell and tube fluids (RPN=96)
  - external_leakage: Loss of containment, fire/explosion risk if hydrocarbon (RPN=140)

## Equipment: T-501 — Demethanizer
- Type: column (tray_column)
- Criticality: critical
- Design: 35.0 barg / -50.0 °C
- Material: 
- Status: running
- Known failure modes:
  - fouling: Increased pressure drop, reduced separation efficiency, flooding (RPN=210)
  - plugged: Tray/packing blockage, loss of mass transfer (RPN=140)
- Receives from: T-401
- Feeds to: E-501

## Equipment: E-501 — Gas-Gas Exchanger
- Type: heat_exchanger (plate)
- Criticality: high
- Design: None barg / None °C
- Material: 
- Status: running
- Known failure modes:
  - fouling: Reduced heat transfer, increased pressure drop, higher outlet temperatures (RPN=200)
  - tube_failure: Cross-contamination between shell and tube fluids (RPN=96)
  - external_leakage: Loss of containment, fire/explosion risk if hydrocarbon (RPN=140)
- Receives from: T-501

## Equipment: P-601A — Cooling Water Pump (A)
- Type: pump (centrifugal_pump)
- Criticality: high
- Design: None barg / None °C
- Material: 
- Status: running
- Instruments (5):
  - PT-601D: pressure (0-10 barg) [H=7, HH=8]
  - FT-601: flow (0-2000 m³/h) [H=1800, L=400, LL=200]
  - TT-601: temperature (0-50 °C) [H=38, HH=42]
  - VT-601: vibration (0-15 mm/s) [H=8, HH=12]
  - IT-601: current (0-400 A) [H=300, HH=350, L=30]
- Known failure modes:
  - bearing_failure: Increased vibration → temperature rise → seizure (RPN=144)
  - seal_failure: External leakage, reduced discharge pressure (RPN=168)
  - cavitation: Erosion damage, vibration, noise, reduced performance (RPN=175)
  - shaft_misalignment: Increased vibration at 1x and 2x RPM, coupling wear (RPN=120)
- Feeds to: E-201, E-302

## Equipment: C-601 — Instrument Air Compressor
- Type: compressor (screw_compressor)
- Criticality: critical
- Design: 8.0 barg / None °C
- Material: 
- Status: running
- Instruments (2):
  - PT-601IA: pressure (0-10 barg) [H=8, HH=9, L=6, LL=5]
  - TT-601IA: temperature (0-120 °C) [H=90, HH=100]
- Known failure modes:
  - surge: Rapid flow reversal, severe vibration, potential mechanical damage (RPN=54)
  - bearing_failure: Vibration increase, temperature rise, potential catastrophic failure (RPN=135)
  - fouling: Reduced efficiency, higher discharge temperature, reduced capacity (RPN=210)

## Equipment: V-701 — Flare KO Drum
- Type: drum (N/A)
- Criticality: critical
- Design: 3.5 barg / None °C
- Material: 
- Status: running
- Receives from: V-101

## Equipment: XV-101 — Inlet ESD Valve
- Type: valve (ball_valve)
- Criticality: critical
- Design: None barg / None °C
- Material: 
- Status: running
- Known failure modes:
  - stuck: Valve fails to move to required position (RPN=120)
  - internal_leakage: Passing when closed, cannot achieve tight shutoff (RPN=150)

## Equipment: FV-301 — Amine Flow Control Valve
- Type: valve (control_valve)
- Criticality: high
- Design: None barg / None °C
- Material: 
- Status: running
- Known failure modes:
  - stuck: Valve fails to move to required position (RPN=120)
  - internal_leakage: Passing when closed, cannot achieve tight shutoff (RPN=150)