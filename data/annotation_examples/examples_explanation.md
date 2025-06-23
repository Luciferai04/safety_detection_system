# Annotation Examples Explanation

This directory contains example annotations for different scenarios in thermal power plants.

## compliant_worker_boiler.txt
**Description:** Worker in boiler area with full PPE

**Scenario Type:** compliant

**Annotations:**
1. `2 0.45 0.65 0.35 0.7` - person
2. `0 0.45 0.28 0.18 0.22` - helmet
3. `1 0.45 0.52 0.32 0.45` - reflective_jacket

## violation_worker_control.txt
**Description:** Worker in control room without helmet

**Scenario Type:** violation

**Annotations:**
1. `2 0.6 0.55 0.28 0.8` - person
2. `1 0.6 0.45 0.25 0.4` - reflective_jacket

## multiple_workers_maintenance.txt
**Description:** Multiple workers with varying compliance

**Scenario Type:** mixed_compliance

**Annotations:**
1. `2 0.25 0.6 0.22 0.65` - person
2. `0 0.25 0.32 0.15 0.18` - helmet
3. `1 0.25 0.48 0.2 0.35` - reflective_jacket
4. `2 0.55 0.7 0.25 0.6` - person
5. `1 0.55 0.58 0.22 0.32` - reflective_jacket
6. `2 0.8 0.45 0.15 0.35` - person
7. `0 0.8 0.32 0.08 0.1` - helmet
8. `1 0.8 0.42 0.12 0.2` - reflective_jacket

## challenging_steam_conditions.txt
**Description:** Workers in steamy boiler area - challenging visibility

**Scenario Type:** challenging_conditions

**Annotations:**
1. `2 0.4 0.55 0.3 0.7` - person
2. `0 0.4 0.28 0.16 0.2` - helmet
3. `1 0.4 0.48 0.28 0.4` - reflective_jacket

## distant_switchyard_workers.txt
**Description:** Workers at distance in switchyard

**Scenario Type:** distant_workers

**Annotations:**
1. `2 0.15 0.75 0.12 0.3` - person
2. `0 0.15 0.65 0.06 0.08` - helmet
3. `1 0.15 0.72 0.1 0.15` - reflective_jacket
4. `2 0.85 0.68 0.1 0.25` - person
5. `0 0.85 0.6 0.05 0.07` - helmet

