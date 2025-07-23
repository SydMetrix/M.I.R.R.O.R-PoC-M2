# M2: Glitch Pattern Recognition (M.I.R.R.O.R Protocol)

## 📌 Overview

This module detects **reflective logic anomalies** (glitches) in AI-generated responses, identifying structurally malformed or deceptive reasoning patterns before they escalate into hallucinations or belief loops.

**M2** acts as the **pre-reflective diagnostic layer** in the M.I.R.R.O.R Protocol.

---

## 🧠 Objectives

- Identify **Glitch Texture Patterns (GTP)**: malformed logic structures such as belief recursion, semantic substitution, and skewed reasoning.
- Compute **Cognitive Stability Index (CSI)** for each prompt–response pair.
- Route critical cases to downstream corrective systems:
  - **SAHL**: Subsystem Anti-Hallucination Layer.
  - **ARP-X**: Axis Reconstruction Protocol.

---

## 🔍 Detection Mechanism

### Glitch Types:
- `SemanticSubLoop`: Substitution of genuine logic with trained stylistic simulation.
- `SelfAffirmingTrap`: Recursive logic affirming its own assumption without contradiction.
- `SkewedMirror`: Surface-level coherence hiding internal misalignment.

### CSI Calculation:
- Penalizes:
  - High semantic overlap (logic echo)
  - Detected substitution or recursion
- Produces a score in `[0.0 – 1.0]`:
  - 🟢 Stable (≥ 0.8)
  - 🟡 Mild Divergence (0.6–0.8)
  - 🟠 Medium Divergence (0.4–0.6)
  - 🔴 High Divergence (< 0.4)

---

## 🛠️ Features

- ✅ **Semantic Role Embedding** (via `spaCy` and `transformers`)
- ✅ **Glitch Signature Construction**: `GTP::[Type]+[Type]+...`
- ✅ **Reflective Routing** logic (`route_sahl`, `route_arp_x`)
- ✅ **FAISS**-backed vector matching for glitch pattern indexing
- ✅ **Resilient JSON saving** with automatic corruption handling

---

## 🔧 How to Use

```bash
$ python GPR.py
