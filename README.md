# SATLocalSearchLab 🧩  
**Local Search Algorithms for Boolean SAT — GSAT, WalkSAT, HSAT & Variants**

SATLocalSearchLab is a Python implementation and benchmarking framework for **local search algorithms applied to Boolean SAT problems**.  
It focuses on comparing **GSAT, GWSAT, HSAT, WalkSAT, Tabu Search**, and **custom heuristic variants** on standard CNF benchmark instances.

This project was developed as part of an **AI / Metaheuristic Optimization assignment**, with an emphasis on:
- heuristic design
- performance evaluation
- parameter sensitivity analysis
- reproducible experimentation

---

## 🧠 Algorithms Implemented

### Core SAT Local Search Algorithms
- **GSAT**
- **GWSAT** (GSAT with random walk)
- **HSAT** (tie-breaking by age)
- **WalkSAT**

### Advanced & Extended Variants
- **HSAT with Tabu Search**
- **Grimes’ HSAT variant**
- **Grimes’ WalkSAT variant**

Each algorithm uses:
- make / break counts
- unsatisfied clause tracking
- randomized restarts
- configurable stopping conditions

---

## 🗂 Problem Representation

SAT instances are expected in **DIMACS CNF format**.

Internal data structures include:
- `state`: current variable assignment
- `clauses`: list of clauses
- `unsat_clauses`: indices of unsatisfied clauses
- `makecounts`: number of clauses that would become satisfied
- `breakcounts`: number of clauses that would become unsatisfied
- `litToClauses`: literal → clause index mapping
- `lastFlip`: tabu / age tracking

Variables are indexed **1..n** (index 0 unused) for direct mapping to CNF variables.

---

## ⚙️ Requirements

### Python
```bash
Python 3.8+
