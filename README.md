## Files
- `crew_scheduling_data_gen.py` — Generates the synthetic dataset (SEED=42)
- `crew_scheduling_solver.py` — Solves the MILP across four DGCA regulatory scenarios (S1–S4)

## How to Run
1. Run `crew_scheduling_data_gen.py` first — this creates `crew_scheduling_dataset.xlsx`
2. Place `crew_scheduling_dataset.xlsx` in the same folder as the solver
3. Run `crew_scheduling_solver.py` — outputs `scenario_results.xlsx`

## Dependencies
Please refer to scripts, as dependecies are clearly given.

## Note
Dataset generated with SEED=42 (fixed for reproducibility).
Solver uses HiGHS via scipy.optimize.milp (Python 3.10+).
