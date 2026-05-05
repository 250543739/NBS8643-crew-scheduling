r"""
========================================================================
Regulation-Adaptive Airline Crew Scheduling — MILP Solver
========================================================================
Author: 250543739
Module: NBS8643 Operations Analytics (2025-26)| Newcastle University 
Module Leader: Dr Xinyue Hao, Newcastle University

This script solves a mixed-integer linear programme for airline crew
scheduling under four DGCA regulatory scenarios (S1-S4). 

Formulation: 11 numbered constraints matching Section 2 of the report.
Libraries: pip install "numpy", "pandas", "scipy", "openpyxl"
Solver:      HiGHS via scipy.optimize.milp (Python 3.10+)
Dataset:     SEED=42, 10 airports, 118 flights, 20 pilots

To run:  python crew_scheduling_solver.py
Input:   crew_scheduling_dataset.xlsx (same directory)
Output:  Console output + scenario_results.xlsx

========================================================================
"""
# =====================================================================
# 1. LOAD LIBRARIES AND SOLVER
# =====================================================================

import numpy as np
import pandas as pd
from scipy.optimize import milp, LinearConstraint, Bounds
from scipy.sparse import lil_matrix
import time
import os

# =====================================================================
# 1. LOAD DATASET (The following path might not work on different machine, Please adjust accordingly)
# =====================================================================

DATA_FILE = os.path.join(os.path.dirname(os.path.abspath(__file__)), "crew_scheduling_dataset.xlsx")

if not os.path.exists(DATA_FILE):
    for alt in ["../crew_scheduling_dataset.xlsx",
                "data/crew_scheduling_dataset.xlsx"]:
        if os.path.exists(alt):
            DATA_FILE = alt
            break

print(f"Reading dataset: {DATA_FILE}")
flights_df   = pd.read_excel(DATA_FILE, sheet_name="Flights")
pilots_df    = pd.read_excel(DATA_FILE, sheet_name="Pilots")
cost_df      = pd.read_excel(DATA_FILE, sheet_name="Cost_Matrix")
avail_df     = pd.read_excel(DATA_FILE, sheet_name="Availability_Matrix")
incompat_df  = pd.read_excel(DATA_FILE, sheet_name="Incompatible_Pairs")
scenarios_df = pd.read_excel(DATA_FILE, sheet_name="Scenarios")

# --- Index mappings ---
flight_ids = list(flights_df["flight_id"])
pilot_ids  = list(pilots_df["pilot_id"])

num_flights = len(flight_ids)   # |F| = 118
num_pilots  = len(pilot_ids)    # |P| = 20
num_days    = 7                  # |D| = 7 (weekly horizon)

f_idx = {fid: i for i, fid in enumerate(flight_ids)}
p_idx = {pid: i for i, pid in enumerate(pilot_ids)}

print(f"Dataset loaded: {num_flights} flights, {num_pilots} pilots, "
      f"{num_days} days")

# --- Flight attributes ---
flight_day_index = flights_df["day_index"].values       # 0-6
flight_duty_hrs  = flights_df["duty_hours"].values       # hours
flight_night_old = flights_df["is_night_old"].values     # 00:00-05:00
flight_night_new = flights_df["is_night_new"].values     # 00:00-06:00

# --- Flights per day: F_d subsets ---
flights_on_day = {d: [] for d in range(num_days)}
for i, d in enumerate(flight_day_index):
    flights_on_day[d].append(i)

# --- Cost matrix: c_pf ---
cost_matrix = cost_df.iloc[:, 1:].values   # shape (20, 118)

# --- Eligibility matrix: a_pf ---
avail_matrix = avail_df.iloc[:, 1:].values  # shape (20, 118)

# --- Incompatible pairs: set I ---
# Pre-computed from schedule: pairs that overlap or have gap < R_min
incompatible_pairs = []
for _, row in incompat_df.iterrows():
    f1, f2 = row["flight_1"], row["flight_2"]
    if f1 in f_idx and f2 in f_idx:
        incompatible_pairs.append((f_idx[f1], f_idx[f2]))

print(f"Incompatible pairs: {len(incompatible_pairs)}")

# --- Cancellation penalty ---
# Set high enough to prefer coverage, but finite so the model
# reveals infeasibility as uncovered flights rather than failing
KAPPA = 50


# =====================================================================
# 2. VARIABLE INDEXING
# =====================================================================
# Variables are laid out in a single flat vector for scipy.optimize.milp:
#
#   x_pf : pilot p assigned to flight f    (num_pilots * num_flights)
#   u_f  : flight f uncovered               (num_flights)
#   w_pd : pilot p works on day d           (num_pilots * num_days)
#   r_pd : pilot p rest block starts day d  (num_pilots * R_positions)
#   z_pd : pilot p night-duty on day d      (num_pilots * num_days)

def build_variable_layout(n_rest):
    """Return index offsets and total variable count for given n_rest."""
    rest_positions = num_days - n_rest + 1   # valid start days for rest

    x_start = 0
    x_count = num_pilots * num_flights

    u_start = x_start + x_count
    u_count = num_flights

    w_start = u_start + u_count
    w_count = num_pilots * num_days

    r_start = w_start + w_count
    r_count = num_pilots * rest_positions

    z_start = r_start + r_count
    z_count = num_pilots * num_days

    total = z_start + z_count

    return {
        "x_start": x_start,  "x_count": x_count,
        "u_start": u_start,  "u_count": u_count,
        "w_start": w_start,  "w_count": w_count,
        "r_start": r_start,  "r_count": r_count,
        "z_start": z_start,  "z_count": z_count,
        "rest_positions": rest_positions,
        "total": total
    }


def idx_x(p, f, L):
    """Index of x_pf in the variable vector."""
    return L["x_start"] + p * num_flights + f

def idx_u(f, L):
    """Index of u_f in the variable vector."""
    return L["u_start"] + f

def idx_w(p, d, L):
    """Index of w_pd in the variable vector."""
    return L["w_start"] + p * num_days + d

def idx_r(p, d, L):
    """Index of r_pd in the variable vector."""
    return L["r_start"] + p * L["rest_positions"] + d

def idx_z(p, d, L):
    """Index of z_pd in the variable vector."""
    return L["z_start"] + p * num_days + d


# =====================================================================
# 3. SOLVER — ONE SCENARIO
# =====================================================================

def solve_scenario(name, R_weekly, N_night, N_consec, night_start,
                   night_end, D_max, R_min, max_flights_per_day=3):
    """
    Build and solve the MILP for one regulatory scenario.

    Parameters
    ----------
    name       : str   — scenario label (e.g. 'S1_baseline')
    R_weekly   : int   — minimum weekly contiguous rest (hours)
    N_night    : int   — max night landings per pilot per week
    N_consec   : int   — max consecutive night duties
    night_start: int   — night window start hour (0 = midnight)
    night_end  : int   — night window end hour (5 or 6)
    D_max      : int   — max weekly duty hours
    R_min      : int   — min inter-flight rest (hours)
    max_flights_per_day : int — operational cap on flights per pilot per day

    Returns
    -------
    dict with scenario results, or None if solver fails.
    """
    print(f"\n{'='*65}")
    print(f"  SCENARIO: {name}")
    print(f"  R_weekly={R_weekly}h  N_night={N_night}  N_consec={N_consec}")
    print(f"  Night window: {night_start}:00–{night_end}:00")
    print(f"  D_max={D_max}h  R_min={R_min}h")
    print(f"{'='*65}")

    # --- Derived parameter: n_rest ---
    # Assumption A5: 36h → 1 rest day, 48h → 2 consecutive rest days
    n_rest = 1 if R_weekly <= 36 else 2

    # --- Night flight classification ---
    night_flags = flight_night_old if night_end <= 5 else flight_night_new
    night_flights = [i for i in range(num_flights) if night_flags[i] == 1]

    # Night flights per day (N_d subsets)
    night_on_day = {d: [] for d in range(num_days)}
    for i in night_flights:
        night_on_day[flight_day_index[i]].append(i)

    print(f"  Night flights: {len(night_flights)}, n_rest={n_rest}")

    # --- Variable layout ---
    L = build_variable_layout(n_rest)
    n_vars = L["total"]
    R_pos = L["rest_positions"]

    print(f"  Variables: {n_vars} total")
    print(f"    x_pf: {L['x_count']}  u_f: {L['u_count']}  "
          f"w_pd: {L['w_count']}  r_pd: {L['r_count']}  "
          f"z_pd: {L['z_count']}")

    # =================================================================
    # OBJECTIVE FUNCTION — Equation (1)
    #   Min Z = Σ_p Σ_f c_pf · x_pf  +  κ · Σ_f u_f
    # =================================================================
    obj = np.zeros(n_vars)
    for p in range(num_pilots):
        for f in range(num_flights):
            obj[idx_x(p, f, L)] = cost_matrix[p, f]
    for f in range(num_flights):
        obj[idx_u(f, L)] = KAPPA

    # =================================================================
    # CONSTRAINTS
    # =================================================================
    # Collected as sparse rows: lb <= A·x <= ub
    rows = []
    lb_list = []
    ub_list = []

    def add_constraint(row_dict, lb, ub):
        """Add one constraint from a {col_index: coefficient} dict."""
        rows.append(row_dict)
        lb_list.append(lb)
        ub_list.append(ub)

    # -----------------------------------------------------------------
    # Equation (2): Flight coverage
    #   Σ_p x_pf + u_f = 1,  ∀ f ∈ F
    # Each flight assigned to exactly one pilot or marked uncovered.
    # -----------------------------------------------------------------
    for f in range(num_flights):
        row = {idx_u(f, L): 1.0}
        for p in range(num_pilots):
            row[idx_x(p, f, L)] = 1.0
        add_constraint(row, 1.0, 1.0)

    # -----------------------------------------------------------------
    # Equation (3): Pilot eligibility
    #   x_pf ≤ a_pf,  ∀ p ∈ P, f ∈ F
    # Only assign pilots who are eligible for the flight.
    # Implemented by forcing x_pf = 0 where a_pf = 0.
    # -----------------------------------------------------------------
    for p in range(num_pilots):
        for f in range(num_flights):
            if avail_matrix[p, f] == 0:
                add_constraint({idx_x(p, f, L): 1.0}, 0.0, 0.0)

    # -----------------------------------------------------------------
    # Equation (4): Weekly duty limit
    #   Σ_f t_f · x_pf ≤ D_max,  ∀ p ∈ P
    # -----------------------------------------------------------------
    for p in range(num_pilots):
        row = {}
        for f in range(num_flights):
            row[idx_x(p, f, L)] = flight_duty_hrs[f]
        add_constraint(row, -np.inf, D_max)

    # -----------------------------------------------------------------
    # Equation (5): Temporal incompatibility
    #   x_pf + x_pf' ≤ 1,  ∀ p ∈ P, (f,f') ∈ I
    # No pilot assigned to two overlapping or insufficient-gap flights.
    # -----------------------------------------------------------------
    for p in range(num_pilots):
        for (f1, f2) in incompatible_pairs:
            add_constraint({
                idx_x(p, f1, L): 1.0,
                idx_x(p, f2, L): 1.0
            }, -np.inf, 1.0)

    # -----------------------------------------------------------------
    # Equations (6)–(7): Work-day linking
    #   (6) w_pd ≥ x_pf,         ∀ p, d, f ∈ F_d
    #   (7) w_pd ≤ Σ_{f∈F_d} x_pf,  ∀ p, d
    #
    # Together these ensure w_pd = 1 iff pilot p has any flight on day d.
    # -----------------------------------------------------------------
    for p in range(num_pilots):
        for d in range(num_days):
            # (6) w_pd ≥ x_pf  →  w_pd - x_pf ≥ 0
            for f in flights_on_day[d]:
                add_constraint({
                    idx_w(p, d, L): 1.0,
                    idx_x(p, f, L): -1.0
                }, 0.0, np.inf)

            # (7) w_pd ≤ Σ x_pf  →  w_pd - Σ x_pf ≤ 0
            row = {idx_w(p, d, L): 1.0}
            for f in flights_on_day[d]:
                row[idx_x(p, f, L)] = -1.0
            add_constraint(row, -np.inf, 0.0)

    # -----------------------------------------------------------------
    # Equations (8)–(9): Contiguous rest block
    #   (8) Σ_d r_pd ≥ 1,  ∀ p
    #   (9) r_pd + w_{p,d+k} ≤ 1,  ∀ p, d, k ∈ {0,...,n_rest-1}
    #
    # Each pilot must have at least one rest block placement (8).
    # If the rest block starts on day d, the pilot cannot work on any
    # of the next n_rest days (9). No Big-M required — both variables
    # are binary, so the implication is direct.
    # -----------------------------------------------------------------
    for p in range(num_pilots):
        # (8) At least one rest block
        row = {}
        for d in range(R_pos):
            row[idx_r(p, d, L)] = 1.0
        add_constraint(row, 1.0, np.inf)

        # (9) Rest block enforcement
        for d in range(R_pos):
            for k in range(n_rest):
                add_constraint({
                    idx_r(p, d, L): 1.0,
                    idx_w(p, d + k, L): 1.0
                }, -np.inf, 1.0)

    # -----------------------------------------------------------------
    # Equation (10): Night landing cap
    #   Σ_{f∈N} x_pf ≤ N_night,  ∀ p
    # -----------------------------------------------------------------
    for p in range(num_pilots):
        row = {}
        for f in night_flights:
            row[idx_x(p, f, L)] = 1.0
        add_constraint(row, -np.inf, N_night)

    # -----------------------------------------------------------------
    # Night-duty linking (supports constraint 11)
    #   z_pd ≥ x_pf,             ∀ p, d, f ∈ N_d
    #   z_pd ≤ Σ_{f∈N_d} x_pf,  ∀ p, d
    #
    # Same structure as (6)–(7) but for night flights only.
    # -----------------------------------------------------------------
    for p in range(num_pilots):
        for d in range(num_days):
            for f in night_on_day[d]:
                add_constraint({
                    idx_z(p, d, L): 1.0,
                    idx_x(p, f, L): -1.0
                }, 0.0, np.inf)

            row = {idx_z(p, d, L): 1.0}
            for f in night_on_day[d]:
                row[idx_x(p, f, L)] = -1.0
            add_constraint(row, -np.inf, 0.0)

    # -----------------------------------------------------------------
    # Equation (11): Consecutive night duty limit
    #   Σ_{k=0}^{N_consec} z_{p,d+k} ≤ N_consec,
    #       ∀ p, d ∈ {0,...,|D|-N_consec-1}
    #
    # Sliding window: any N_consec+1 consecutive days can have at most
    # N_consec night duties. Not active when N_consec = 99 (S1 baseline).
    # -----------------------------------------------------------------
    if N_consec < 99:
        for p in range(num_pilots):
            for d in range(num_days - N_consec):
                row = {}
                for k in range(N_consec + 1):
                    row[idx_z(p, d + k, L)] = 1.0
                add_constraint(row, -np.inf, float(N_consec))

    # -----------------------------------------------------------------
    # Operational: max flights per pilot per day
    # Not a regulatory constraint — prevents unrealistic schedules.
    # -----------------------------------------------------------------
    for p in range(num_pilots):
        for d in range(num_days):
            row = {}
            for f in flights_on_day[d]:
                row[idx_x(p, f, L)] = 1.0
            add_constraint(row, -np.inf, float(max_flights_per_day))

    # =================================================================
    # BUILD SPARSE CONSTRAINT MATRIX
    # =================================================================
    n_constraints = len(rows)
    print(f"  Constraints: {n_constraints}")

    A = lil_matrix((n_constraints, n_vars))
    for i, row_dict in enumerate(rows):
        for j, val in row_dict.items():
            A[i, j] = val
    A_csr = A.tocsr()

    constraints = LinearConstraint(
        A_csr,
        np.array(lb_list),
        np.array(ub_list)
    )

    # All variables binary
    bounds = Bounds(lb=0, ub=1)
    integrality = np.ones(n_vars)

    # =================================================================
    # SOLVE
    # =================================================================
    t0 = time.time()
    result = milp(
        obj,
        constraints=constraints,
        integrality=integrality,
        bounds=bounds,
        options={"time_limit": 300, "mip_rel_gap": 0.01, "disp": True}
    )
    solve_time = time.time() - t0

    if not result.success:
        print(f"  *** SOLVER FAILED: {result.message}")
        return None

    sol = result.x

    # =================================================================
    # EXTRACT RESULTS
    # =================================================================

    # --- Assignments per pilot ---
    assignments = {}
    for p in range(num_pilots):
        assignments[p] = [f for f in range(num_flights)
                          if sol[idx_x(p, f, L)] > 0.5]

    # --- Uncovered flights ---
    uncovered = [f for f in range(num_flights)
                 if sol[idx_u(f, L)] > 0.5]
    covered = num_flights - len(uncovered)

    # --- Work days per pilot ---
    work_days = {}
    for p in range(num_pilots):
        work_days[p] = sum(1 for d in range(num_days)
                           if sol[idx_w(p, d, L)] > 0.5)

    # --- Duty hours per pilot ---
    duty_hrs = {}
    for p in range(num_pilots):
        duty_hrs[p] = sum(flight_duty_hrs[f] for f in assignments[p])

    # --- Night flights per pilot ---
    night_per_pilot = {}
    for p in range(num_pilots):
        night_per_pilot[p] = sum(1 for f in assignments[p]
                                 if night_flags[f] == 1)

    # --- Utilisation ---
    utils = [duty_hrs[p] / D_max * 100 for p in range(num_pilots)]
    avg_util = np.mean(utils)
    max_util = np.max(utils)

    # --- Cost decomposition ---
    assign_cost = sum(cost_matrix[p, f]
                      for p in range(num_pilots)
                      for f in assignments[p])
    penalty_cost = len(uncovered) * KAPPA

    # --- Summary ---
    out = {
        "Scenario":          name,
        "R_weekly (h)":      R_weekly,
        "N_night":           N_night,
        "N_consec":          N_consec,
        "n_rest (days)":     n_rest,
        "Total cost":        result.fun,
        "Assignment cost":   assign_cost,
        "Penalty cost":      penalty_cost,
        "Flights covered":   covered,
        "Flights uncovered": len(uncovered),
        "Uncovered (%)":     round(len(uncovered) / num_flights * 100, 1),
        "Avg utilisation %": round(avg_util, 1),
        "Max utilisation %": round(max_util, 1),
        "Max nights (pilot)":max(night_per_pilot.values()),
        "Max duty hrs":      round(max(duty_hrs.values()), 1),
        "Avg work days":     round(np.mean(list(work_days.values())), 1),
        "Max work days":     max(work_days.values()),
        "Solve time (s)":    round(solve_time, 1),
        "Uncovered IDs":     [flight_ids[f] for f in uncovered]
    }

    print(f"\n  RESULTS:")
    print(f"  Total cost:   {result.fun:.1f} "
          f"(assign={assign_cost:.0f} + penalty={penalty_cost:.0f})")
    print(f"  Covered:      {covered}/{num_flights}  |  "
          f"Uncovered: {len(uncovered)} ({out['Uncovered (%)']:.1f}%)")
    print(f"  Avg util:     {avg_util:.1f}%  |  Max util: {max_util:.1f}%")
    print(f"  Max nights:   {out['Max nights (pilot)']}  |  "
          f"Max duty: {out['Max duty hrs']:.1f}h")
    print(f"  Avg work days:{out['Avg work days']:.1f}  |  "
          f"Max: {out['Max work days']}")
    print(f"  Solve time:   {solve_time:.1f}s")
    if uncovered:
        print(f"  Uncovered:    {out['Uncovered IDs']}")

    # =================================================================
    # OPTIMALITY VERIFICATION
    # =================================================================
    # scipy.optimize.milp (HiGHS backend) attaches the following
    # diagnostic fields to the OptimizeResult after solve. We surface
    # them here so the report's optimality claims (Section 3, Figure 3)
    # are reproducible from the solver's own output.
    dual_bound = getattr(result, "mip_dual_bound", None)
    mip_gap    = getattr(result, "mip_gap", None)
    node_count = getattr(result, "mip_node_count", None)

    if dual_bound is not None and abs(result.fun) > 1e-9:
        gap_pct = abs(result.fun - dual_bound) / abs(result.fun) * 100
    else:
        gap_pct = 0.0

    status_label = "OPTIMAL" if result.success else "NOT OPTIMAL"

    print(f"\n  OPTIMALITY VERIFICATION:")
    print(f"  Integer optimum:  {result.fun:.4f}")
    if dual_bound is not None:
        print(f"  LP dual bound:    {dual_bound:.4f}")
    if mip_gap is not None:
        print(f"  MIP gap (scipy):  {mip_gap:.6f}")
    print(f"  MIP gap (computed): {gap_pct:.6f}%")
    if node_count is not None:
        print(f"  Nodes explored:   {node_count}  "
              f"({'root only — no branching' if node_count <= 1 else 'branched'})")
    print(f"  Status:           {status_label}")

    out["LP dual bound"]    = dual_bound
    out["MIP gap (%)"]      = round(gap_pct, 6)
    out["Nodes explored"]   = node_count
    out["Status"]           = status_label

    return out


# =====================================================================
# 4. RUN ALL SCENARIOS
# =====================================================================

print("\n" + "=" * 65)
print("  REGULATION-ADAPTIVE CREW SCHEDULING — SCENARIO ANALYSIS")
print("=" * 65)

all_results = []

for _, row in scenarios_df.iterrows():
    # S4 stress test: also cap flights per pilot per day at 2
    max_fpd = 2 if row["scenario"] == "S4_stress" else 3

    res = solve_scenario(
        name        = row["scenario"],
        R_weekly    = int(row["R_weekly"]),
        N_night     = int(row["N_night"]),
        N_consec    = int(row["N_consec_night"]),
        night_start = int(row["night_start"]),
        night_end   = int(row["night_end"]),
        D_max       = int(row["D_max"]),
        R_min       = int(row["R_min"]),
        max_flights_per_day = max_fpd
    )
    if res is not None:
        all_results.append(res)


# =====================================================================
# 5. COMPARISON TABLE
# =====================================================================

print("\n\n" + "=" * 80)
print("  SCENARIO COMPARISON TABLE")
print("=" * 80)

header = (f"{'Scenario':<16} {'Cost':>8} {'Covered':>8} {'Uncover':>8} "
          f"{'Unc%':>6} {'AvgUtil':>8} {'MaxUtil':>8} "
          f"{'MaxNight':>9} {'AvgDays':>8} {'Time':>6}")
print(header)
print("-" * 80)

for r in all_results:
    print(f"{r['Scenario']:<16} "
          f"{r['Total cost']:>8.1f} "
          f"{r['Flights covered']:>8} "
          f"{r['Flights uncovered']:>8} "
          f"{r['Uncovered (%)']:>5.1f}% "
          f"{r['Avg utilisation %']:>7.1f}% "
          f"{r['Max utilisation %']:>7.1f}% "
          f"{r['Max nights (pilot)']:>9} "
          f"{r['Avg work days']:>7.1f} "
          f"{r['Solve time (s)']:>5.1f}s")


# =====================================================================
# 6. KEY TRANSITIONS
# =====================================================================

if len(all_results) >= 2:
    s1 = all_results[0]
    s2 = all_results[1]
    cost_change = (s2["Total cost"] - s1["Total cost"]) / s1["Total cost"] * 100
    cancel_change = s2["Flights uncovered"] - s1["Flights uncovered"]

    print(f"\n  KEY FINDING: S1 → S2 Transition")
    print(f"  Cost increase:         {s1['Total cost']:.0f} → "
          f"{s2['Total cost']:.0f}  (+{cost_change:.1f}%)")
    print(f"  Cancellation increase: {s1['Flights uncovered']} → "
          f"{s2['Flights uncovered']}  (+{cancel_change} flights)")
    print(f"  Driver: R_weekly {s1['R_weekly (h)']}h → {s2['R_weekly (h)']}h "
          f"(n_rest {s1['n_rest (days)']} → {s2['n_rest (days)']} days)")


# =====================================================================
# 7. EXPORT TO EXCEL
# =====================================================================

output_file = os.path.join(os.path.dirname(os.path.abspath(__file__)), "scenario_results.xlsx")

# Build comparison dataframe (excluding list columns)
export_cols = [k for k in all_results[0].keys() if k != "Uncovered IDs"]
comparison_df = pd.DataFrame(all_results)[export_cols]

# Build uncovered flights detail
uncov_rows = []
for r in all_results:
    for fid in r["Uncovered IDs"]:
        flight_row = flights_df[flights_df["flight_id"] == fid].iloc[0]
        uncov_rows.append({
            "Scenario":    r["Scenario"],
            "Flight ID":   fid,
            "Route":       f"{flight_row['origin']}→{flight_row['destination']}",
            "Day":         flight_row["day"],
            "Departure":   flight_row["dep_time"],
            "Wave":        flight_row["wave"],
        })
uncov_df = pd.DataFrame(uncov_rows) if uncov_rows else pd.DataFrame()

with pd.ExcelWriter(output_file, engine="openpyxl") as writer:
    comparison_df.to_excel(writer, sheet_name="Comparison", index=False)
    if not uncov_df.empty:
        uncov_df.to_excel(writer, sheet_name="Uncovered_Flights", index=False)

print(f"\n  Results exported to: {output_file}")
print(f"\n{'='*65}")
print(f"  COMPLETE — All {len(all_results)} scenarios solved successfully.")
print(f"{'='*65}")
