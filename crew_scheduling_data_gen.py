"""
NBS8643 Operations Analytics — Data Generation Script
======================================================

Author: 250543739
Module: NBS8643 Operations Analytics (2025-26)| Newcastle University 
Module Leader: Dr Xinyue Hao, Newcastle University

This script generates synthetic data for a regulation-adaptive MILP model
for airline crew scheduling. All base values are derived from real-world
sources.

Inspired by the sample random data python script which was uploaded in the canvas

Libraries: pip install "numpy", "pandas", "openpyxl"

# ---------------------------------------------------------------
# Note FOR DIFFERENT MACHINES:
# The line below automatically saves the file in the same folder
# as this script — which works fine when run from a terminal.
#
# However, if you are using a Jupyter notebook or get a NameError
# mentioning __file__, just swap it out for one of these instead:
#
#   Simple fix (saves in whatever folder you're currently in):
#   out_path = os.path.join(os.getcwd(), "crew_scheduling_dataset.xlsx")
#
#   Or paste your own folder path directly:
#   out_path = r"C:\Users\YourName\Desktop\crew_scheduling_dataset.xlsx"  # Windows
#   out_path = "/Users/YourName/Desktop/crew_scheduling_dataset.xlsx"     # Mac

"""

import os
import numpy as np   # Numerical computing (random numbers, arrays)
import pandas as pd  # Tabular data (DataFrames) and Excel export
from itertools import combinations  # For generating flight pair comparisons

# =============================================================================
# SECTION 0: MASTER SEED — Fix once happy, never change :)
# =============================================================================
SEED = 42  # Reproducibility seed (fixing our seed once)

# =============================================================================
# SECTION 1: AIRPORT NETWORK
# =============================================================================
# 10 real IndiGo domestic airports
# Source: goindigo.in route map; Aviation A2Z (2025.
# Hub designation based on IndiGo's primary bases (DEL, BOM, BLR)

AIRPORTS = pd.DataFrame([
    {"code": "DEL", "city": "New Delhi",   "role": "hub",   "base_pilots": 6},
    {"code": "BOM", "city": "Mumbai",      "role": "hub",   "base_pilots": 5},
    {"code": "BLR", "city": "Bengaluru",   "role": "hub",   "base_pilots": 4},
    {"code": "HYD", "city": "Hyderabad",   "role": "spoke", "base_pilots": 2},
    {"code": "MAA", "city": "Chennai",     "role": "spoke", "base_pilots": 2},
    {"code": "CCU", "city": "Kolkata",     "role": "spoke", "base_pilots": 1},
    {"code": "AMD", "city": "Ahmedabad",   "role": "spoke", "base_pilots": 0},
    {"code": "PNQ", "city": "Pune",        "role": "spoke", "base_pilots": 0},
    {"code": "GOI", "city": "Goa",         "role": "spoke", "base_pilots": 0},
    {"code": "JAI", "city": "Jaipur",      "role": "spoke", "base_pilots": 0},
])
# Total pilots: 6+5+4+2+2+1 = 20 (within 15-25 range per Dr Hao's requirement)

# =============================================================================
# SECTION 2: ROUTE NETWORK WITH REAL FLIGHT TIMES
# =============================================================================
# Flight durations (minutes) from goindigo.in official schedule pages
# Distance (km) from public aviation data
# daily_freq_oneway = representative scaled frequency for our sub-network
#   (real IndiGo has 2200+ daily flights; we model a subset of ~15/day)
# route_type: hub-hub, hub-spoke, or spoke-spoke (determines frequency tier)

ROUTES = pd.DataFrame([
    # Hub-Hub routes (highest frequency)
    {"origin": "DEL", "dest": "BOM", "flight_time_min": 130, "distance_km": 1163, "daily_freq": 2, "route_type": "hub-hub"},
    {"origin": "DEL", "dest": "BLR", "flight_time_min": 165, "distance_km": 1740, "daily_freq": 2, "route_type": "hub-hub"},
    {"origin": "BOM", "dest": "BLR", "flight_time_min": 100, "distance_km":  981, "daily_freq": 1, "route_type": "hub-hub"},

    # Hub-Spoke routes (medium frequency — most daily, some 3-4x/week)
    {"origin": "DEL", "dest": "HYD", "flight_time_min": 140, "distance_km": 1260, "daily_freq": 1, "route_type": "hub-spoke"},
    {"origin": "DEL", "dest": "MAA", "flight_time_min": 170, "distance_km": 1760, "daily_freq": 1, "route_type": "hub-spoke"},
    {"origin": "DEL", "dest": "CCU", "flight_time_min": 135, "distance_km": 1305, "daily_freq": 1, "route_type": "hub-spoke"},
    {"origin": "DEL", "dest": "AMD", "flight_time_min": 105, "distance_km":  776, "daily_freq": 0, "route_type": "hub-spoke"},  # 3-4x/week
    {"origin": "DEL", "dest": "JAI", "flight_time_min":  65, "distance_km":  260, "daily_freq": 1, "route_type": "hub-spoke"},
    {"origin": "BOM", "dest": "HYD", "flight_time_min":  80, "distance_km":  620, "daily_freq": 1, "route_type": "hub-spoke"},
    {"origin": "BOM", "dest": "MAA", "flight_time_min": 115, "distance_km": 1040, "daily_freq": 0, "route_type": "hub-spoke"},  # 3-4x/week
    {"origin": "BOM", "dest": "GOI", "flight_time_min":  70, "distance_km":  440, "daily_freq": 1, "route_type": "hub-spoke"},
    {"origin": "BOM", "dest": "AMD", "flight_time_min":  90, "distance_km":  443, "daily_freq": 0, "route_type": "hub-spoke"},  # 3-4x/week
    {"origin": "BOM", "dest": "PNQ", "flight_time_min":  50, "distance_km":  150, "daily_freq": 1, "route_type": "hub-spoke"},
    {"origin": "BLR", "dest": "HYD", "flight_time_min":  70, "distance_km":  495, "daily_freq": 1, "route_type": "hub-spoke"},
    {"origin": "BLR", "dest": "MAA", "flight_time_min":  60, "distance_km":  290, "daily_freq": 1, "route_type": "hub-spoke"},
    {"origin": "BLR", "dest": "CCU", "flight_time_min": 165, "distance_km": 1560, "daily_freq": 0, "route_type": "hub-spoke"},  # 3-4x/week

    # Spoke-Spoke routes (lower frequency — not every day)
    {"origin": "MAA", "dest": "HYD", "flight_time_min":  60, "distance_km":  520, "daily_freq": 0, "route_type": "spoke-spoke"},
    {"origin": "HYD", "dest": "CCU", "flight_time_min": 130, "distance_km": 1200, "daily_freq": 0, "route_type": "spoke-spoke"},
    {"origin": "GOI", "dest": "DEL", "flight_time_min": 145, "distance_km": 1520, "daily_freq": 0, "route_type": "spoke-spoke"},
    {"origin": "JAI", "dest": "BOM", "flight_time_min": 120, "distance_km":  950, "daily_freq": 0, "route_type": "spoke-spoke"},
])

# =============================================================================
# SECTION 3: REGULATORY PARAMETERS (DGCA CAR S7/J/III Rev 2)
# =============================================================================
# These change across scenarios — the core of the regulation-adaptive model

SCENARIOS = {
    "S1_baseline": {
        "description": "Pre-amendment rules",
        "R_weekly": 36,          # Weekly contiguous rest (hours)
        "N_night": 6,            # Max night landings per week
        "N_consec_night": 99,    # No specific cap (effectively unlimited)
        "night_start": 0,        # Night window start (hour, 0 = midnight)
        "night_end": 5,          # Night window end (hour)
        "D_max": 60,             # Max weekly flight duty hours
        "R_min": 10,             # Min inter-flight rest (hours)
    },
    "S2_partial": {
        "description": "Only rest tightened",
        "R_weekly": 48,
        "N_night": 6,
        "N_consec_night": 2,
        "night_start": 0,
        "night_end": 5,
        "D_max": 60,
        "R_min": 10,
    },
    "S3_full_dgca": {
        "description": "Full DGCA 2025 regulation",
        "R_weekly": 48,
        "N_night": 2,
        "N_consec_night": 2,
        "night_start": 0,
        "night_end": 6,          # Extended night window
        "D_max": 60,
        "R_min": 12,             # Tightened rest
    },
    "S4_stress": {
        "description": "Stress test — even stricter",
        "R_weekly": 48,
        "N_night": 1,
        "N_consec_night": 2,
        "night_start": 0,
        "night_end": 6,
        "D_max": 60,
        "R_min": 12,
    },
}

# =============================================================================
# SECTION 4: FLIGHT SCHEDULE GENERATION
# =============================================================================
# Methodology: Structured wave-based scheduling with noise
# Approach Followed: base → structure → noise → cap

def generate_flight_schedule(
    seed: int = SEED,
    day_noise: float = 0.12,    # Day-of-week demand variability (~12%)
    time_jitter_min: int = 15,  # Max departure time jitter in minutes
):
    """
    Generate a weekly flight schedule for the 10-airport sub-network.

    Target: 80-120 flights/week (Dr Hao requirement), ~14-17 per day.

    Flight waves mirror real airline scheduling patterns:
      Morning (06:00-09:30): ~35% — business travel peak
      Midday  (11:00-14:30): ~20% — moderate traffic
      Evening (17:00-20:30): ~30% — return-home peak
      Night   (21:00-00:30): ~15% — red-eye / connecting

    Source: Barnhart, Belobaba & Odoni (2003); IndiGo published timetables
    """
    rng = np.random.default_rng(seed)  # Reproducible RNG 

    days = ["Mon", "Tue", "Wed", "Thu", "Fri", "Sat", "Sun"]

    # Day-of-week demand multipliers (busier Fri/Sun, quieter Tue)
    # Source: Standard airline demand patterns. 
    day_demand_boost = {
        "Mon": 1.00, "Tue": 0.88, "Wed": 0.95,
        "Thu": 1.00, "Fri": 1.12, "Sat": 0.90, "Sun": 1.08
    }

    # Wave windows: (earliest_dep_hour, latest_dep_hour)
    wave_windows = {
        "morning": (6.0,   9.5),   # 06:00 - 09:30
        "midday":  (11.0, 14.5),   # 11:00 - 14:30
        "evening": (17.0, 20.5),   # 17:00 - 20:30
        "night":   (21.0, 24.5),   # 21:00 - 00:30 (next day)
    }

    # Wave probabilities for random assignment (sum to 1.0)
    wave_names = ["morning", "midday", "evening", "night"]
    wave_probs = [0.35, 0.20, 0.30, 0.15]

    flights = []  # Master list to collect all flights
    flight_id = 1

    for day_idx, day in enumerate(days):
        boost = day_demand_boost[day]

        for _, route in ROUTES.iterrows():
            base_freq = route["daily_freq"]

            # For routes with freq=0 (not daily), randomly include ~3 days/week
            if base_freq == 0:
                if rng.random() < 3.0 / 7.0:  # ~3 days per week
                    effective_freq = 1
                else:
                    continue  # Skip this route today
            else:
                # Apply day demand boost with noise (Dr Hao's method)
                noisy_freq = base_freq * boost
                noisy_freq *= rng.normal(1.0, day_noise)  # Normal noise, SD=12%
                noisy_freq = np.clip(noisy_freq, base_freq * 0.75, base_freq * 1.35)  # Cap
                effective_freq = max(1, int(np.round(noisy_freq)))

            # Assign EACH flight to a wave using weighted random choice
            # (random but structured)
            assigned_waves = rng.choice(wave_names, size=effective_freq, p=wave_probs)

            for wave_name in assigned_waves:
                wave_start, wave_end = wave_windows[wave_name]

                # Base departure time: uniform within wave window
                base_dep_hour = rng.uniform(wave_start, wave_end)

                # Add time jitter (±jitter minutes) — operational variability
                jitter = rng.integers(-time_jitter_min, time_jitter_min + 1)
                dep_hour = base_dep_hour + (jitter / 60.0)
                dep_hour = max(0.5, dep_hour)  # Don't go before 00:30

                # Flight duration with small operational variance (±10 min)
                base_duration = route["flight_time_min"]
                duration_noise = rng.integers(-10, 11)  # ±10 min
                actual_duration = max(base_duration - 15, base_duration + duration_noise)

                # Arrival time
                arr_hour = dep_hour + (actual_duration / 60.0)

                # Flight duty period = flight time + 60 min pre/post duties
                # Source: DGCA FDP calculation methodology
                duty_hours = (actual_duration + 60) / 60.0

                # Convert to readable times
                dep_h = int(dep_hour) % 24
                dep_m = int((dep_hour % 1) * 60)
                arr_h = int(arr_hour) % 24
                arr_m = int((arr_hour % 1) * 60)

                dep_time_str = f"{dep_h:02d}:{dep_m:02d}"
                arr_time_str = f"{arr_h:02d}:{arr_m:02d}"

                # Store raw arrival hour for night classification at solve time
                arr_hour_of_day = arr_hour % 24

                # Night landing flag (under NEW DGCA rule: 00:00-06:00)
                is_night_new = 1 if (arr_hour_of_day < 6.0) else 0
                # Night landing flag (under OLD DGCA rule: 00:00-05:00)
                is_night_old = 1 if (arr_hour_of_day < 5.0) else 0

                flights.append({
                    "flight_id": f"F{flight_id:03d}",
                    "origin": route["origin"],
                    "destination": route["dest"],
                    "day": day,
                    "day_index": day_idx,
                    "dep_time": dep_time_str,
                    "arr_time": arr_time_str,
                    "dep_hour": round(dep_hour % 24, 2),
                    "arr_hour": round(arr_hour_of_day, 2),
                    "duration_min": actual_duration,
                    "duty_hours": round(duty_hours, 2),
                    "wave": wave_name,
                    "is_night_old": is_night_old,  # 00:00-05:00
                    "is_night_new": is_night_new,  # 00:00-06:00
                    "route_type": route["route_type"],
                    "distance_km": route["distance_km"],
                })
                flight_id += 1

    flights_df = pd.DataFrame(flights)

    # Sort by day_index then departure time for clean output
    flights_df = flights_df.sort_values(
        by=["day_index", "dep_hour"], ignore_index=True
    )

    return flights_df


# =============================================================================
# SECTION 5: PILOT GENERATION
# =============================================================================
# Methodology: Base assignment from hub distribution + binomial availability noise
# (staff supply with attendance probabilities)

def generate_pilots(
    seed: int = SEED,
    num_pilots: int = 20,
    # Attendance probabilities 
    weekday_att_range: tuple = (0.88, 0.98),  # Pilots more reliable than agency staff
    weekend_att_range: tuple = (0.82, 0.95),  # Slightly lower on weekends
):
    """
    Generate pilot roster with base airports and weekly availability.

    Pilot-to-aircraft ratio: 7.6 at IndiGo (lowest in India)
    Source: Government data to Lok Sabha, March 2026 (Business Standard)

    Availability modelled using Binomial distribution:
      available ~ Bernoulli(attendance_prob) per pilot per day
    """
    rng = np.random.default_rng(seed)

    days = ["Mon", "Tue", "Wed", "Thu", "Fri", "Sat", "Sun"]

    # Build pilot list with base airport assignments
    # Distribution follows hub traffic volume (more pilots at busier bases)
    pilot_bases = []
    for _, airport in AIRPORTS.iterrows():
        for _ in range(airport["base_pilots"]):
            pilot_bases.append(airport["code"])

    pilots = []
    for i, base in enumerate(pilot_bases):
        pilot_id = f"P{i+1:02d}"

        # Generate daily availability using Bernoulli (binomial with n=1)
        availability = {}
        for d in days:
            if d in ("Sat", "Sun"):
                att_prob = rng.uniform(*weekend_att_range)
            else:
                att_prob = rng.uniform(*weekday_att_range)

            # Bernoulli: 1 = available, 0 = unavailable (training, leave, sick)
            available = int(rng.binomial(1, att_prob))
            availability[d] = available

        pilots.append({
            "pilot_id": pilot_id,
            "base_airport": base,
            "max_weekly_duty_hrs": 60,  # DGCA CAR S7/J/III (unchanged across scenarios)
            **{f"avail_{d}": availability[d] for d in days},
        })

    pilots_df = pd.DataFrame(pilots)
    return pilots_df


# =============================================================================
# SECTION 6: COST MATRIX GENERATION
# =============================================================================
# Assignment cost c_{p,f} based on:
#   - Flight duration (longer = costlier)
#   - Base match (deadhead if pilot not at origin — +50% premium)
#   - Night premium (+20% for night-window arrivals)
#   - Normalised to integer scale 1-10
# Sources: IndiGo pilot salary data; Barnhart et al. (2003)

def generate_cost_matrix(flights_df, pilots_df, seed: int = SEED):
    """
    Generate pilot-flight assignment cost matrix.

    Cost components (normalised to 1-10 integer scale):
      Base cost:     proportional to duty_hours (1-4)
      Deadhead:      +3 if pilot base ≠ flight origin
      Night premium: +1 if arrival in night window (00:00-06:00)
      Small noise:   ±0.5 random variation (operational variability)

    Uncovered flight penalty (M_u): 50 (5-10× most expensive assignment)
    Source: EUROCONTROL Standard Inputs Ed. 10 (2024), Chapter 14
    """
    rng = np.random.default_rng(seed + 100)  # Offset seed to avoid correlation with schedule

    cost_records = []

    for _, pilot in pilots_df.iterrows():
        for _, flight in flights_df.iterrows():
            # Base cost: proportional to duty hours (scale 1-4)
            # Short flight (~1.5h duty): cost ~1
            # Long flight (~4h duty):    cost ~4
            base_cost = np.clip(flight["duty_hours"] / 1.2, 1, 4)

            # Deadhead premium: +3 if pilot is not based at flight origin
            if pilot["base_airport"] != flight["origin"]:
                deadhead = 3
            else:
                deadhead = 0

            # Night premium: +1 if flight is a night landing (new DGCA rule)
            if flight["is_night_new"] == 1:
                night_premium = 1
            else:
                night_premium = 0

            # Small random noise (±0.5) — operational variability
            noise = rng.uniform(-0.5, 0.5)

            # Total cost (round to integer, clip to 1-10)
            total = base_cost + deadhead + night_premium + noise
            total = int(np.clip(np.round(total), 1, 10))

            cost_records.append({
                "pilot_id": pilot["pilot_id"],
                "flight_id": flight["flight_id"],
                "cost": total,
            })

    cost_df = pd.DataFrame(cost_records)

    # Pivot to matrix form: rows=pilots, columns=flights
    cost_matrix = cost_df.pivot(
        index="pilot_id", columns="flight_id", values="cost"
    )

    return cost_matrix


# =============================================================================
# SECTION 7: AVAILABILITY MATRIX
# =============================================================================
# A_{p,f} = 1 if pilot p can fly flight f, 0 otherwise
# Conditions: (1) pilot available that day, (2) pilot based at origin OR
# could reasonably reposition (hub connectivity)

def generate_availability_matrix(flights_df, pilots_df):
    """
    Generate pilot-flight availability matrix A_{p,f}.

    A pilot is eligible for a flight if:
      1. Available on that day (from pilot availability schedule)
      2. Based at the flight's origin airport, OR
         based at a hub connected to the origin (repositioning possible)

    Hub connectivity assumption: pilots at DEL/BOM/BLR can reposition
    to any other airport (at higher cost, captured in cost matrix).
    Spoke-based pilots can only fly from their base.
    """
    hubs = {"DEL", "BOM", "BLR"}

    avail_records = []

    for _, pilot in pilots_df.iterrows():
        base = pilot["base_airport"]

        for _, flight in flights_df.iterrows():
            day = flight["day"]
            origin = flight["origin"]

            # Check 1: Is pilot available on this day?
            day_available = pilot[f"avail_{day}"]
            if day_available == 0:
                avail_records.append({
                    "pilot_id": pilot["pilot_id"],
                    "flight_id": flight["flight_id"],
                    "available": 0,
                })
                continue

            # Check 2: Can pilot reach the flight origin?
            if base == origin:
                # Same base — always eligible
                eligible = 1
            elif base in hubs:
                # Hub-based pilot can reposition to any origin (at cost)
                eligible = 1
            elif origin in hubs:
                # Spoke-based pilot can fly TO a hub (return positioning)
                eligible = 1
            else:
                # Spoke-to-spoke with different bases — not eligible
                eligible = 0

            avail_records.append({
                "pilot_id": pilot["pilot_id"],
                "flight_id": flight["flight_id"],
                "available": eligible,
            })

    avail_df = pd.DataFrame(avail_records)

    # Pivot to matrix form
    avail_matrix = avail_df.pivot(
        index="pilot_id", columns="flight_id", values="available"
    )

    return avail_matrix


# =============================================================================
# SECTION 8: INCOMPATIBLE FLIGHT PAIRS
# =============================================================================
# Two flights are incompatible for the same pilot if:
#   (a) They overlap in time (same day), OR
#   (b) Gap between them < minimum rest (R_min hours)
# This is pre-computed; constraint 8 uses: x_{p,f} + x_{p,f'} <= 1

def generate_incompatible_pairs(flights_df, R_min_hours: float = 12.0):
    """
    Identify pairs of flights that cannot be assigned to the same pilot.

    Incompatibility: same-day overlap OR insufficient rest gap.
    Uses R_min = 12 hours (strictest scenario) as default for pre-computation.
    The solver can adjust this per scenario.
    """
    incompatible = []

    # Group flights by day for efficient comparison
    for day in flights_df["day"].unique():
        day_flights = flights_df[flights_df["day"] == day]

        for (i, f1), (j, f2) in combinations(day_flights.iterrows(), 2):
            # Check overlap: f1 departs before f2 arrives AND f2 departs before f1 arrives
            f1_start = f1["dep_hour"]
            f1_end = f1["arr_hour"] if f1["arr_hour"] > f1["dep_hour"] else f1["arr_hour"] + 24
            f2_start = f2["dep_hour"]
            f2_end = f2["arr_hour"] if f2["arr_hour"] > f2["dep_hour"] else f2["arr_hour"] + 24

            # Check if flights overlap
            overlap = (f1_start < f2_end) and (f2_start < f1_end)

            if overlap:
                incompatible.append({
                    "flight_1": f1["flight_id"],
                    "flight_2": f2["flight_id"],
                    "reason": "overlap",
                    "gap_hours": 0,
                })
                continue

            # Check rest gap (if no overlap)
            # Gap = start of later flight - end of earlier flight
            if f1_end <= f2_start:
                gap = f2_start - f1_end
            else:
                gap = f1_start - f2_end

            # Include duty buffer: need R_min hours between arrival + 60min post-flight
            effective_gap = gap - (60 / 60.0)  # subtract 1 hour post-flight duty

            if effective_gap < R_min_hours:
                incompatible.append({
                    "flight_1": f1["flight_id"],
                    "flight_2": f2["flight_id"],
                    "reason": "insufficient_rest",
                    "gap_hours": round(effective_gap, 2),
                })

    # Also check consecutive-day pairs (e.g., late Sunday + early Monday)
    day_order = ["Mon", "Tue", "Wed", "Thu", "Fri", "Sat", "Sun"]
    for d_idx in range(len(day_order) - 1):
        today = day_order[d_idx]
        tomorrow = day_order[d_idx + 1]

        today_flights = flights_df[flights_df["day"] == today]
        tomorrow_flights = flights_df[flights_df["day"] == tomorrow]

        for _, f1 in today_flights.iterrows():
            f1_end = f1["arr_hour"]
            if f1_end < f1["dep_hour"]:
                f1_end += 24  # Overnight flight

            for _, f2 in tomorrow_flights.iterrows():
                # Gap = (24 - f1_end) + f2_dep - 1 (post-flight duty)
                gap = (24 - f1_end) + f2["dep_hour"] - 1.0

                if gap < R_min_hours:
                    incompatible.append({
                        "flight_1": f1["flight_id"],
                        "flight_2": f2["flight_id"],
                        "reason": "cross_day_insufficient_rest",
                        "gap_hours": round(gap, 2),
                    })

    return pd.DataFrame(incompatible)


# =============================================================================
# SECTION 9: PARAMETER TABLE (for report Section 3)
# =============================================================================

def generate_parameter_table():
    """
    Create the parameter table with sources — goes directly into the report.
    Every value must be justified (Dr Hao's explicit requirement).
    """
    params = [
        {"Parameter": "Number of airports", "Symbol": "|A|", "Value": "10", "Source": "IndiGo route network (goindigo.in)"},
        {"Parameter": "Number of flights/week", "Symbol": "|F|", "Value": "~100", "Source": "Scaled sub-network (Kasirzadeh et al., 2017)"},
        {"Parameter": "Number of pilots", "Symbol": "|P|", "Value": "20", "Source": "Ratio 7.6:1 (Business Standard, Mar 2026)"},
        {"Parameter": "Max weekly duty hours", "Symbol": "D_max", "Value": "60 hrs", "Source": "DGCA CAR S7/J/III Rev 2"},
        {"Parameter": "Weekly contiguous rest", "Symbol": "R_weekly", "Value": "36/48 hrs", "Source": "DGCA CAR S7/J/III Rev 2"},
        {"Parameter": "Min inter-flight rest", "Symbol": "R_min", "Value": "10-12 hrs", "Source": "DGCA CAR S7/J/III Rev 2"},
        {"Parameter": "Night landings/week cap", "Symbol": "N_night", "Value": "1/2/6", "Source": "DGCA CAR S7/J/III Rev 2"},
        {"Parameter": "Night window", "Symbol": "-", "Value": "00:00-05:00/06:00", "Source": "DGCA CAR S7/J/III Rev 2"},
        {"Parameter": "Consecutive night duties", "Symbol": "N_consec", "Value": "Max 2", "Source": "DGCA CAR S7/J/III Rev 2"},
        {"Parameter": "Uncovered flight penalty", "Symbol": "M_u", "Value": "50", "Source": "EUROCONTROL Std Inputs Ed.10 (2024)"},
        {"Parameter": "Deadhead cost premium", "Symbol": "-", "Value": "+3 units", "Source": "Barnhart et al. (2003)"},
        {"Parameter": "Night cost premium", "Symbol": "-", "Value": "+1 unit", "Source": "DGCA rest requirements"},
    ]
    return pd.DataFrame(params)


# =============================================================================
# SECTION 10: MAIN EXECUTION — Generate everything
# =============================================================================

if __name__ == "__main__":

    print("=" * 70)
    print("NBS8643 — Airline Crew Scheduling Data Generation")
    print("=" * 70)
    print(f"Random seed: {SEED}")
    print()

    # --- Generate Flights ---
    print("Generating flight schedule...")
    flights = generate_flight_schedule(seed=SEED)
    print(f"  Total flights generated: {len(flights)}")
    print(f"  Flights per day:")
    print(flights.groupby("day").size().reindex(
        ["Mon", "Tue", "Wed", "Thu", "Fri", "Sat", "Sun"]
    ).to_string(header=False))
    print(f"  Wave distribution:")
    print(flights["wave"].value_counts().to_string(header=False))
    print(f"  Night flights (old rule, arr 00:00-05:00): {flights['is_night_old'].sum()}")
    print(f"  Night flights (new rule, arr 00:00-06:00): {flights['is_night_new'].sum()}")
    print()

    # --- Generate Pilots ---
    print("Generating pilot roster...")
    pilots = generate_pilots(seed=SEED)
    print(f"  Total pilots: {len(pilots)}")
    print(f"  Base distribution:")
    print(pilots["base_airport"].value_counts().to_string(header=False))
    print()

    # --- Generate Cost Matrix ---
    print("Generating cost matrix (pilot × flight)...")
    cost_matrix = generate_cost_matrix(flights, pilots, seed=SEED)
    print(f"  Matrix shape: {cost_matrix.shape}")
    print(f"  Cost range: {cost_matrix.min().min()} to {cost_matrix.max().max()}")
    print(f"  Mean cost: {cost_matrix.mean().mean():.2f}")
    print()

    # --- Generate Availability Matrix ---
    print("Generating availability matrix...")
    avail_matrix = generate_availability_matrix(flights, pilots)
    total_cells = avail_matrix.shape[0] * avail_matrix.shape[1]
    available_cells = int(avail_matrix.sum().sum())
    print(f"  Matrix shape: {avail_matrix.shape}")
    print(f"  Available assignments: {available_cells}/{total_cells} ({100*available_cells/total_cells:.1f}%)")
    print()

    # --- Generate Incompatible Pairs ---
    print("Generating incompatible flight pairs...")
    incompat = generate_incompatible_pairs(flights, R_min_hours=12.0)
    print(f"  Total incompatible pairs: {len(incompat)}")
    if len(incompat) > 0:
        print(f"  By reason:")
        print(incompat["reason"].value_counts().to_string(header=False))
    print()

    # --- Parameter Table ---
    param_table = generate_parameter_table()

    # --- Scenario Table ---
    scenario_df = pd.DataFrame(SCENARIOS).T
    scenario_df.index.name = "scenario"

    # =================================================================
    # EXPORT TO EXCEL 
    # =================================================================
    out_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), "crew_scheduling_dataset.xlsx")
    print(f"Saving to: {out_path}")

    with pd.ExcelWriter(out_path, engine="openpyxl") as writer:
        flights.to_excel(writer, sheet_name="Flights", index=False)
        pilots.to_excel(writer, sheet_name="Pilots", index=False)
        AIRPORTS.to_excel(writer, sheet_name="Airports", index=False)
        ROUTES.to_excel(writer, sheet_name="Routes", index=False)
        cost_matrix.to_excel(writer, sheet_name="Cost_Matrix")
        avail_matrix.to_excel(writer, sheet_name="Availability_Matrix")
        if len(incompat) > 0:
            incompat.to_excel(writer, sheet_name="Incompatible_Pairs", index=False)
        param_table.to_excel(writer, sheet_name="Parameters", index=False)
        scenario_df.to_excel(writer, sheet_name="Scenarios")

    print(f"\nSaved successfully: {out_path}")
    print(f"\nSheets: Flights, Pilots, Airports, Routes, Cost_Matrix,")
    print(f"        Availability_Matrix, Incompatible_Pairs, Parameters, Scenarios")
    print()
    print("=" * 70)
    print("Data generation complete. Ready for MILP solver.")
    print("=" * 70)
