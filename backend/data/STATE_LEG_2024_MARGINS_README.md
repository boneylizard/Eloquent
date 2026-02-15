# 2024 presidential margins

## State-level (governor races)

State-level 2024 Trump margins for **governor** races (VA, NJ, etc.) are computed from [tonmcg/US_County_Level_Election_Results_08-24](https://github.com/tonmcg/US_County_Level_Election_Results_08-24) (2024 county-level presidential CSV). Aggregating by state gives two-party Trump margin per state. To refresh:

- **API:** `POST /election/simulation/calibration/refresh-2024-state-margins` — refreshes 2024 baseline: fetches the CSV, aggregates by state, writes `pres_2024_state_margins.json` (used for governor calibration 2024 vs now).
- **CLI:** `python -m app.pres_2024_county_loader` from repo root (with `backend` on PYTHONPATH or run from `backend`).

If `pres_2024_state_margins.json` exists, governor races use it; otherwise the scraper falls back to hardcoded `PRES_2024_STATE_MARGIN`.

---

## District-level (state legislative: VA House, NJ Assembly)

Optional file: `state_leg_2024_presidential_margins.json`

**We do not use state-level 2024 margins for state legislative districts.** Comparing a district result to the state margin (e.g. VA House D47 to “Virginia went D+5.78”) would be wrong: that district might have been D+25 or R+15 in 2024. Using state margins would bias the simulation. So `trump_2024_margin` and `swing_toward_d` for VA House and NJ Assembly are **only** set when we have **district-level** 2024 presidential data in this file (or in `TRUMP_2024_STATE_LEG_OVERRIDES` for specific races). Otherwise those fields are omitted.

## Format

```json
{
  "VA_House": { "1": 12.5, "2": -3.2, "22": 2.1 },
  "NJ_Assembly": { "1": -15.0, "3": -8.2 }
}
```

- Keys are district numbers as **strings** (VA 1–100, NJ 1–40).
- Values are **Trump margin** in points: (R share − D share) in 2024 presidential, two-party. Negative = Trump lost that district.

## Data sources (district-level 2024 pres)

Use these to build the JSON so swing is comparable to the same electoral area.

### Virginia House of Delegates (100 districts)

- **VPAP (Virginia Public Access Project)**  
  [2024 Presidential Results by State Legislative Districts](https://www.vpap.org/visuals/visual/2024-presidential-results-by-state-legislative-districts/)  
  Provides 2024 presidential results by VA House district. Download or use the table; compute two-party Trump margin per district and fill `VA_House` in the JSON.

- **Virginia Department of Elections**  
  Official results by precinct; you need a precinct-to–House-district mapping (e.g. shapefiles or district assignment files) to aggregate to district. VPAP usually does this for you.

### New Jersey Assembly (40 districts)

- **New Jersey Division of Elections**  
  Official results; may need to aggregate by legislative district using district boundaries/precinct maps.

- **New Jersey Globe / politico-style outlets**  
  Often publish “2024 presidential results by legislative district” or “partisan index” by district. Convert to Trump margin (R% − D% two-party) per district and fill `NJ_Assembly`.

- **Daily Kos Elections**  
  Sometimes publishes 2020/2024 presidential results by state legislative district; check for NJ.

### State legislative specials (e.g. TX Senate 9, GA State Senate 35)

For races that are a single district, add an override in `ballotpedia_scraper.py`:

```python
TRUMP_2024_STATE_LEG_OVERRIDES: Dict[str, float] = {
    "TX_Senate_9": 17.0,   # Trump margin in that district
    "GA_Senate_35": 12.0,  # if you have the number
}
```

Key format: `{STATE}_{Senate|House}_{district_number_or_identifier}`. Value: Trump margin in points (positive = R won that district).

## Calibration top-four 2024 margins (confirmed)

| Race | 2024 R margin (Trump margin) | Source |
|------|-----------------------------|--------|
| **TN-07** (special) | R+22 | Ballotpedia 119th specials table "2024 Presidential MOV" column (district-level). |
| **VA Governor 2025** | D+5.9 (stored as -5.9) | `pres_2024_state_margins.json` from tonmcg county CSV. Harris won VA. |
| **NJ Governor 2025** | D+5.9 (stored as -5.9) | Same state file. Harris won NJ. |
| **TX State Senate 9** | R+17 | `TRUMP_2024_STATE_LEG_OVERRIDES`; verify with district-level source if available. |

Governor margins are refilled from the state file when calibration is loaded, so stale stored values are overwritten.

## Summary

- **Statewide races (e.g. VA/NJ Governor 2025):** We use state 2024 margin; same electorate, so it’s correct.
- **State legislative districts (VA House, NJ Assembly, state leg specials):** We only set `trump_2024_margin` and `swing_toward_d` when we have **district-level** (or race-specific) 2024 pres data. No state-margin fallback — it’s not relevant and would bias the simulation.
