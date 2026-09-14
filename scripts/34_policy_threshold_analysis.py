#!/usr/bin/env python3
"""
Script 34: SCE Policy Threshold Analysis (Corrected)
Ahmedabad HPE Project

Identifies winter PM2.5 extreme episodes, checks whether each is followed
by a summer extreme-heat day within the SAME annual cycle (since only
2019 data is available), and reports policy tier counts using each tier's
own stated trigger rule instead of raw unfiltered episode counts.
"""

import numpy as np
import pandas as pd
import xarray as xr
from pathlib import Path

PROJECT_DIR = Path(__file__).parent.parent
OUTPUT_DIR = PROJECT_DIR / "outputs"


def get_time_values(da):
    for c in ['time', 'valid_time', 'date', 'day']:
        if c in da.coords:
            return da[c]
    return pd.date_range('2019-01-01', periods=len(da), freq='D')


def merge_into_episodes(dates):
    """Merge a DatetimeIndex of extreme days into consecutive-day episodes."""
    episodes = []
    if len(dates) == 0:
        return episodes
    dates = dates.sort_values()
    start = prev = dates[0]
    for d in dates[1:]:
        if d == prev + pd.Timedelta(days=1):
            prev = d
        else:
            episodes.append({'start': start, 'end': prev, 'length': (prev - start).days + 1})
            start = prev = d
    episodes.append({'start': start, 'end': prev, 'length': (prev - start).days + 1})
    return episodes


def main():
    print("=" * 70)
    print("SCE POLICY THRESHOLD ANALYSIS (CORRECTED)")
    print("=" * 70)

    # ---- Load data ----
    cams = xr.open_dataset(OUTPUT_DIR / "pm25_daily_cams_ahmedabad_2019.nc")
    era5 = xr.open_dataset(OUTPUT_DIR / "era5_daily_2019.nc")

    pm25_da = cams["pm25"].mean(dim=["latitude", "longitude"]) if "latitude" in cams["pm25"].dims else cams["pm25"]
    t2m_da = era5["tmax"] if "tmax" in era5 else era5["t2m"]

    pm25 = pd.Series(pm25_da.values, index=pd.to_datetime(get_time_values(pm25_da).values))
    t2m = pd.Series(t2m_da.mean(dim=["latitude", "longitude"]).values,
                     index=pd.to_datetime(get_time_values(t2m_da).values))

    common = pm25.index.intersection(t2m.index)
    pm25 = pm25.loc[common].sort_index()
    t2m = t2m.loc[common].sort_index()

    if t2m.mean() > 100:
        print("WARNING: t2m mean > 100, appears to be Kelvin. Converting to Celsius.")
        t2m = t2m - 273.15

    # ---- Thresholds (absolute, annual, matched 90/90 and 95/95) ----
    pm90 = pm25.quantile(0.90)
    pm95 = pm25.quantile(0.95)
    t90 = t2m.quantile(0.90)

    print(f"\nPM2.5 90th percentile: {pm90:.1f} ug/m3")
    print(f"PM2.5 95th percentile: {pm95:.1f} ug/m3")
    print(f"Tmax 90th percentile: {t90:.1f} C")

    # ---- Winter PM extreme episodes (Nov-Feb) ----
    winter_months = [11, 12, 1, 2]
    summer_months = [3, 4, 5, 6]

    pm_extreme_days = pm25[pm25 > pm90].index
    pm_severe_days = pm25[pm25 > pm95].index
    heat_extreme_days = t2m[t2m > t90].index

    winter_pm_extreme_days = pm_extreme_days[pm_extreme_days.month.isin(winter_months)]
    winter_pm_severe_days = pm_severe_days[pm_severe_days.month.isin(winter_months)]
    summer_heat_days = heat_extreme_days[heat_extreme_days.month.isin(summer_months)]

    winter_episodes = merge_into_episodes(winter_pm_extreme_days)
    severe_episodes = merge_into_episodes(winter_pm_severe_days)

    print(f"\n--- WINTER PM EXTREME EPISODES (>P90 = {pm90:.1f}) ---")
    print(f"Total winter episodes: {len(winter_episodes)}")

    # ---- Data-boundary check: Nov-Dec episodes' summer falls in the NEXT year ----
    for ep in winter_episodes:
        ep['evaluable'] = ep['start'].month not in [11, 12]

    evaluable_episodes = [ep for ep in winter_episodes if ep['evaluable']]
    non_evaluable_episodes = [ep for ep in winter_episodes if not ep['evaluable']]

    print(f"Evaluable episodes (Jan-Feb, summer season present in data): {len(evaluable_episodes)}")
    print(f"Non-evaluable episodes (Nov-Dec, summer falls in following year, "
          f"outside data window): {len(non_evaluable_episodes)}")

    # ---- Check which evaluable episodes are SCE-linked ----
    for ep in evaluable_episodes:
        future_heat = summer_heat_days[summer_heat_days > ep['end']]
        ep['sce_linked'] = len(future_heat) > 0
        ep['lead_time'] = (future_heat[0] - ep['end']).days if len(future_heat) > 0 else None

    sce_linked = [ep for ep in evaluable_episodes if ep['sce_linked']]

    print(f"\n--- SCE-LINKED EPISODES (episode-based) ---")
    print(f"SCE-linked winter episodes: {len(sce_linked)} out of {len(evaluable_episodes)} evaluable "
          f"({len(winter_episodes)} total, {len(non_evaluable_episodes)} not evaluable due to data boundary)")

    if sce_linked:
        lead_times = [ep['lead_time'] for ep in sce_linked]
        print(f"Lead time (episode end to first heat day): {min(lead_times)}-{max(lead_times)} days")
        print(f"Mean lead time: {np.mean(lead_times):.0f} days")
        print(f"\n{'Start':<12} {'End':<12} {'Len':>4} {'Lead':>6}")
        print("-" * 40)
        for ep in sce_linked:
            print(f"{str(ep['start'].date()):<12} {str(ep['end'].date()):<12} "
                  f"{ep['length']:>4} {ep['lead_time']:>6}")

    print(f"\n--- WINTER SEVERE EPISODES (>P95 = {pm95:.1f}) ---")
    print(f"Total severe winter episodes: {len(severe_episodes)}")

    # ================================================================
    # POLICY TIERS — each count now filtered by that tier's own rule
    # ================================================================
    print("\n" + "=" * 70)
    print("PROPOSED SCE-ENHANCED HEAT ACTION PLAN THRESHOLDS")
    print("=" * 70)
    print("INTEGRATION: Ahmedabad Municipal Corporation (AMC) Heat Action Plan (HAP)")
    print("Ahmedabad pioneered India's first Heat Action Plan in 2013. The SCE")
    print("framework enhances HAP by adding a WINTER PM WATCH phase that triggers")
    print("preparedness ahead of the summer heat season.\n")

    # Tier 1: PM>P90 for >=2 consecutive days
    tier1 = [ep for ep in sce_linked if ep['length'] >= 2]

    # Severe (P95) episodes, evaluable + SCE-linked
    for ep in severe_episodes:
        ep['evaluable'] = ep['start'].month not in [11, 12]
        if ep['evaluable']:
            future_heat = summer_heat_days[summer_heat_days > ep['end']]
            ep['sce_linked'] = len(future_heat) > 0
        else:
            ep['sce_linked'] = False
    severe_evaluable = [ep for ep in severe_episodes if ep['evaluable'] and ep['sce_linked']]

    # Tier 2: PM>P90 for >=3 days OR PM>P95 for >=2 days (union, no double count)
    tier2_from_p90 = [ep for ep in sce_linked if ep['length'] >= 3]
    tier2_from_p95 = [ep for ep in severe_evaluable if ep['length'] >= 2]
    tier2_starts = set(ep['start'] for ep in tier2_from_p90) | set(ep['start'] for ep in tier2_from_p95)

    # Tier 3: PM>P95 for >=3 days (IMD forecast condition not retrospectively testable)
    tier3 = [ep for ep in severe_evaluable if ep['length'] >= 3]

    print("TIER 1 (YELLOW / Winter PM Watch):")
    print(f"  Trigger: PM2.5 > {pm90:.1f} ug/m3 for 2+ consecutive days, Nov-Feb")
    print("  Action:  (a) Public health advisory via AMC/108")
    print("           (b) Hospitals flag heat-vulnerable patients")
    print("           (c) Activate heat-health surveillance ahead of predicted heat")
    print(f"  2019 occurrences meeting trigger: {len(tier1)} episodes "
          f"(of {len(sce_linked)} total SCE-linked episodes)")

    print("\nTIER 2 (ORANGE / PM Warning):")
    print(f"  Trigger: PM2.5 > {pm90:.1f} for 3+ days OR > {pm95:.1f} for 2+ days, Nov-Feb")
    print("  Action:  (a) Pre-position ORS, ice packs, IV fluids")
    print("           (b) Activate cooling-center inventory")
    print("           (c) Targeted SMS alerts to vulnerable groups")
    print(f"  2019 occurrences meeting trigger: {len(tier2_starts)} episodes")

    print("\nTIER 3 (RED / Pre-Heat Emergency):")
    print(f"  Trigger: PM2.5 > {pm95:.1f} for 3+ days, Nov-Feb, AND IMD forecasts")
    print("           above-normal summer temperatures")
    print("           [forecast condition not testable retrospectively;")
    print("            PM-episode criterion shown alone below]")
    print("  Action:  (a) Full HAP activation ahead of forecast Tmax >42C")
    print("           (b) Work-hour restrictions, construction/transport")
    print("           (c) School early-dismissal protocols")
    print("           (d) Emergency medical staffing")
    print(f"  2019 occurrences meeting PM-episode criterion: {len(tier3)} episodes")

    if sce_linked:
        lead_times = [ep['lead_time'] for ep in sce_linked]
        print("\nLEAD TIME:")
        print(f"  Mean: {np.mean(lead_times):.0f} days (range: {min(lead_times)}-{max(lead_times)})")
        print("  NOTE: this is fixed calendar lead time ahead of the climatologically")
        print("  certain hot season, not a predictive forecast of heat severity from")
        print("  PM levels — framed as a preparedness trigger, not an early-warning forecast.")

    print("\n" + "=" * 70)
    print("DATA LIMITATION")
    print("=" * 70)
    print(f"{len(non_evaluable_episodes)} of {len(winter_episodes)} winter episodes (Nov-Dec) could")
    print("not be evaluated: their following summer season falls in the next calendar")
    print("year, outside this single-year (2019) dataset. Multi-year data is needed")
    print("to evaluate these fully.")

    # ---- Save ----
    rows = []
    for ep in evaluable_episodes:
        rows.append({'start': ep['start'], 'end': ep['end'], 'length': ep['length'],
                      'evaluable': True, 'sce_linked': ep['sce_linked'],
                      'lead_time_days': ep['lead_time']})
    for ep in non_evaluable_episodes:
        rows.append({'start': ep['start'], 'end': ep['end'], 'length': ep['length'],
                      'evaluable': False, 'sce_linked': None, 'lead_time_days': None})
    pd.DataFrame(rows).to_csv(OUTPUT_DIR / "sce_policy_episodes.csv", index=False)
    print(f"\nSaved: {OUTPUT_DIR / 'sce_policy_episodes.csv'}")

    cams.close()
    era5.close()


if __name__ == "__main__":
    main()
    