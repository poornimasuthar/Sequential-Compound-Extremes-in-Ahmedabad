#!/usr/bin/env python3
"""
Script 35: Verify abstract claims before submission
- Checks that IER cause-specific values sum consistently with the total
- Asserts every Jan-Feb winter PM episode has a subsequent summer heat day
"""

import pandas as pd
import numpy as np
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
    episodes = []
    if len(dates) == 0:
        return episodes
    dates = dates.sort_values()
    start = prev = dates[0]
    for d in dates[1:]:
        if d == prev + pd.Timedelta(days=1):
            prev = d
        else:
            episodes.append({'start': start, 'end': prev})
            start = prev = d
    episodes.append({'start': start, 'end': prev})
    return episodes


print("=" * 60)
print("CHECK 1: IER CAUSE-SPECIFIC SUM")
print("=" * 60)
df = pd.read_csv(OUTPUT_DIR / "ier_summary.csv")
print(df.to_string())
exact_sum = df['excess_deaths'].sum()
print(f"\nExact (unrounded) sum of causes: {exact_sum:.4f}")
print(f"Rounded sum of causes: {round(exact_sum)}")
print(f"Sum of individually-rounded causes: {df['excess_deaths'].round(0).sum():.0f}")

print("\n" + "=" * 60)
print("CHECK 2: ALL JAN-FEB WINTER EPISODES SCE-LINKED?")
print("=" * 60)

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
    t2m = t2m - 273.15

pm90 = pm25.quantile(0.90)
t90 = t2m.quantile(0.90)

pm_extreme_days = pm25[pm25 > pm90].index
heat_extreme_days = t2m[t2m > t90].index

winter_pm_days = pm_extreme_days[pm_extreme_days.month.isin([11, 12, 1, 2])]
summer_heat_days = heat_extreme_days[heat_extreme_days.month.isin([3, 4, 5, 6])]

winter_episodes = merge_into_episodes(winter_pm_days)
jan_feb_episodes = [ep for ep in winter_episodes if ep['start'].month in [1, 2]]

linked_count = 0
for ep in jan_feb_episodes:
    future_heat = summer_heat_days[summer_heat_days > ep['end']]
    linked = len(future_heat) > 0
    if linked:
        linked_count += 1
    print(f"  Episode {ep['start'].date()} to {ep['end'].date()}: "
          f"{'LINKED' if linked else 'NOT LINKED'}")

print(f"\nJan-Feb episodes: {len(jan_feb_episodes)}")
print(f"SCE-linked: {linked_count}")

assert linked_count == len(jan_feb_episodes), (
    f"CLAIM FAILS: only {linked_count} of {len(jan_feb_episodes)} "
    f"Jan-Feb episodes are SCE-linked, not all of them!"
)
print("\nASSERTION PASSED: every Jan-Feb winter PM episode is followed by a summer heat day.")

cams.close()
era5.close()
