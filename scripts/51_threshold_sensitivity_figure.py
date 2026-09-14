#!/usr/bin/env python3
"""
AGU Threshold Sensitivity Figure
Generates publication-ready figure for AGU abstract upload.
"""

import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from pathlib import Path

PROJECT_DIR = Path(__file__).parent.parent
OUTPUT_DIR = PROJECT_DIR / "outputs"
FIG_DIR = PROJECT_DIR / "figures"

df = pd.read_csv(OUTPUT_DIR / "agu_threshold_sensitivity.csv")

fig, axes = plt.subplots(1, 2, figsize=(7.5, 3.5))

# Color palette
c_sce = '#1ABC9C'  # teal
c_hpe = '#E74C3C'  # coral
c_link = '#2E86AB' # blue
c_ann = '#7F8C8D'  # grey

# Panel A: Episode counts vs percentile
ax = axes[0]
x = df['percentile'].values
width = 2.5
ax.bar(x - width/2, df['sce_linked_episodes'], width, color=c_sce, edgecolor='black', linewidth=0.8, label='SCE-linked episodes')
ax.bar(x + width/2, df['hpe_days'], width, color=c_hpe, edgecolor='black', linewidth=0.8, label='HPE days')
ax.set_xlabel('Percentile Threshold (%)', fontsize=11)
ax.set_ylabel('Event Count', fontsize=11)
ax.set_title('(a) Episode Count vs. Threshold', fontsize=12, fontweight='bold')
ax.set_xticks(x)
ax.set_xticklabels([f'{int(p)}th' for p in x])
ax.legend(fontsize=9, loc='upper right')
ax.grid(axis='y', alpha=0.3)
ax.set_ylim(0, max(df['sce_linked_episodes'].max(), 1) + 2)

# Panel B: Linkage rate vs percentile
ax = axes[1]
ax.plot(x, df['linkage_rate_pct'], 'o-', color=c_link, linewidth=2.5, markersize=10, 
        markerfacecolor=c_link, markeredgecolor='black', markeredgewidth=1)
ax.axhline(y=75, color=c_ann, linestyle='--', linewidth=1.5, label='75% reference')
ax.set_xlabel('Percentile Threshold (%)', fontsize=11)
ax.set_ylabel('SCE Linkage Rate (%)', fontsize=11)
ax.set_title('(b) Linkage Rate Stability', fontsize=12, fontweight='bold')
ax.set_xticks(x)
ax.set_xticklabels([f'{int(p)}th' for p in x])
ax.set_ylim(0, 105)
ax.legend(fontsize=9, loc='lower left')
ax.grid(axis='y', alpha=0.3)

# Annotate values
for xi, yi in zip(x, df['linkage_rate_pct']):
    ax.annotate(f'{yi:.0f}%', (xi, yi), textcoords="offset points", xytext=(0, 10), 
                ha='center', fontsize=9, fontweight='bold')

plt.tight_layout()
fig.savefig(FIG_DIR / 'agu_threshold_sensitivity.png', dpi=300, bbox_inches='tight')
fig.savefig(FIG_DIR / 'agu_threshold_sensitivity.pdf', dpi=300, bbox_inches='tight')
print(f"Saved: {FIG_DIR / 'agu_threshold_sensitivity.png'}")
print(f"Saved: {FIG_DIR / 'agu_threshold_sensitivity.pdf'}")
