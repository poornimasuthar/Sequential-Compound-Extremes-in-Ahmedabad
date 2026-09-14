import pandas as pd
import matplotlib.pyplot as plt
import json

# Load SCE timeline
df = pd.read_csv("outputs/sce_results/sce_timeline_2019.csv")
df['date'] = pd.to_datetime(df['date'])

fig, axes = plt.subplots(3, 1, figsize=(14, 10), sharex=True)

# PM2.5
axes[0].plot(df['date'], df['pm25'], 'b-', alpha=0.7, lw=0.8)
axes[0].axhline(y=49.5, color='r', ls='--', label='75th percentile (49.5)')
axes[0].fill_between(df['date'], 49.5, df['pm25'].max(), 
                     where=df['high_pm'], alpha=0.3, color='blue')
axes[0].set_ylabel('PM2.5 (µg/m³)')
axes[0].set_title('(a) Daily PM2.5 - High Pollution Days', fontweight='bold', loc='left')
axes[0].legend()

# Temperature
axes[1].plot(df['date'], df['temperature'], 'orange', alpha=0.7, lw=0.8)
axes[1].axhline(y=41.9, color='r', ls='--', label='95th percentile (41.9°C)')
axes[1].fill_between(df['date'], 41.9, df['temperature'].max(),
                     where=df['extreme_temp'], alpha=0.3, color='orange')
axes[1].set_ylabel('Tmax (°C)')
axes[1].set_title('(b) Daily Maximum Temperature - Extreme Heat Days', fontweight='bold', loc='left')
axes[1].legend()

# Combined
axes[2].fill_between(df['date'], 0, 1, where=df['high_pm'], 
                     alpha=0.3, color='blue', label='High PM2.5')
axes[2].fill_between(df['date'], 0, 1, where=df['extreme_temp'],
                     alpha=0.3, color='orange', label='Extreme heat')
axes[2].set_ylabel('Event')
axes[2].set_xlabel('Month')
axes[2].set_title('(c) Inverse Seasonality: No Overlap = 3 HPE Days, But 15 SCE-30 Events', 
                  fontweight='bold', loc='left')
axes[2].legend(loc='upper right')
axes[2].set_ylim(0, 1)

plt.suptitle('Sequential Compound Extreme (SCE) Framework: Ahmedabad 2019', 
             fontsize=14, fontweight='bold', y=1.02)
plt.tight_layout()
plt.savefig('figures/fig8_sce_timeline.png', dpi=300, bbox_inches='tight')
plt.close()

print("Saved: figures/fig8_sce_timeline.png")
