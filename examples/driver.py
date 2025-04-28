import os
import numpy as np
import matplotlib.pyplot as plt
from lab_04.regression import multi_regress

# Set up figures directory
figures_dir = os.path.join(os.path.dirname(__file__), "..", "figures")
os.makedirs(figures_dir, exist_ok=True)

# Load earthquake data
data_path = os.path.join(os.path.dirname(__file__), "M_data1.txt")
data = np.loadtxt(data_path)
times = data[:, 0]
magnitudes = data[:, 1]

# Plot raw magnitudes vs time
plt.figure(figsize=(8, 5))
plt.scatter(times, magnitudes, s=3, color='blue')
plt.xlabel("Time (hours)")
plt.ylabel("Magnitude")
plt.title("Earthquake Magnitudes Over Time")
for boundary in [34, 45, 71, 96, 120]:
    plt.axvline(x=boundary, color='red', linestyle='--', linewidth=1)
plt.xlim(0, 120)
plt.grid(True)
plt.tight_layout()
plt.savefig(os.path.join(figures_dir, "raw_data_intervals.png"))
plt.show()
plt.close()

# Define original intervals
intervals = [
    (0, 34),
    (34, 45),
    (45, 71),
    (71, 96),
    (96, 120)
]

# Analyze each interval
for idx, (start, end) in enumerate(intervals, start=1):
    if idx < len(intervals):
        mask = (times >= start) & (times < end)
    else:
        mask = (times >= start) & (times <= end)

    mags_interval = magnitudes[mask]
    if mags_interval.size == 0:
        continue

    # Prepare cumulative counts
    unique_mags, counts = np.unique(mags_interval, return_counts=True)
    cum_counts = np.cumsum(counts[::-1])[::-1]
    logN = np.log10(cum_counts)

    Z = np.column_stack((np.ones_like(unique_mags), unique_mags))  # Design matrix

    # Perform regression
    a, residuals, r_squared = multi_regress(logN, Z)

    intercept = a[0]
    slope = a[1]
    a_value = intercept
    b_value = -slope

    # Print results
    print(f"Interval {idx}: {start}-{end} hours")
    print(f"  a = {a_value:.4f}")
    print(f"  b = {b_value:.4f}")
    print(f"  r² = {r_squared:.4f}")
    print("-" * 30)

    # Plot fits
    plt.figure(figsize=(8, 5))
    plt.scatter(unique_mags, logN, color='blue', label='Data points')
    line_x = np.array([unique_mags.min(), unique_mags.max()])
    line_y = intercept + slope * line_x
    plt.plot(line_x, line_y, 'r-', label='Best-fit line')

    plt.xlabel("Magnitude (M)")
    plt.ylabel("log₁₀(N ≥ M)")
    plt.title(f"Interval {idx}: {start}-{end} hours")
    text_info = f"a = {a_value:.3f}\nb = {b_value:.3f}\nr² = {r_squared:.3f}"
    plt.text(0.95, 0.95, text_info, transform=plt.gca().transAxes,
             verticalalignment='top', horizontalalignment='right',
             bbox=dict(facecolor='white', alpha=0.8))
    plt.legend()
    plt.grid(True)
    plt.tight_layout()

    # Save the figure
    plt.savefig(os.path.join(figures_dir, f"interval_{idx}_fit.png"))
    plt.show()
    plt.close()
