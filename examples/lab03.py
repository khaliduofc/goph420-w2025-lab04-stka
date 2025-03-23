import numpy as np
import matplotlib.pyplot as plt
import os

# Physical parameters
rho1 = 1800
rho2 = 2500
beta1 = 1900
beta2 = 3200
H = 4000

# Output path for plots
figures_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "figures"))
os.makedirs(figures_dir, exist_ok=True)

# Derived constants
zeta_max = H * np.sqrt(1 / beta1**2 - 1 / beta2**2)
eps = 1e-6
max_iter = 100

def dispersion_function(zeta, f):
    if zeta <= 0 or zeta >= zeta_max:
        return np.nan
    lhs = (rho2 / rho1) * np.sqrt(H**2 * ((1 / beta1**2) - (1 / beta2**2)) - zeta**2) / zeta
    rhs = np.tan(2 * np.pi * f * zeta)
    return lhs - rhs

def root_secant_modified(x0, dx, f):
    errors = []
    for i in range(max_iter):
        f_x = f(x0)
        f_xdx = f(x0 + dx)
        if np.isnan(f_x) or np.isnan(f_xdx) or f_xdx - f_x == 0:
            break
        x1 = x0 - f_x * dx / (f_xdx - f_x)
        err = abs((x1 - x0) / x1) if x1 != 0 else np.inf
        errors.append(err)
        if err < eps:
            return x1, i + 1, np.array(errors)
        x0 = x1
    return x0, max_iter, np.array(errors)

def compute_cL_from_zeta(zeta):
    val = (1 / beta1**2) - (zeta**2 / H**2)
    return np.sqrt(1 / val)

def compute_lambdaL(cL, f):
    return cL / f

frequencies = np.round(np.arange(0.1, 5.01, 0.1), 3)
selected_freqs = [0.1, 0.5, 1.0, 2.0]

# === Plot F(zeta) with asymptotes ===
plt.figure(figsize=(10, 10))
for j, f in enumerate(selected_freqs):
    plt.subplot(len(selected_freqs), 1, j + 1)
    def F(zeta): return dispersion_function(zeta, f)

    asymptotes = [0.01]
    asymptotes += [0.25 * (2 * k + 1) / f for k in range(40) if 0.25 * (2 * k + 1) / f < zeta_max]
    asymptotes.append(zeta_max)

    for a in asymptotes:
        plt.axvline(x=a, color='r', linestyle=':')

    for k in range(len(asymptotes) - 1):
        zeta_vals = np.linspace(asymptotes[k] + 1e-3, asymptotes[k + 1] - 1e-3, 1000)
        F_vals = [F(z) for z in zeta_vals]
        plt.plot(zeta_vals, F_vals, "-b")

    plt.grid()
    plt.xlabel("zeta")
    plt.ylabel("F(zeta)")
    plt.xlim(0.0, zeta_max)
    plt.ylim(-5.0, 5.0)
    plt.title(f"F(zeta) for Frequency {f} Hz")

plt.tight_layout()
plt.savefig(os.path.join(figures_dir, "F_zeta_vs_freq.png"))
plt.show()

# === Solve modes using bracketing ===
def solve_modes_by_bracketing(f, asymptotes):
    mode_roots, mode_cL, mode_lambdaL, mode_errors, mode_iters = [], [], [], [], []
    for k in range(len(asymptotes) - 1):
        a_left = asymptotes[k]
        a_right = asymptotes[k + 1]
        guess = a_right - 0.01
        func = lambda z: dispersion_function(z, f)
        root, iters, errs = root_secant_modified(guess, 1e-4, func)
        if not np.isnan(root) and a_left < root < a_right:
            mode_roots.append(root)
            cL = compute_cL_from_zeta(root)
            lambdaL = compute_lambdaL(cL, f)
            mode_cL.append(cL)
            mode_lambdaL.append(lambdaL)
            mode_errors.append(errs)
            mode_iters.append(iters)
    return mode_roots, mode_cL, mode_lambdaL, mode_iters, mode_errors

zeta_curves, cL_curves, lambdaL_curves = {}, {}, {}
frequency_record = []

print("\n=== Detailed Output for f = 0.1, 0.5, 1.0, 2.0, 5.0 Hz ===")
for f in frequencies:
    asymptotes = [0.25 * (2 * k + 1) / f for k in range(40) if 0.25 * (2 * k + 1) / f < zeta_max]
    asymptotes = [0.01] + asymptotes + [zeta_max]

    roots, cLs, lambdas, iters_list, errors_list = solve_modes_by_bracketing(f, asymptotes)
    zeta_curves[f] = roots
    cL_curves[f] = cLs
    lambdaL_curves[f] = lambdas
    frequency_record.append(f)

    if f in [0.1, 0.5, 1.0, 2.0, 5.0]:
        print(f"\n--- Frequency: {f:.3f} Hz ---")
        for i in range(len(roots)):
            print(f"Mode {i}: zeta = {roots[i]:.6f}, cL = {cLs[i]:.2f}, lambdaL = {lambdas[i]:.2f}, Iterations: {iters_list[i]}")
            print("Errors:", errors_list[i])

# === Plotting Arrays ===
max_modes = 6
zeta_array = np.full((max_modes, len(frequencies)), np.nan)
cL_array = np.full((max_modes, len(frequencies)), np.nan)
lambda_array = np.full((max_modes, len(frequencies)), np.nan)

for j, f in enumerate(frequencies):
    for m in range(min(len(zeta_curves[f]), max_modes)):
        zeta_array[m, j] = zeta_curves[f][m]
        cL_array[m, j] = cL_curves[f][m]
        lambda_array[m, j] = lambdaL_curves[f][m]

xticks = np.round(np.arange(0.1, 5.1, 0.2), 1)

# === Plot zeta vs Frequency ===
plt.figure(figsize=(10, 5))
for m in range(max_modes):
    plt.plot(frequencies, zeta_array[m], marker='o', label=f"Mode {m}")
plt.title("zeta vs Frequency")
plt.xlabel("Frequency (Hz)")
plt.ylabel("zeta")
plt.xticks(xticks)
plt.yticks(np.arange(0, 1.9, 0.1))
plt.grid(True)
plt.legend()
plt.tight_layout()
plt.savefig(os.path.join(figures_dir, "zeta_vs_freq.png"))
plt.show()

# === Plot cL vs Frequency ===
plt.figure(figsize=(10, 5))
for m in range(max_modes):
    plt.plot(frequencies, cL_array[m], marker='o', label=f"Mode {m}")
plt.title("Love Wave Velocity cL vs Frequency")
plt.xlabel("Frequency (Hz)")
plt.ylabel("cL (m/s)")
plt.xticks(xticks)
plt.grid(True)
plt.legend()
plt.tight_layout()
plt.savefig(os.path.join(figures_dir, "cL_vs_freq.png"))
plt.show()

# === Plot lambdaL vs Frequency ===
plt.figure(figsize=(10, 5))
for m in range(max_modes):
    plt.plot(frequencies, lambda_array[m], marker='o', label=f"Mode {m}")
plt.title("Love Wave Wavelength lambdaL vs Frequency")
plt.xlabel("Frequency (Hz)")
plt.ylabel("lambdaL (m)")
plt.xticks(xticks)
plt.grid(True)
plt.legend()
plt.tight_layout()
plt.savefig(os.path.join(figures_dir, "lambdaL_vs_freq.png"))
plt.show()
