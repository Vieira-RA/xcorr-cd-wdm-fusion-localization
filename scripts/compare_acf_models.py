"""
scripts/compare_acf_models.py

Compares the autocorrelation function and its area for:
- OU process (exponential ACF)
- ZOH process (triangular ACF)

Plots empirical ACFs together with theoretical curves.
"""

import numpy as np
import matplotlib.pyplot as plt
from correlation_models import (
    generate_ou_process,
    generate_zoh_process,
    compute_acf,
    area_under_acf
)

def theoretical_exp_acf(lags, Tc):
    """Theoretical exponential ACF: exp(-|τ|/Tc)."""
    return np.exp(-np.abs(lags) / Tc)

def theoretical_tri_acf(lags, Tc):
    """Theoretical triangular ACF: max(1 - |τ|/Tc, 0)."""
    return np.maximum(1.0 - np.abs(lags) / Tc, 0.0)

def main():
    # Parameters
    Tc = 1.0          # correlation time [s]
    duration = 400.0  # total simulation time (must be >> Tc)
    dt_sim = 0.001     # simulation time step (fine enough to resolve ACF)
    max_lag = 3.0     # maximum lag for ACF plot [s]

    # 1. Generate OU process (exponential)
    t_ou, X_ou = generate_ou_process(duration, Tc, dt_sim, seed=42)
    lags_ou, acf_ou = compute_acf(t_ou, X_ou, max_lag=max_lag)
    area_ou = area_under_acf(lags_ou, acf_ou)

    # 2. Generate ZOH process (triangular)
    t_zoh, X_zoh = generate_zoh_process(duration, Tc, dt_sim, seed=42)
    lags_zoh, acf_zoh = compute_acf(t_zoh, X_zoh, max_lag=max_lag)
    area_zoh = area_under_acf(lags_zoh, acf_zoh)

    # Theoretical expectations
    theo_area_exp = 2.0 * Tc
    theo_area_tri = Tc

    # Plotting
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))

    # Left: Exponential OU
    ax1.plot(lags_ou, acf_ou, 'b-', label='Simulated OU', linewidth=2)
    ax1.plot(lags_ou, theoretical_exp_acf(lags_ou, Tc), 'r--', label='Theoretical exp(-|τ|/Tc)', linewidth=2)
    ax1.set_xlabel('Lag τ [s]')
    ax1.set_ylabel('Autocorrelation R(τ)')
    ax1.set_title('OU Process (Exponential ACF)')
    ax1.legend()
    ax1.grid(True, alpha=0.3)

    # Right: ZOH
    ax2.plot(lags_zoh, acf_zoh, 'b-', label='Simulated ZOH', linewidth=2)
    ax2.plot(lags_zoh, theoretical_tri_acf(lags_zoh, Tc), 'r--', label='Theoretical triangular', linewidth=2)
    ax2.set_xlabel('Lag τ [s]')
    ax2.set_ylabel('Autocorrelation R(τ)')
    ax2.set_title('ZOH Process (Triangular ACF)')
    ax2.legend()
    ax2.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig('../output/acf_comparison.png', dpi=150)
    plt.show()

    # Print area results
    print("=" * 50)
    print("Area under autocorrelation (two‑sided integral):")
    print(f"OU process (exp)   : simulated = {area_ou:.4f} s, theoretical = {theo_area_exp:.4f} s")
    print(f"ZOH process (tri)  : simulated = {area_zoh:.4f} s, theoretical = {theo_area_tri:.4f} s")
    print(f"Ratio (exp / tri)  : simulated = {area_ou/area_zoh:.2f}, theoretical = 2.00")
    print("=" * 50)

if __name__ == "__main__":
    main()