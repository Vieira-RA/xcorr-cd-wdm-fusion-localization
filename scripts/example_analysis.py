# scripts/example_analysis.py
import sys
sys.path.append('/home/240404662/PhD/sop-simulation-lib/src')  # temporary, we'll install properly later
from polarization_tools import normalize_stokes
import numpy as np

S = np.array([1.0, 0.6, 0.3, 0.2])
S_norm = normalize_stokes(S)
print("Original:", S)
print("Normalized:", S_norm)