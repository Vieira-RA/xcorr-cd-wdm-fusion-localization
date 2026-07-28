import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import math

# ============================================================
# 1. Define the four vertices on the Poincaré sphere
#    (unit vectors in S1, S2, S3 space)
# ============================================================
P1 = np.array([0, 0, 1])                      # North pole
P2 = np.array([2*np.sqrt(2)/3, 0, -1/3])     # S1 axis
P3 = np.array([-np.sqrt(2)/3, np.sqrt(6)/3, -1/3])
P4 = np.array([-np.sqrt(2)/3, -np.sqrt(6)/3, -1/3])

points = [P1, P2, P3, P4]
labels = ['P1 (North Pole)', 'P2 (S1 axis)', 'P3', 'P4']

# ============================================================
# 2. Print the full Stokes vectors [S0, S1, S2, S3]
# ============================================================
print("=" * 60)
print("FOUR REFERENCE SOPs FOR THE TETRAHEDRON")
print("=" * 60)
print("Format: [S0, S1, S2, S3]")
print()
for i, p in enumerate(points):
    full = np.concatenate(([1.0], p))  # S0 = 1 for all
    print(f"  P{i+1}: {full}")

# ============================================================
# 3. Verify angular distances (should all be 109.47°)
# ============================================================
print("\n" + "=" * 60)
print("ANGULAR DISTANCES BETWEEN POINTS (degrees)")
print("=" * 60)
for i in range(4):
    for j in range(i+1, 4):
        cos_theta = np.dot(points[i], points[j])
        # Clamp to avoid numerical issues
        theta = math.acos(np.clip(cos_theta, -1, 1))
        print(f"  P{i+1} ↔ P{j+1}: {math.degrees(theta):.4f}°")

# ============================================================
# 4. (Optional) Plot on a 3D Poincaré sphere
# ============================================================
fig = plt.figure(figsize=(8, 8))
ax = fig.add_subplot(111, projection='3d')

# Draw the unit sphere
u = np.linspace(0, 2 * np.pi, 100)
v = np.linspace(0, np.pi, 100)
x = np.outer(np.cos(u), np.sin(v))
y = np.outer(np.sin(u), np.sin(v))
z = np.outer(np.ones(np.size(u)), np.cos(v))
ax.plot_surface(x, y, z, color='lightblue', alpha=0.2, edgecolor='none')

# Plot the four vertices
colors = ['red', 'green', 'blue', 'orange']
for i, (p, label, color) in enumerate(zip(points, labels, colors)):
    ax.scatter(p[0], p[1], p[2], color=color, s=80, label=label)
    ax.text(p[0]*1.1, p[1]*1.1, p[2]*1.1, label, fontsize=10)

ax.set_xlabel('S1')
ax.set_ylabel('S2')
ax.set_zlabel('S3')
ax.set_title('Regular Tetrahedron on the Poincaré Sphere')
ax.legend()
ax.set_xlim([-1.1, 1.1])
ax.set_ylim([-1.1, 1.1])
ax.set_zlim([-1.1, 1.1])
ax.set_box_aspect([1, 1, 1])
plt.tight_layout()
plt.savefig('tetrahedron_poincare.png', dpi=150)
plt.show()

print("\n✅ Plot saved as 'tetrahedron_poincare.png'")