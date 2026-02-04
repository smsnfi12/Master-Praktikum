import numpy as np

rng = np.random.default_rng(156)  # Seed optional

new_vals = np.array(['r0 +0', 'r90 +1', 'g0 +2', 'g-45 x0', 'g45 x1', 'r-45 x2'])
farbe_vals = np.array(['r', 'g'])
x_vals = np.array([0, 1, 2])
rot_vals = np.array([-45, 0, 45])
green_vals = np.array([-45, 0, 90])
y_vals = np.array([0, 45])
z_vals = np.array([0, 45])

new = rng.choice(new_vals, size=52)
farbe = rng.choice(farbe_vals, size=52)
rot = rng.choice(rot_vals, size=52)
green = rng.choice(green_vals, size=52)
x = rng.choice(x_vals, size=52)
y = rng.choice(y_vals, size=52)
z = rng.choice(z_vals, size=52)

#for i, (farbe_i, xi, yi, zi) in enumerate(zip(farbe, x, y, z), start=1):
#    print(f"{i:02d} {farbe_i}   x={xi:>3}, y={yi:>2}, z={zi:>2}")

for i, (new_i, yi, zi) in enumerate(zip(new, y, z), start=1):
    print(f"{i:02d} {new_i}   y={yi:>2}, z={zi:>2}")

