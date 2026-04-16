import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import norm, lognorm
from fractions import Fraction

# === PLACEHOLDERS ===
# Normal CDF parameters

x_label = "Input"
y_label = "CDF"
custom_xticks = [0, 0.25, 0.5, 0.75, 1]

# Lognormal parameters
# === COMPUTATION ===
# j1 - Normal CDF over (0, 1)
x1 = np.linspace(0, 1, 500)
mean = 0.5  # mean of normal
std_dev = 0.2  # std deviation of normal
j1 = norm.cdf(x1, loc=mean, scale=std_dev)

# j2 - Lognormal CCDF (complementary CDF)
x2 = np.linspace(0.01, 1, 500)  # avoid 0 for lognormal
j2_median = 0.1  # similar to mean 
j2_shape = 0.2  # similar to std deviation
j2 = 1 - lognorm.cdf(x2, s=j2_shape, scale=j2_median)

# j3 - Lognormal CCDF
x3 = np.linspace(0, 1, 500)
j3_median = 0.3  # similar to mean 
j3_shape = 0.6  # similar to std deviation
j3 = 1 - lognorm.cdf(x3, s=j3_shape, scale=j3_median)  # avoid log(0)

# === PLOT ===
fig, axes = plt.subplots(1, 3, figsize=(10, 4))

# --- Subplot 1: Normal CDF ---
axes[0].plot(x1, j1, color="blue", linewidth=2)
axes[0].set_xlim(0, 1)
axes[0].set_ylim(0, 1)
axes[0].set_xlabel("$\Pr$(detection)")
axes[0].set_ylabel("$j_1$")
axes[0].set_xticks(custom_xticks)
axes[0].set_yticks([0, 1])
# axes[0].set_title("$\Pr$(detection)")
axes[0].grid(True, linestyle="--", alpha=0.6)

# --- Subplot 2: Lognormal Complementary CDF ---
axes[1].plot(x2, j2, color="red", linewidth=2)
axes[1].set_xlim(0, 1 / 5)
axes[1].set_ylim(0, 1)
axes[1].set_xlabel("$\Pr$(false alarms/yr)")
axes[1].set_ylabel("$j_2$")
axes[1].set_yticks([0, 1])
axes[1].grid(True, linestyle="--", alpha=0.6)
# axes[1].set_title("Lognormal Complementary CDF")

# Custom fractional x-ticks
x_ticks_frac = [1 / 20, 1 / 10, 1 / 5]
axes[1].set_xticks(x_ticks_frac)
axes[1].set_xticklabels([f"{Fraction(x).limit_denominator()}" for x in x_ticks_frac])

# --- Subplot 3: Lognormal CDF ---
axes[2].plot(x3, j3, color="green", linewidth=2)
axes[2].set_xlim(0, 1)
axes[2].set_ylim(0, 1)
axes[2].set_xlabel("Anm. magnitude [unit/yr]")
axes[2].set_ylabel("$j_3$")
axes[2].set_yticks([0, 1])
# axes[2].set_title("Lognormal CDF")
axes[2].grid(True, linestyle="--", alpha=0.6)

# Fractional x-ticks for Lognormal CDF
x_ticks_frac_cdf = [0, 0.1, 0.3, 0.6]
# axes[2].set_xticks(x_ticks_frac_cdf)
# axes[2].set_xticklabels(
#     [f"{Fraction(x).limit_denominator()}" for x in x_ticks_frac_cdf]
# )

# j_global = j1 * j2 * j3

plt.tight_layout()
plt.savefig("out.png")
