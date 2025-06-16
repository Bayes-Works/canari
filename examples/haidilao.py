import numpy as np
import matplotlib.pyplot as plt

# Constants
original_beef_price = 11.95
promo_beef_price = 3.95
general_discount = 0.31

# Variables to explore
beef_counts = np.arange(0, 30)  # Number of beef plates
total_price = np.linspace(0, 300, 300)  # Total original price

# Create 2D meshgrid
beef_grid, total_grid = np.meshgrid(beef_counts, total_price)

# Calculate non-beef portion
non_beef_total = np.maximum(total_grid - beef_grid * original_beef_price, 0)

# Compute promo costs
cost_promo1 = beef_grid * promo_beef_price + non_beef_total
cost_promo2 = (beef_grid * original_beef_price + non_beef_total) * (1 - general_discount)

# Compute savings
savings = cost_promo2 - cost_promo1  # Positive = Promo 1 better

print(savings)
print(beef_grid)
print(total_grid)

# Plot correctly aligned heatmap and contour
plt.figure(figsize=(10, 6))
im = plt.imshow(
    savings,
    extent=[beef_counts.min(), beef_counts.max(), total_price.min(), total_price.max()],
    aspect='auto',
    origin='lower',
    cmap='coolwarm'
)

# Add correct contour on the same (x, y) scale
cs = plt.contour(
    beef_grid, total_grid, savings, levels=[0], colors='black', linewidths=1.5, linestyles='--'
)

plt.clabel(cs, fmt='no saving', colors='black')

# Final plot settings
plt.colorbar(im, label='Savings with beef promotion ($)')
plt.xlabel('Number of Beef Plates')
plt.ylabel('Total Price Before Discount ($)')
plt.title('Which to choose in HaiDiLao? Beef Promotion vs General Discount')
plt.grid(True, linestyle='--', alpha=0.3)
plt.tight_layout()

# # Put black color on the parts where price for beef is bigger than total price
# plt.imshow(
#     (beef_grid * original_beef_price < total_grid).astype(int),
#     extent=[beef_counts.min(), beef_counts.max(), total_price.min(), total_price.max()],
#     aspect='auto',
#     origin='lower',
#     cmap='gray',
#     alpha=0.5
# )
plt.show()
