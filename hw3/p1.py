#problem 1 
import numpy as np
import numpy as np 
import matplotlib.pyplot as plt
from plot_error_ellipse import *

mu_x = np.array([0, 0])
cov_x = np.array([
    [0.7, 0.73],
    [0.73, 1.1]
])
mu_y = np.array([0, 0])
cov_y = np.array([
    [0.7, 0.19],
    [0.19, 0.16]
])
cov_xy = np.array([
    [0.63, 0.23],
    [0.72, 0.31]
])
y = np.array([0.27, 0.62])

# calculate posterior mean and covariance 
mu_pos = mu_x + cov_xy @ np.linalg.inv(cov_y) @ (y - mu_y)
cov_pos = cov_x - cov_xy @ np.linalg.inv(cov_y) @ cov_xy.T 
print(mu_pos)
print(cov_pos)

plt.figure(figsize=(8, 8))
plot_ellipse(mu_x, cov_x, 0.95) # 95% ellipse for prior 
plt.gca().get_lines()[-1].set_label('Prior (95%)')
plot_ellipse(mu_pos, cov_pos, 0.95) # 95% elliipse for posterior 
plt.gca().get_lines()[-1].set_color('red')
plt.gca().get_lines()[-1].set_label('Posterior (95%)')
plt.scatter(mu_pos[0], mu_pos[1], color='red', marker='o', label='Posterior Mean')
plt.scatter(y[0], y[1], color='green', marker='o', s=100, label='GPS Measurement y')

plt.axhline(0, color='black', alpha=0.3)
plt.axvline(0, color='black', alpha=0.3)
plt.legend()
plt.axis('equal')
plt.title("Problem 1")
plt.savefig("p1.png", dpi=300, bbox_inches="tight")
plt.show()