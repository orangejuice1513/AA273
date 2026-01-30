# find estimated state trajectory using kalman filter with a different sensor model 
# (only velocity sensor)
import numpy as np 
import matplotlib.pyplot as plt

from plot_error_ellipse import *
from trajectory import * 

N = 50 # number of timesteps 
v_scale = 10 # scaling factor to see velocity vectors on graph 

# definitions 
Q = np.block([
    [np.zeros((2, 2)), np.zeros((2, 2))],
    [np.zeros((2, 2)), np.eye(2)]
    ]) #cov(W)
R = 9 * np.eye(2) #cov(V)
C = C = np.block([
    [np.zeros((2, 2)), np.eye(2)]
])

# initial conditions 
x_true = np.array([[1000], [0], [0], [50]])  #initial state 
mu_t = np.array([[1000], [0], [0], [50]])  #initial state 
cov_t = np.block([ #initial covariance 
    [np.eye(2), np.zeros((2,2))],
    [np.zeros((2,2)), np.eye(2)]
])
t = 0
plt.figure(figsize=(12, 8))

k_trajectory = np.zeros((N+1, 4)) #stores trajectory (array of states)
k_trajectory[0] = mu_t.ravel() #add the first datapoint
true_path = np.zeros((N+1, 4)) #stores true path 
true_path[0] = x_true.ravel() #add the first datapoint 

# implement kalman filter 
for t in range(N):
    W_t = np.random.multivariate_normal(np.zeros(4), Q).reshape(4, 1)
    V_t = np.random.multivariate_normal(np.zeros(2), R).reshape(2, 1)
    x_true = A @ x_true + B @ u_t + W_t # true state

    # prediction step 
    pred_mu_t = A @ mu_t + B @ u_t
    pred_cov_t = A @ cov_t @ A.T + Q 

    # calculate kalman gain 
    K_t = pred_cov_t @ C.T @ np.linalg.inv(C @ pred_cov_t @ C.T + R)

    # get measurement
    y_t = C @ x_true + V_t 

    # update step 
    mu_t = pred_mu_t + K_t @ (y_t - C @ pred_mu_t)
    cov_t = pred_cov_t - K_t @ C @ pred_cov_t
    
    k_trajectory[t+1] = mu_t.ravel() 
    true_path[t+1] = x_true.ravel() 

    # plot 95% error ellipses for position 
    # plot_ellipse(mu_t[0:2].ravel(), cov_t[0:2, 0:2], 0.95)

    # plot 95% error ellipses for velocity 
    if t % 6 == 0:
        p = mu_t[0:2].ravel()
        v = mu_t[2:4].ravel()
        v_tip_scaled = p + (v * v_scale) #scale vector so we can see
        v_cov_scaled = cov_t[2:4, 2:4] * (v_scale**2)
        plt.quiver(p[0], p[1], v[0]*v_scale, v[1]*v_scale, 
                   angles='xy', scale_units='xy', scale=1, 
                   color='blue', alpha=0.6, width=0.003, 
                   label='Velocity' if t==0 else "") 
        plot_ellipse(v_tip_scaled, v_cov_scaled, 0.95)

# plot some example trajectories 
p1 = k_trajectory[:, 0]
p2 = k_trajectory[:, 1]
x1 = true_path[:, 0]
x2 = true_path[:, 1]

# plt.plot(p1, p2, '-m', linewidth=2, label='kalman filter estimated path')
plt.plot(x1, x2, '-b', linewidth=2, label='true path')
plt.plot()
plt.xlabel('x position (m)')
plt.ylabel('y position (m)')
plt.title('Kalman Filter State Estimation of Drone Trajectory with Velocity Ellipses')
plt.legend() 
plt.axis('equal') 
plt.savefig("p5_velocity_ellipses.png", dpi=300, bbox_inches="tight")
plt.show()
print("finished!")


