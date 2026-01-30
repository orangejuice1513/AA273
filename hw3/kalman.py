# find estimated state trajectory using kalman filter
import numpy as np 
import matplotlib.pyplot as plt

from plot_error_ellipse import *
from trajectory import * 

# definitions 
Q = np.eye(2) #cov(W)
R = 9 * np.eye(2) #cov(V)

# initial conditions 
mu_t = np.array([[1500], [100], [0], [55]]) #initial state 
cov_t = np.block([ #initial covariance 
    [250000 * np.eye(2), np.zeros((2,2))],
    [np.zeros(2,2), np.eye(2)]
])

k_trajectory = np.zeros((N+1, 4)) #stores trajectory (array of states)
k_trajectory[0] = mu_t.ravel() #add the first datapoint

# implement kalman filter 
for i in range(N):
    # update step 
    pred_mu_t = A @ mu_t + B @ mu_t
    pred_cov_t = A @ cov_t @ A.T + Q 

    # calculate kalman gain 
    K_t = pred_cov_t @ C.T @ np.linalg.inv(C @ pred_cov_t @ C.T + R)
    # get measurement noise 
    V_t = np.random.multivariate_normal(mean=np.zeros(2), cov=R).reshape(2, 1)
    # get measurement
    y_t = C @ mu_t + V_t     

    # prediction step 
    mu_t = pred_mu_t + K_t @ (y_t - C @ pred_mu_t)
    cov_t = pred_cov_t - K_t @ C @ pred_cov_t
    
    k_trajectory[i+1] = mu_t.ravel() 


# plot some example trajectories 
p1 = k_trajectory[:, 0]
p2 = k_trajectory[:, 1]

plt.plot(p1, p2, '-m', linewidth=2, label='true trajectory')
plt.xlabel('x position (m)')
plt.ylabel('y position (m)')
plt.title('Kalman Filter State Estimation of Drone Trajectory')
plt.legend() 
plt.savefig("drone_trajectory.png", dpi=300, bbox_inches="tight")
plt.show()
print("finished!")


# plot error ellipse 
