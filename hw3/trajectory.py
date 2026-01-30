# simulate trajectory and measurement process of drone 
from turtle import position
import numpy as np 
import matplotlib.pyplot as plt 

# system matrix definitions 
N = 1000
t = 0
A = np.block([
    [np.eye(2), np.eye(2)],
    [np.zeros((2, 2)), np.eye(2)]
])
B = np.block([
    [np.zeros((2, 2))],
    [np.eye(2)]
])
C = np.block([
    [np.eye(2), np.zeros((2, 2))]
])
Q = np.eye(2) #cov(W)
R = 9 * np.eye(2) #cov(V)
u_t = -2.5 * np.array([ #control
    [np.cos(0.05 * t)], 
    [np.sin(0.05 * t)]
])

# initial conditions
p_t = np.array([[1000], [0]]) #initial position 
v_t = np.array([[0], [50]]) #initial velocity 
x_t = np.vstack((p_t, v_t)) #initial state 

trajectory = np.zeros((N+1, 4)) #stores trajectory 
trajectory[0] = x_t.ravel() #add the first datapoint

measurements = np.zeros((N+1, 2)) #stores gps measurements 

# trajectory simulation 
for i in range(N):
    # take noise samples 
    W_t = np.random.multivariate_normal(mean=np.zeros(2), cov=Q).reshape(2, 1)
    V_t = np.random.multivariate_normal(mean=np.zeros(2), cov=R).reshape(2, 1)

    # propogate forward
    x_t = A @ x_t + B @ u_t + np.vstack((np.zeros((2,1)), W_t)) 
    y_t = C @ x_t + V_t

    trajectory[i+1] = x_t.ravel()
    measurements[i:1] = y_t.ravel()

# plot some example trajectories 
p1 = trajectory[:, 0]
p2 = trajectory[:, 1]
y1 = measurements[:, 0]
y2 = measurements[:, 1]

plt.plot(p1, p2, '-m', linewidth=2, label='true trajectory')
plt.scatter(y1, y2, s=10, alpha=0.6, label='gps measurements')
plt.xlabel('x position (m)')
plt.ylabel('y position (m)')
plt.title('Drone Trajectory')
plt.legend() 
plt.savefig("drone_trajectory.png", dpi=300, bbox_inches="tight")
plt.show()
print("finished!")



