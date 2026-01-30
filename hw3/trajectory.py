# simulate trajectory and measurement process of drone 
import numpy as np 

# system matrix definitions 
N = 1000
t = 0
A = np.block(
    [np.eye(2), np.eye(2)],
    [np.zeros((2, 2)), np.eye(2)]
)
B = np.block([
    [np.zeros((2, 2))],
    [np.eye(2)]
])
C = np.block([
    [np.eye(2), np.zeros((2, 2))]
])
Q = np.eye(2) #cov(W)
R = 9 @ np.eye(2) #cov(V)
u_t = -2.5[ #control 
    [np.cos(0.05 * t)], 
    [np.sin(0.05 * t)]
]

# initial conditions 
p_0 = [[1000], [0]] #initial position 
v_0 = [[0], [50]] #initial velocity 

trajectory = np.zeros(N) #save trajectory here 

# trajectory simulation 
for i in range(N):
    # take noise samples 
    W_t = np.random.multivariate_normal([0, 0].T, Q, 1)
    V_t = np.random.multivariate_normal([0, 0].T, R, 1)
    
    # propogate forward 
    x_t = A @ x_t + B @ u_t + np.block([0, 0], [W_t])
    y_t = C @ x_t + V_t

    trajectory.append(x_t) 

# plot some example trajectories 



