# AA273 Homework 2 Problem 4 
import numpy as np 
import matplotlib.pyplot as plt
from scipy.linalg import sqrtm

N_SAMPLES = 1000 #number of samples
P = 0.95 

def plot_error_ellipse(P, N_SAMPLES, mu, cov, filename):
    """
    given P and a 2d gaussian, plot error ellipse
    """ 
    samples = np.random.multivariate_normal(mu, cov, size=N_SAMPLES)
    E = (1 - P) / (2 * np.pi * np.linalg.det(sqrtm(cov)))
    plt.scatter(*samples.T, s=2, alpha=0.3) 
    plot_ellipse(mu, cov, P)

    inside = 0 # proportion of points that lie inside the ellipse 
    for sample in samples: 
        if point_in_ellipse(sample, mu, cov, E):
            inside += 1 # count the points inside the ellipse 
    proportion = inside / N_SAMPLES 

    plt.grid(True)
    plt.title(f"95% Confidence Ellipse {filename}")
    print(f"proportion of points inside ellipse{filename}: {proportion}")
    plt.savefig(filename, dpi=300) 
    plt.close()
    return 

def point_in_ellipse(point, mu, cov, E):
    """
    returns true if the point is in the ellipse (2d)
    """
    const = 1 / (2 * np.pi * np.linalg.det(sqrtm(cov)))
    return (point - mu).T @ np.linalg.inv(cov) @ (point - mu) <= 2 * np.log(const / E)

def plot_ellipse(mu, cov, P):
    s = -2 * np.log(1 - P) 
    t = np.linspace(0, 2*np.pi, 100)
    circle = np.array([np.cos(t), np.sin(t)]) * np.sqrt(s)
    xy = np.array(mu)[:, None] + sqrtm(cov).real @ circle
    
    plt.plot(xy[0], xy[1], label=f'{int(P*100)}%')
    plt.axis('equal')

plot_error_ellipse(P, N_SAMPLES, [0, 0], [[10, 2], [2, 2]], "1")
plot_error_ellipse(P, N_SAMPLES, [5, 2], [[2, 0], [0, 2]], "2")
plot_error_ellipse(P, N_SAMPLES, [20, 3], [[2, 1.5], [1.5, 2]], "3")

