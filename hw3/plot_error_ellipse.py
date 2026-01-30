# plot 2d error ellipse 
import numpy as np 
import matplotlib.pyplot as plt
from scipy.linalg import sqrtm

P = 0.95 

def plot_error_ellipse(P, points, mu, cov, filename):
    """
    given P and a 2d gaussian, plot error ellipse
    """ 
    E = (1 - P) / (2 * np.pi * np.linalg.det(sqrtm(cov)))

    inside = 0 # proportion of points that lie inside the ellipse 
    for point in points: 
        if point_in_ellipse(point, mu, cov, E):
            inside += 1 # count the points inside the ellipse 
    proportion = inside / len(points) 

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
    plt.plot(xy[0], xy[1])
    plt.axis('equal')

