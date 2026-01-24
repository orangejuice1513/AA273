# AA273 Homework 2 Problem 4 
import numpy as np 
from scipy import linalg 


N_SAMPLES = 1000 #number of samples
P = 0.95 

def plot_error_ellipse(P, N_SAMPLES, mu, cov):
    """
    given P and a 2d gaussian, plot error ellipse
    """ 
    # get samples  
    samples = np.random.multivariate_normal(mu, cov, size=N_SAMPLES)

    # make error ellipse 
    ellipse = (1 - P) / (2 * np.pi * scipy.linalg.sqrtm(cov))

    # plot 


    # proportion of points that lie inside the ellipse 
    inside = 0
    for sample in samples: 
        if sample < ellipse:
            inside += 1 # count the points inside the ellipse 
    proportion = inside / N_SAMPLES 
    print(f"Proportion of points inside ellipse:", proportion)
    return 

plot_error_ellipse(P, N_SAMPLES, [0, 0], [[1, 0.5], [0.5, 1]])
plot_error_ellipse(P, N_SAMPLES, [5, 2], [[1, 1], [1, 1]])
plot_error_ellipse(P, N_SAMPLES, [20, 3], [[5, 1], [1, 5]])












