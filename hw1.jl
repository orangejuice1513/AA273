# AA273 Homework 1 Problem 4 Part b 
using Printf
using LinearAlgebra
# intialization 
infection_rate = 0.001
recovery_rate = 0.05 
T = [ 1 - infection_rate recovery_rate;
      infection_rate 1 - recovery_rate]
M_n = [0.9 0;
       0 0.01] # measurement likelihood for negative test 
M_p = [0.1 0;
       0 0.99] # measurement likelihood for positive test 
pi_t = [1.0 0.0]' # initial state 
one_vec = ones(2,1)

# recursive bayes 
for i in 1:98 #98 negative tests 
    pi_t = M_n * T * pi_t / (one_vec.T * M_n * T * pi_t) 
end 

# 99th day is positive 
pi_t = M_p * T * pi_t / (one_vec' * M_p * T * pi_t)
display(pi_t) 
@printf("Probability: %f", pi_t[2])