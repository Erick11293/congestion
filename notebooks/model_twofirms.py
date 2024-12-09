#!/usr/bin/env python
# coding: utf-8

# In[1]:


import numpy as np


# In[2]:


import matplotlib.pyplot as plt


# Low type probabilities: p_0, p_1, p_2
# High type probabilitites: q_0, q_1, q_2

# In[3]:


y1_dom = np.linspace(0,1,1000)
y2_dom = np.linspace(0,1,1000)


# In[4]:


y1_mesh, y2_mesh = np.meshgrid(y1_dom, y2_dom)


# ##### Low types

# In[5]:


def compute_equlibrium(prob, s_1, s_2):
    
    p_0, p_1, p_2, q_0, q_1, q_2 = prob
    
    u_0_l = y1_mesh * 0
    u_1_l = theta_x * ((1 - q_2 - q_0) * 0 + (q_2 + q_0) * y1_mesh) + (1 - theta_x) * ((1 - p_2 - p_0) * y1_mesh/2 + (p_2 + p_0) * y1_mesh) - s_1
    u_2_l = theta_x * ((1 - q_1 - q_0) * 0 + (q_1 + q_0) * y2_mesh) + (1 - theta_x) * ((1 - p_1 - p_0) * y2_mesh/2 + (p_1 + p_0) * y2_mesh) - s_2
    u_b_l = ((y1_mesh > y2_mesh) * 
             (theta_x * (q_0 * y1_mesh + q_1 * y2_mesh + q_2 * y1_mesh + (1 - q_0 -q_1 - q_2) * ( 0 )) + 
              (1 - theta_x) * (p_0 * y1_mesh + p_1 * (y1_mesh / 2 + y2_mesh / 2) + p_2 * (y1_mesh) + (1 - p_0 - p_1 - p_2) * (y1_mesh/2 + y2_mesh/4)) 
            )
             + (y1_mesh <= y2_mesh) * 
             (theta_x * (q_0 * y2_mesh + q_1 * y2_mesh + q_2 * y1_mesh + (1 - q_0 -q_1 - q_2) * ( 0 )) + 
              (1 - theta_x) * (p_0 * y2_mesh + p_1 * y2_mesh + p_2 * (y1_mesh / 2 + y2_mesh / 2) + (1 - p_0 - p_1 - p_2) * (y2_mesh/2 + y1_mesh/4)) 
            )
             
            ) - s_1 - s_2
    
    # Stack the matrices into a 3D array
    stacked_matrices = np.stack([u_0_l, u_1_l, u_2_l, u_b_l])
    
    # Find the maximum value across all matrices
    max_value = np.max(stacked_matrices, axis = 0)
    
    # Find the index of the maximum value
    max_index = np.argmax(stacked_matrices, axis = 0)
    
    p_0_n = (max_index == 0).sum()/ max_index.size
    p_1_n = (max_index == 1).sum()/ max_index.size
    p_2_n = (max_index == 2).sum()/ max_index.size
    p_b_n = 1 - p_0_n - p_1_n - p_2_n
    
    
    max_index_cond1 = (max_index + 1) * (y1_mesh > y2_mesh) - 1
    
    n_c1 = ((max_index_cond1 == 0).sum() + (max_index_cond1 == 1).sum() 
            + (max_index_cond1 == 2).sum() + (max_index_cond1 == 3).sum()  
           )
    p_0_c1 = (max_index_cond1 == 0).sum()/n_c1
    p_1_c1 = (max_index_cond1 == 1).sum()/n_c1
    p_2_c1 = (max_index_cond1 == 2).sum()/n_c1
    p_b_c1 = (max_index_cond1 == 3).sum()/n_c1
    
    
    max_index_cond2 = (max_index + 1) * (y1_mesh < y2_mesh) - 1
    
    n_c2 = ((max_index_cond2 == 0).sum() + (max_index_cond2 == 1).sum() 
            + (max_index_cond2 == 2).sum() + (max_index_cond2 == 3).sum()  
           )
    p_0_c2 = (max_index_cond2 == 0).sum()/n_c2
    p_1_c2 = (max_index_cond2 == 1).sum()/n_c2
    p_2_c2 = (max_index_cond2 == 2).sum()/n_c2
    p_b_c2 = (max_index_cond2 == 3).sum()/n_c2
    
    ##### High types
    
    u_0_h = y1_mesh * 0
    u_1_h = theta_x * ((1 - q_2 - q_0) * y1_mesh/2 + (q_2 + q_0) * y1_mesh) + (1 - theta_x) * y1_mesh - s_1
    u_2_h = theta_x * ((1 - q_1 - q_0) * y2_mesh/2 + (q_1 + q_0) * y2_mesh) + (1 - theta_x) * y2_mesh - s_2
    u_b_h = ((y1_mesh > y2_mesh) * 
             (theta_x * (q_0 * y1_mesh  + q_1 * (y1_mesh/2 + y2_mesh/2) + q_2 * y1_mesh + (1 - q_0 -q_1 - q_2) * ( y1_mesh/2 + y2_mesh/4 )) + 
              (1 - theta_x) * y1_mesh 
            )
             + (y1_mesh <= y2_mesh) * 
             (theta_x * (q_0 * y2_mesh + q_1 * (y2_mesh/2 + y1_mesh/2) + q_2 * y1_mesh + (1 - q_0 -q_1 - q_2) * ( y2_mesh/2 + y1_mesh/4 )) + 
              (1 - theta_x) * y2_mesh
            )
            ) - s_1 - s_2
    
    # Stack the matrices into a 3D array
    stacked_matrices = np.stack([u_0_h, u_1_h, u_2_h, u_b_h])
    
    # Find the maximum value across all matrices
    max_value = np.max(stacked_matrices, axis = 0)
    
    # Find the index of the maximum value
    max_index = np.argmax(stacked_matrices, axis = 0)
    
    q_0_n = (max_index == 0).sum()/ max_index.size
    q_1_n = (max_index == 1).sum()/ max_index.size
    q_2_n = (max_index == 2).sum()/ max_index.size
    q_b_n = 1 - q_0_n - q_1_n - q_2_n
    
    # Conditional
    
    max_index_cond1 = (max_index + 1) * (y1_mesh > y2_mesh) - 1
    
    n_c1 = ((max_index_cond1 == 0).sum() + (max_index_cond1 == 1).sum() 
            + (max_index_cond1 == 2).sum() + (max_index_cond1 == 3).sum()  
           )
    q_0_c1 = (max_index_cond1 == 0).sum()/n_c1
    q_1_c1 = (max_index_cond1 == 1).sum()/n_c1
    q_2_c1 = (max_index_cond1 == 2).sum()/n_c1
    q_b_c1 = (max_index_cond1 == 3).sum()/n_c1
    
    
    max_index_cond2 = (max_index + 1) * (y1_mesh < y2_mesh) - 1
    
    n_c2 = ((max_index_cond2 == 0).sum() + (max_index_cond2 == 1).sum() 
            + (max_index_cond2 == 2).sum() + (max_index_cond2 == 3).sum()  
           )
    q_0_c2 = (max_index_cond2 == 0).sum()/n_c2
    q_1_c2 = (max_index_cond2 == 1).sum()/n_c2
    q_2_c2 = (max_index_cond2 == 2).sum()/n_c2
    q_b_c2 = (max_index_cond2 == 3).sum()/n_c2
    
    
    prob_n = [p_0_n, p_1_n, p_2_n, q_0_n, q_1_n, q_2_n]
    
    # Probability of being high type conditional on applying to 1 and 2
    
    h_1 = (theta_x * q_1_n + theta_x * q_b_n) / ( theta_x * q_1_n + (1 - theta_x) * p_1_n + theta_x * q_b_n + (1 - theta_x) * p_b_n)
    h_2 = (theta_x * q_2_n + theta_x * q_b_n) / ( theta_x * q_2_n + (1 - theta_x) * p_2_n + theta_x * q_b_n + (1 - theta_x) * p_b_n)
    
    
    # Probability of preferring 1 being low type conditional on applying to both
    
    f_l = p_b_c1 / ( p_b_c1 + p_b_c2) 
    
    # Probability of preferring 1 being high type conditional on applying to both

    if q_b_c1 + q_b_c2 > 0:
        f_h = q_b_c1 / ( q_b_c1 + q_b_c2) 
    else:
        f_h = 0


    return prob_n, h_1, h_2, f_h


# In[6]:


def get_profits(theta_x, s_1, s_2,c, prob_init = [1,0,0,1,0,0]):
    
    distance = 1000

    prob = prob_init
    
    for i in range(100):
        prob_n, _, _, _ = compute_equlibrium(prob, s_1, s_2)
        distance_n = np.linalg.norm(np.array(prob_n) - np.array(prob))
    
        # Check if the distance is stucked, since the problem is an approximation, it shall never reach the equilibrium
        
        if (distance - distance_n < 0.000001):
            prob_f = list((np.array(prob) + np.array(prob_n))/2)
            break
        else:
            prob = prob_n
            distance = distance_n
        
    
    prob_n, h_1, h_2, f_h = compute_equlibrium(prob, s_1, s_2)
    
    p_0, p_1, p_2, q_0, q_1, q_2 = prob_n
    
    p_b = 1 - p_0 - p_1 - p_2
    q_b = 1 - q_0 - q_1 - q_2
    
    ##### Benefits
    
    
    pi_1 = (h_1 ** 2 * ( q_1 ** 2 * (1 - 2*c) + 2 * q_1 * (q_0 + q_2) * (1 - c) + q_b ** 2 * (1 - (1 - f_h)**2 - 2 * c) + 2 * q_b * (q_0 + q_2) * (f_h - c) + 2 * q_b * (q_1) * (f_h/2 + 1/2 - 2*c)) +
          2 * h_1 * (1-h_1) * ( q_1 * (p_1 + p_b) * (1 - 2*c) + q_1 * (p_0 + p_2) * (1 - c) + q_b * (p_1 + p_b) * (f_h - 2*c) + q_b * (p_0 + p_2) * (f_h - c)) +
          (1 - h_1) ** 2 * ( (p_1 + p_b) ** 2 * (- 2*c) + 2 * (p_1 + p_b) * (p_0 + p_2) * (- c))
         )
    
    pi_2 = (h_2 ** 2 * ( q_2 ** 2 * (1 - 2*c) + 2 * q_2 * (q_0 + q_1) * (1 - c) + q_b ** 2 * (1 - (f_h)**2 - 2 * c) + 2 * q_b * (q_0 + q_1) * (1 - f_h - c) + 2 * q_b * (q_2) * ((1 - f_h)/2 + 1/2 - 2*c)) +
          2 * h_2 * (1-h_2) * ( q_2 * (p_2 + p_b) * (1 - 2*c) + q_2 * (p_0 + p_1) * (1 - c) + q_b * (p_2 + p_b) * (1 -f_h - 2*c) + q_b * (p_0 + p_1) * (1 - f_h - c)) +
          (1 - h_2) ** 2 * ( (p_2 + p_b) ** 2 * (- 2*c) + 2 * (p_2 + p_b) * (p_0 + p_1) * (- c))
         )

    return pi_1, pi_2


# In[7]:


from scipy import optimize


# In[8]:


def get_reaction(s_2, theta_x, c):
    # Get reaction given s_2
    result = optimize.minimize(
            lambda signal: -get_profits(theta_x, signal, s_2,c)[0],
            s_2,
            method='Nelder-Mead',
            bounds = [(0, 1)],
            options={
                'maxiter': 1000,    # Maximum number of iterations
                'disp': False,       # Display progress
                'adaptive': True    # Use adaptive parameters
            }
        )
    return result.x[0]


# In[9]:


s2_grid = np.linspace(0.05,0.95,100)


# In[10]:


c = 0.2


# In[11]:


theta_x = 0.2


# In[ ]:


reactions = [*map(lambda s2: get_reaction(s2, theta_x, c), s2_grid)]


# In[ ]:


plt.plot(s2_grid, s2_grid)
plt.plot(s2_grid, reactions)


# In[ ]:





# In[ ]:




