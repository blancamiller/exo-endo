# exo-endo
Recreate and extend Dietterich et. al. (2019)

# Discovering and Removing Exogenous State Variables and Rewards for Reinforcement Learning

__Goal:__ learn the exogenous state variables from data 


## Problem 3 Set-up: High dimensional linear system

#### 30-dimensional State MDP, where the state variables are:
- 15 exogenous: X_{t} = [X_1t, ..., X_15t]^T
- 15 endogenous: E_{t} = [E_1t, ..., E_15t]^T

#### State transition function:

     X_t+1 = M_x * X_t + epsilon_x

     E_t+1 = M_e *     + epsilon_e

  where
  - M_x is element of the Reals 15x15 is the transition function for the exogenous MRP and
  - M_e is element of the Reals 15x31 is the transition function for the endogenous MDP and involves E_t, X_t, and A_t, and
    - E_t: the current time, t, endogenous state variable 
    - X_t: the current time, t, endogenous state variable
    - A_t: the current time, t, action variable
  - epsilon_x is element of the Reals 15x1 is the exogenous noise, whose elemetns are distributed accoding to N(0, 0.09)
  - epsilon_e is element of the Reals 51x1 is the endogenous noise, whose elements are distributed accodgin to N(0, 0.04)

- S_t is an observed state vector and a linear mixture of the hidden exogenous and endogenous states
- S_t = M * [], where M is an element of the Reals 30x30

The elements in M_x, M_e and M are generated according to N(0, 1)
    - each row of each matrix is normalized to sum to 0.99 for stability
    - the starting state is the zero vector

Reward at time t
- R_t = R_xt + R_et
  - Exogenous reward: R_xt = -3 * avg(X_t)
  - Endogenous reward: R_et = exp[-|avg(E_t) - 1|],
  
  where average is the average over a vector's elements
  

## 4 Q-learning Configurations:

1.
2.
3.
4. 