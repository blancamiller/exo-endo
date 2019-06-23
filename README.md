# exo-endo
Recreate and extend Dietterich et. al. (2019)

# Discovering and Removing Exogenous State Variables and Rewards for Reinforcement Learning

__Goal:__ Discover/learn the exogenous and endogenous state variables from training data. This discovery is accomplished by:


__(1)__ Learning 3 functions, F_exo, F_end, and G, that are parametrized by w_x, w_e, w_G, s.t. 

```
	x = F_exo(s; w_x)
	e = F_end(s; w_e)
	s = G(F_exo(s), F_end(s); w_G), where s recovers the exogenous and endogenous state parts.
```


__(2)__ Capturing as much exogenous state as possible s.t. the following constraints are met:

    - satisfy the conditional independence relationship: P(s'|s,a) = P(x'|x)P(e',x'|e,x,a)
    - recover the original state from the exogenous and endogenous parts   


We can then fomulate the MDP decomposition problem as an optimization problem where our goal is to maximize the expected size of F_exo:

```
	   argmax	EXP [ |F_exo(s', w_x)| ]
	w_x,w_e,w_G
	    s.t.	^I( F_exo(s';w_x) ;  [F_end(s; w_e), a] | F_exo(s; w_x) ) < e 
	    		EXP [ ||G(F_exo(s'; w_x), F_end(s', w_e); w_G )  - s'|| ] < e'
```

## Problem 3 Set-up: High dimensional linear system

#### State MDP: 30-dimensional with exogenous and endogenous state variables defined as:

````
	X_{t} = [X_1t, ..., X_15t]^T
	E_{t} = [E_1t, ..., E_15t]^T
````
#### State Transition Function:

````
     X_t+1 = M_x * X_t + epsilon_x
     
		   |E_t|
     E_t+1 = M_e * |X_t| + epsilon_e
     	     	   |A_t|
````

  where
  - M_x is element of the Reals 15x15 is the transition function for the exogenous MRP and
  - M_e is element of the Reals 15x31 is the transition function for the endogenous MDP and involves E_t, X_t, and A_t, and
    - E_t: the current time, t, endogenous state variable 
    - X_t: the current time, t, endogenous state variable
    - A_t: the current time, t, action variable
  - epsilon_x is element of the Reals 15x1 is the exogenous noise, whose elemetns are distributed accoding to N(0, 0.09)
  - epsilon_e is element of the Reals 51x1 is the endogenous noise, whose elements are distributed accodgin to N(0, 0.04)

#### State Vector

- S_t is an observed state vector and a linear mixture of the hidden exogenous and endogenous states
- S_t = M * [], where M is an element of the Reals 30x30

The elements in M_x, M_e and M are generated according to N(0, 1)
    - each row of each matrix is normalized to sum to 0.99 for stability
    - the starting state is the zero vector

#### Reward at time t and its corresponding exogenous and endogenous reward 
- R_t = R_xt + R_et
  - Exogenous reward: R_xt = -3 * avg(X_t)
  - Endogenous reward: R_et = exp[-|avg(E_t) - 1|],
  
  where average is the average over a vector's elements
  

## 4 Q-learning Configurations:

1. Full-MDP: Q-learning on the full MDP 
2. Endo MDP Global: Q-learning on the decomposed MDPs discovered by the Global (algorithm 1)
3. Endo MDP Stepwise: Q-learning on the Stepwise state decomposition (algorithm 2) 
4. End MDP Oracle: Q-learning on the true endogenous MDP

## Experimental Set-up

- The Q function is represented as a neural network with a single hidden later of 20 tanh units and a linear output layer.
- The Q-learning updates are implemented with stochastic gradient descent.
- Exploration is achieved via Boltzmann exploration with temperature parameter Beta.
- Given the current Q-values, the action a_t is selected according to:

```

  	    	       exp( Q(s_t, a) / beta )
  a_t ~ pi(a|s) = _________________________________

		  sum_i( exp( Q(s_t, a_i) / beta ))

```