import numpy as np

"""

# Hyperparameters:

   alpha: the learning rate/step-size   

   epsilon: a value that adjusts exploration & exploitation; as our agent learns, we need less exploration and more exploitation to yield more utility for our policy; as the nunber of episodes increases, the value of epsilon will decrease

   gamma: the discount value that gives greater weight to near-term reward and less weight to long-tem reward; gamma descreases exponentially based on the number of steps left in the episode 

   num_episodes: the number of episodes we want to run q-learning algorithm 

"""


def q_learning(alpha=0.1, epsilon=0.1, gamma=0.90, num_episodes=100):

    # Randomly initialize Q(s,a), except Q(terminal, *)=0
    q_init = np.random.choice() # -- change the Q-value fcn. depending on the environment  
    
    # Loop for each episode
    for current_episode in range(num_episodes):

        done = False 
        while not done:

            # Initialize the starting state
            s = 0 # -- change depending on the environment 
            
            # Loop for each step of the episode:
            for step in range(len(current_episode)):
                
                # Choose action, a, from the current state, s, using a policy derived from Q
                
                
                #Take action a, observe r and next_state
                
                
                # Q-Fcn: Q(S,A)<--Q(S,A) + alpha[R+gamma * max_aQ(S',a)-Q(S,A)]
                
                
                # S <-- S'
                
                
            # Until S is terminal 


        
        
