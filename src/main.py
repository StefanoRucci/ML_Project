import gym
import time
import numpy as np
import random
from IPython.display import clear_output

# setup the environment
env=gym.make("CliffWalking-v0",render_mode='human')
env.reset()
env.render()
print('Initial state of the system')

# observation space - states of the environment (they are 48) 
print("Observation space", env.observation_space)

# actions: up->0, right->1, down->1, left->3.
print("Action space", env.action_space)

# create the Q-Table
state_space_size = env.observation_space.n
action_space_size = env.action_space.n

#Creating a q-table and intialising all values as 0
q_table = np.zeros((state_space_size,action_space_size))
print(q_table)

#Number of episodes
num_episodes = 1000
#Max number of steps per episode
max_steps_per_episode = 300

learning_rate = 0.1
discount_rate = 0.99

#Greedy strategy
exploration_rate = 1
max_exploration_rate = 1
min_exploration_rate = 0.01
exploration_decay_rate = 0.01

rewards_all_episodes = [] #List to contain all the rewards of all the episodes given to the agent

#Q-Learning Algorithm

for episode in range(num_episodes): #Contains that happens in an episode
    state = env.reset()
    state=state[0]
    print("NEW EPISODE NUMBER ", episode)
    
    done = False #Tells us whether episode is finished
    rewards_current_episode = 0 #We start with 0 each episode

    for step in range(max_steps_per_episode): #Contains that happens in a time step
        
        #Exploration-exploitation trade off
        exploration_rate_threshold = random.uniform(0,1)
        
        if exploration_rate_threshold > exploration_rate:
            action = np.argmax(q_table[state,:])
        else:
            action = env.action_space.sample()

        #Taking action
        returnValue=env.step(action) 
        new_state = returnValue[0]
        reward = returnValue[1]
        done = returnValue[2]
        info = returnValue[3]

        #Update Q-table for Q(s,a)
        q_table[state, action] = q_table[state, action] * (1 - learning_rate) + learning_rate * (reward + discount_rate * np.max(q_table[new_state, :]))

        state = new_state
        rewards_current_episode += reward

        if state == 47: #Checking if episode is over
            break

    # Exploration rate decay
    exploration_rate = min_exploration_rate + (max_exploration_rate - min_exploration_rate) * np.exp(-exploration_decay_rate*episode)
    print("Expolarion rate:", exploration_rate)
    print("Reward:", rewards_current_episode)

    rewards_all_episodes.append(rewards_current_episode)

# Calculate and print the average reward per thousand episodes
rewards_per_thousand_episodes = np.split(np.array(rewards_all_episodes),num_episodes/1000)
count = 1000

print("********Average reward per thousand episodes********\n")
for r in rewards_per_thousand_episodes:
    print(count, ": ", str(sum(r/1000)))
    count += 1000#Print the updates Q-Table
print("\n\n*******Q-Table*******\n")
print(q_table)
"""
numberOfIterations=30

reward=0

for i in range(numberOfIterations):
    randomAction= env.action_space.sample() #This is a randomaction
    #print("Random action: ", randomAction)
    returnValue=env.step(randomAction) #The return state of the random action
    #print("ReturnValue: ", returnValue)
    env.render()
    print('Iteration: {} and action {}'.format(i+1,randomAction))
    reward+=returnValue[1]
    time.sleep(2)

    # if fall in the climb
    if returnValue[1]==-100:
        print("The reward is: ",reward)
        reward=0
        env.reset()

    # if goal
    if returnValue[0] == 47:
        print("The reward is: ",reward)
        reward=0
        env.reset()

    print("##########################################")

print("The reward is: ",reward)
"""
env.close()