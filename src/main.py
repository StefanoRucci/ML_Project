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
env.close()