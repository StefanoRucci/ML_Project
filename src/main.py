import gym
import time
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
    randomAction= env.action_space.sample()
    #print("Random action: ", randomAction)
    returnValue=env.step(randomAction)
    #print("ReturnValue: ", returnValue)
    env.render()
    print('Iteration: {} and action {}'.format(i+1,randomAction))
    reward+=returnValue[1]
    time.sleep(2)
    if returnValue[1]==-100:
        print("The reward is: ",reward)
        reward=0
        env.reset()
    if returnValue[0] == 47:
        print("The reward is: ",reward)
        reward=0
        env.reset()

    print("##########################################")

print("The reward is: ",reward)
env.close()