import numpy as np
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
import gym
import random
from collections import deque

# Creazione dell'ambiente CliffWalking-v0
env = gym.make('CliffWalking-v0', render_mode="human")

# past experience
EXP_MAX_SIZE=10000 # Max batch size of past experience
BATCH_SIZE=EXP_MAX_SIZE//10 # Training set size
experience = deque([],EXP_MAX_SIZE) # Past experience arranged as a queue

EPS_MAX = 70 # Initial exploration probability
EPS_MIN = 5 # Final exploration probability
GAMMA = .9 # discount factor
LR = 0.01 # learning rate
c_reward = 0 # current cumulative reward
checkpoint_path = './checkpoints/cp.ckpt' # file to record network configuration

# Use a NN to Q-function Q(obs,a)
# NN architecture
model = Sequential()
model.add(Dense(32, input_shape=(2,), activation='relu'))
model.add(Dense(16, activation='relu'))
model.add(Dense(1,activation='linear'))
model.compile(optimizer='sgd', loss='mse')

# initialize environment and get related information and observation
# obs represents current state
#state, info = env.reset(seed=42)

episode = 1 # counting episodes to decrease epsilon
epsilon = EPS_MAX # start with max exploration probability

# Addestramento dell'agente
num_episodes = 500
max_steps_per_episode = 40

for i in range(num_episodes):
    #print(epsilon)
    state, info = env.reset()
    #print(state)
    done = False
    total_reward = 0

    for step in range(max_steps_per_episode):

        action = env.action_space.sample() # pick random action (default)
        rv = random.randint(1,100) # pick random int to decide exploration or exploitation

        if rv >= epsilon:
            candidates = {}
            for a in range(4):
                candidates[a] = model.predict_on_batch(tf.constant([[state, a]]))[0][0]
            action=min(candidates, key=candidates.get)

        returnValue=env.step(action) 
        next_state = returnValue[0]
        reward = returnValue[1]
        done = returnValue[2]
        info = returnValue[3]

        c_reward += reward # cumulate rewaed (for evaluation only)

        # Find next best action max_a q(s',a), observe s' is obs_next
        candidates_next = {}
        for a in range(4):
            candidates_next[a]= model.predict_on_batch(tf.constant([[next_state, a]]))[0][0]
        act_next=max(candidates_next, key=candidates_next.get)

        #Compute corresponding (predicted) reward
        reward_next = candidates_next[act_next]

        # Record experience (will be used to train network)
        if len(experience)>=EXP_MAX_SIZE:
            experience.popleft() # dequeue oldest item

        experience.append([*[state, action], reward + GAMMA*reward_next]) # queue new experience item

        if state == 47:
            break

        state = next_state # update current state

    if len(experience) >= BATCH_SIZE and episode % 10 == 0:
        # sample batch
        batch = random.sample(experience, BATCH_SIZE)
        print(batch)
        #prepare data
        dataset = np.array(batch)
        print(dataset)
        X = dataset[:,:2]
        Y = dataset[:,2]
        print("len(X)=",len(X))
        #train network
        model.fit(tf.constant(X),tf.constant(Y), validation_split=0.2) # fit model
        model.save_weights(checkpoint_path) # save updated weights
        epsilon -= epsilon/100 # reduce epsilon by 1/100
        if epsilon<=EPS_MIN:
            epsilon = EPS_MIN

    # print debug information
    print("----------------------------------episode ", episode)
    print("return=",c_reward)
    print("epsilon=", epsilon)
    print("experience size =", len(experience))
    episode+=1
    state, info = env.reset()
    c_reward = 0
env.close()