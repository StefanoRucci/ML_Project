import numpy as np
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Activation, Flatten
import gym
import random
from collections import deque
import matplotlib.pyplot as plt

# Creazione dell'ambiente CliffWalking-v0
env = gym.make('CliffWalking-v0')#, render_mode="human")

# Hyperparameters
EXP_MAX_SIZE = 10000
BATCH_SIZE = 32
EPS_MAX = 1
EPS_MIN = 0.01
exploration_decay_rate = 0.005
GAMMA = 0.95
LR = 0.005
experience = deque(maxlen=EXP_MAX_SIZE)
c_reward=0
checkpoint_path = './checkpoints/cp.ckpt' # file to record network configuration


state_space_size = env.observation_space.n #48
action_space_size = env.action_space.n #4
# print(state_space_size, action_space_size)

# Use a NN to Q-function Q(obs,a)
# NN architecture
model = Sequential()
model.add(Flatten(input_shape=(48,)))

model.add(Dense(64))
model.add(Activation('relu'))

model.add(Dense(64))
model.add(Activation('relu'))

model.add(Dense(action_space_size))
model.add(Activation('linear'))

print(model.summary())
#model.compile(optimizer='adam', loss='mse')
model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=LR), loss='mse')

# Lista per contenere le ricompense cumulative di tutti gli episodi
rewards_all_episodes = []

# Lista per contenere l'esplorazione ogni episodio
exploration_rates = []

# initialize environment and get related information and observation
# obs represents current state
#state, info = env.reset(seed=42)

episode = 1 # counting episodes to decrease epsilon
epsilon = EPS_MAX # start with max exploration probability

# Addestramento dell'agente
num_episodes = 300
max_steps_per_episode = 100

for i in range(num_episodes):
    #print(epsilon)
    state, info = env.reset()

    for step in range(max_steps_per_episode):

        rv = random.uniform(0, 1) # pick random int to decide exploration or exploitation

        if rv >= epsilon:
            candidates = {}
            for a in range(4):
                state_one_hot = np.eye(48)[state]
                candidates[a] = model.predict_on_batch(tf.constant(np.expand_dims(state_one_hot, axis=0)))[0]

            action = np.argmax(candidates)

        else:
            action = env.action_space.sample() # pick random action (default)

        #print(action)

        returnValue=env.step(action) 
        next_state = returnValue[0]
        reward = returnValue[1]
        done = returnValue[2]
        info = returnValue[3]

        #if action == 1:  # Se l'azione è muoversi a destra
        #    reward += 1  # Ricompensa positiva per spostarsi a destra
        #elif action == 3:  # Se l'azione è muoversi a sinistra
        #    reward -= 1  # Penalità per spostarsi a sinistra

        #if state == 0:
        #    reward -= 50

        #if state == 9 or state == 21 or state == 33:
        #    reward += 10
            
        #elif state == 10 or state == 22 or state == 34:
        #    reward += 20

        #elif state == 11 or state == 23 or state == 35:
        #    reward += 30

        if state == 47:
            reward += 100
            c_reward += reward
            break

        c_reward += reward # cumulate rewaed (for evaluation only)

        # Find next best action max_a q(s',a), observe s' is obs_next
        candidates_next = {}
        for a in range(4):
            next_state_one_hot = np.eye(48)[next_state].flatten()
            candidates_next[a] = model.predict_on_batch(tf.constant(np.expand_dims(next_state_one_hot, axis=0)))[0]

        act_next = np.argmax(candidates_next)

        #Compute corresponding (predicted) reward
        reward_next = candidates_next[act_next]

        # Penalità quando l'azione successiva è andare a sinistra
        #if act_next == 3:
        #    reward_next -= 1  # Sostituisci con il valore desiderato della penalità

        # Bonus quando l'azione successiva è andare a destra
        #if act_next == 1:
        #    reward_next += 1  # Sostituisci con il valore desiderato del bonus

        #if next_state == 0:
        #    reward_next -= 50

        #if next_state == 9 or next_state == 21 or next_state == 33:
        #    reward_next += 10
            
        #elif next_state == 10 or next_state == 22 or next_state == 34:
        #    reward_next += 20

        #elif next_state == 11 or next_state == 23 or next_state == 35:
        #    reward_next += 30

        #if next_state == 47:
        #    reward_next += 100

        # Record experience (will be used to train network)
        if len(experience)>=EXP_MAX_SIZE:
            experience.popleft() # dequeue oldest item

        #experience.append([*[state, action], reward + GAMMA*reward_next]) # queue new experience item
        experience.append([*[state, action, next_state], reward + GAMMA*reward_next])

        state = next_state # update current state

    if len(experience) >= BATCH_SIZE and episode % 10 == 0:
        # sample batch
        batch = random.sample(experience, BATCH_SIZE)
        #print(batch)
        #for example in batch:
        #    print(np.array(example).shape)


        # prepare data
        # Estrai solo i primi tre elementi da ciascuna riga di batch
        dataset_partial = np.array([item[:3] for item in batch])

        # Estrai solo i valori float32 dalla quarta colonna
        float32_values = np.array([item[3].tolist() for item in batch])

        # Concatena i due array
        dataset = np.column_stack((dataset_partial, float32_values))

        X = np.eye(48)[dataset[:, 0].astype(int)]  # Use correct index for state
        actions = dataset[:, 1].astype(int)  # Correct index for action
        Y = model.predict_on_batch(tf.constant(X))

        for i in range(BATCH_SIZE):
            next_state_one_hot = np.eye(48)[int(dataset[i, 2])].flatten()
            Y[i, actions[i]] = dataset[i, 2] + GAMMA * np.max(model.predict_on_batch(tf.constant(np.expand_dims(next_state_one_hot, axis=0))))


        # train network
        model.fit(X, Y, validation_split=0.2, epochs=1)


        #epsilon = epsilon/100 # reduce epsilon by 1/100
        epsilon = EPS_MIN + (EPS_MAX - EPS_MIN) * np.exp(-exploration_decay_rate*episode)
        if epsilon<=EPS_MIN:
            epsilon = EPS_MIN

    rewards_all_episodes.append(c_reward)
    exploration_rates.append(epsilon)


    # print debug information
    print("----------------------------------episode ", episode)
    print("return=",c_reward)
    print("epsilon=", epsilon)
    print("experience size =", len(experience))
    episode+=1
    state, info = env.reset()
    c_reward = 0

# Plot delle ricompense cumulative per episodio
plt.plot(rewards_all_episodes)
plt.xlabel('Episodio')
plt.ylabel('Ricompensa Cumulativa')
plt.title('Ricompensa Cumulativa per Episodio')
plt.show()

# Plot dell'esplorazione durante gli episodi
plt.plot(exploration_rates)
plt.xlabel('Episodio')
plt.ylabel('Tasso di Esplorazione')
plt.title('Tasso di Esplorazione per Episodio')
plt.show()

# Chiudi l'ambiente
env.close()