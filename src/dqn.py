import numpy as np
import tensorflow as tf
from tensorflow import keras
from keras.models import Sequential
from keras.layers import Dense, Activation, Flatten
import gym
import random
from collections import deque
import matplotlib.pyplot as plt

# Creazione dell'ambiente CliffWalking-v0, decommentare per visualizzare ambiente
env = gym.make('CliffWalking-v0')#, render_mode="human")

# Iperparametri
EXP_MAX_SIZE=10000 # Max batch size of past experience
BATCH_SIZE=EXP_MAX_SIZE//10 # Training set size
experience = deque([],EXP_MAX_SIZE) # Past experience arranged as a queue
EPS_MAX = 1
EPS_MIN = 0.01
exploration_decay_rate = 0.005
GAMMA = 0.9
LR = 0.01

state_space_size = env.observation_space.n  # numero di stati = 48
action_space_size = env.action_space.n  # numero di azioni = 4

epsilon = EPS_MAX

# Creazione rete neurale
model = Sequential()
model.add(Flatten(input_shape=(48,)))

model.add(Dense(64))
model.add(Activation('relu'))

model.add(Dense(64))
model.add(Activation('relu'))

model.add(Dense(action_space_size))
model.add(Activation('linear'))

print(model.summary())
model.compile(optimizer="sgd", loss='mse')

# Lista per contenere le ricompense cumulative di tutti gli episodi
rewards_all_episodes = []

# Lista per contenere l'esplorazione ogni episodio
exploration_rates = []

# Addestramento dell'agente
num_episodes = 500
max_steps_per_episode = 100

for episode in range(1, num_episodes + 1):
    state, info = env.reset()
    episode_reward = 0

    for step in range(max_steps_per_episode):

        rv = random.uniform(0, 1) # Random int per decidere se exploration o exploitation
        # Scelta dell'azione
        if rv < epsilon:
            action = env.action_space.sample()  # Esplorazione
        else:
            state_one_hot = np.eye(48)[state]
            Q_values = model.predict_on_batch(np.expand_dims(state_one_hot, axis=0))
            action = np.argmax(Q_values)

        # Esecuzione azione scelta
        returnValue=env.step(action) 
        next_state = returnValue[0]
        reward = returnValue[1]
        done = returnValue[2]
        info = returnValue[3]

        # Aggiunta dell'esperienza al buffer di replay
        next_state_one_hot = np.eye(48)[next_state]
        target = reward + GAMMA * np.max(model.predict_on_batch(np.expand_dims(next_state_one_hot, axis=0)))
        
        # Record dell'esperienze fatte (verrà utilizzato per addestrare la rete)
        if len(experience)>=EXP_MAX_SIZE:
            experience.popleft() # elimina item più vecchio
        experience.append((state, action, next_state, reward, done, target))

        episode_reward += reward

        if done:
            break

        state = next_state

    # Aggiornamento epsilon
    epsilon = max(EPS_MIN, EPS_MAX * np.exp(-exploration_decay_rate * episode))

    # Stampa informazioni sull'episodio
    print("-" * 40)
    print("Episode:", episode)
    print("Reward for this episode:", episode_reward)

    rewards_all_episodes.append(episode_reward)
    exploration_rates.append(epsilon)

    # Addestramento batch ogni 10 episodi
    if episode % 10 == 0 and len(experience) >= BATCH_SIZE:
        print("-" * 40)
        print("...Training...")
        
        # Campionamento batch
        batch = random.sample(experience, BATCH_SIZE)

        # Preparazione del batch per l'addestramento
        X_batch = []
        y_batch = []
        for state, action, next_state, reward, done, target in batch:
            state_one_hot = np.eye(48)[state]
            next_state_one_hot = np.eye(48)[next_state]
            Q_values = model.predict_on_batch(np.expand_dims(state_one_hot, axis=0))
            if done:
                Q_values[0][action] = reward
            else:
                #Q(s,a) = r + gamma * max(Q(s',a'))
                Q_values[0][action] = reward + GAMMA * np.max(model.predict_on_batch(np.expand_dims(next_state_one_hot, axis=0)))
            X_batch.append(state_one_hot)
            y_batch.append(Q_values)

        # Addestramento del modello
        loss = model.train_on_batch(np.array(X_batch).squeeze(), np.array(y_batch).squeeze())

        print("Batch Loss:", loss)
        print("Batch training completed.")
        print("-" * 40)

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
