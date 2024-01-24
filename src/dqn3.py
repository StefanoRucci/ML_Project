import gym
import numpy as np
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
from tensorflow.keras.optimizers import Adam
from collections import deque
import random
import matplotlib.pyplot as plt


# Definisci la rete neurale
def build_dqn(input_size, output_size):
    model = Sequential()
    model.add(Dense(64, input_dim=input_size, activation='relu'))  # Aumentato il numero di neuroni
    model.add(Dense(32, activation='relu'))
    model.add(Dense(16, activation='relu'))
    model.add(Dense(output_size, activation='linear'))
    model.compile(optimizer=Adam(learning_rate=0.001), loss='mse')
    return model

# Definisci la classe del buffer di memoria
class ReplayBuffer:
    def __init__(self, buffer_size):
        self.buffer = deque(maxlen=buffer_size)

    def add(self, experience):
        self.buffer.append(experience)

    def sample(self, batch_size):
        return random.sample(self.buffer, batch_size)

# Parametri dell'ambiente
env = gym.make("CliffWalking-v0")#, render_mode="human")
state_size = env.observation_space.n
action_size = env.action_space.n

# Parametri dell'agente DQN
batch_size = 32
buffer_size = 1000
gamma = 0.99
exploration_rate = 1.0
min_exploration_rate = 0.01
exploration_decay_rate = 0.995
target_update_frequency = 10

# Numero totale di episodi
num_episodes = 300
max_steps_per_episode = 50

# Inizializza la rete neurale del modello DQN
model = build_dqn(state_size, action_size)

# Inizializza il target model (usato per il calcolo del target Q durante l'aggiornamento)
target_model = build_dqn(state_size, action_size)
target_model.set_weights(model.get_weights())

# Inizializza il buffer di memoria
memory = ReplayBuffer(buffer_size)

# Lista per contenere le ricompense cumulative di tutti gli episodi
rewards_all_episodes = []

# Lista per contenere l'esplorazione ogni episodio
exploration_rates = []

# ...

# DQN Algorithm
one_hot_state = np.zeros((1, state_size))  # Definisci one_hot_state all'esterno del loop
for episode in range(num_episodes):

    state = env.reset()
    state = state[0]
    
    done = False
    rewards_current_episode = 0

    for step in range(max_steps_per_episode):
        # Esplorazione-esplorazione trade-off
        exploration_rate_threshold = random.uniform(0, 1)

        if exploration_rate_threshold > exploration_rate:
            # One-hot-encode dello stato
            one_hot_state = np.zeros((1, state_size))
            one_hot_state[0, state] = 1
            # Calcola la distribuzione di probabilità delle azioni
            action_probs = model.predict(one_hot_state)[0]
            #print(action_probs)
            # Campiona un'azione dalla distribuzione di probabilità
            action = np.argmax(action_probs)
            #action = np.random.choice(action_size, p=action_probs)
            #print(action)
        else:
            action = env.action_space.sample()

        #Taking action
        returnValue = env.step(action) 
        new_state = returnValue[0]
        reward = returnValue[1]
        done = returnValue[2]
        info = returnValue[3]

        # Aggiungi l'esperienza al buffer di memoria
        memory.add((state, action, reward, new_state, done))

        state = new_state
        rewards_current_episode += reward

        if done:
            break

    # Addestra il modello DQN utilizzando il campionamento casuale dal buffer di memoria
    if len(memory.buffer) > batch_size and episode % 10 == 0 and episode != 0:
        minibatch = memory.sample(batch_size)
        for state, action, reward, next_state, done in minibatch:
            target = reward
            if not done:
                # One-hot-encode del prossimo stato
                one_hot_next_state = np.zeros((1, state_size))
                one_hot_next_state[0, next_state] = 1

                target = reward + gamma * np.amax(target_model.predict(one_hot_next_state))
            target_f = model.predict(one_hot_state)
            target_f[0][action] = target
            model.fit(one_hot_state, target_f, epochs=1, verbose=1)

    # Aggiorna il target model ogni target_update_frequency episodi
    if episode % target_update_frequency == 0:
        target_model.set_weights(model.get_weights())

    # Riduci il tasso di esplorazione
    exploration_rate = max(min_exploration_rate, exploration_rate * exploration_decay_rate)
    exploration_rates.append(exploration_rate)

    rewards_all_episodes.append(rewards_current_episode)

    # Stampa la ricompensa dell'episodio corrente
    print("Episode {}: Total Reward: {}".format(episode, rewards_current_episode))

# ...


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