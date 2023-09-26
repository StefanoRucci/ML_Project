import numpy as np
import tensorflow as tf
import gym
import random
from collections import deque

# Creazione dell'ambiente CliffWalking-v0
env = gym.make('CliffWalking-v0', render_mode="human")

# Definizione della rete neurale
class QNetwork(tf.keras.Model):
    def __init__(self, num_actions):
        super(QNetwork, self).__init__()
        self.dense1 = tf.keras.layers.Dense(64, activation='relu')
        self.dense2 = tf.keras.layers.Dense(64, activation='relu')
        self.output_layer = tf.keras.layers.Dense(num_actions, activation='linear')

    def call(self, state):
        x = self.dense1(state)
        x = self.dense2(x)
        q_values = self.output_layer(x)
        return q_values

# Iperparametri
learning_rate = 0.01
gamma = 0.95
epsilon_initial = 0.5
epsilon_decay = 0.995
epsilon_min = 0.01
batch_size = 32
replay_memory_size = 10000
target_update_frequency = 10

# Creazione della rete principale e della rete target
num_actions = env.action_space.n
q_network_main = QNetwork(num_actions)
q_network_target = QNetwork(num_actions)
#q_network_target.set_weights(q_network_main.get_weights())

# Definizione della funzione di perdita e dell'ottimizzatore
loss_fn = tf.keras.losses.MeanSquaredError()
optimizer = tf.keras.optimizers.legacy.Adam(learning_rate)

# Creazione del replay buffer
replay_buffer = deque(maxlen=replay_memory_size)

# Funzione per selezionare un'azione
def select_action(state, epsilon):
    if np.random.rand() < epsilon:
        return env.action_space.sample()  # Esplorazione casuale
    else:
        q_values = q_network_main.predict(np.array([[state]]))
        return np.argmax(q_values[0])  # Azione massimale secondo Q

# Addestramento dell'agente
num_episodes = 500
max_steps_per_episode = 40
epsilon = epsilon_initial

for episode in range(num_episodes):
    print(epsilon)
    state = env.reset()
    state = state[0]
    done = False
    total_reward = 0

    for step in range(max_steps_per_episode): #moves for each episode
        action = select_action(state, epsilon)
        returnValue=env.step(action) 
        next_state = returnValue[0]
        reward = returnValue[1]
        done = returnValue[2]
        info = returnValue[3]
        #next_state, reward, done, _ = env.step(action)

        replay_buffer.append((state, action, reward, next_state, done))

        state = next_state
        total_reward += reward

        if len(replay_buffer) >= batch_size:
            batch = np.array(random.sample(replay_buffer, batch_size))
            states, actions, rewards, next_states, dones = batch[:, 0], batch[:, 1], batch[:, 2], batch[:, 3], batch[:, 4]

            q_values_next = q_network_target.predict(np.array([next_states]))
            q_values_max = np.max(q_values_next, axis=1)  # Calcola il massimo per ciascun campione
            targets = rewards + gamma * q_values_max * (1 - dones)

            with tf.GradientTape() as tape:
                #print(states)
                q_values = q_network_main(np.expand_dims(states, axis=1))
                selected_action_values = tf.reduce_sum(q_values * tf.one_hot(actions, num_actions), axis=1)
                loss = loss_fn(targets, selected_action_values)

            grads = tape.gradient(loss, q_network_main.trainable_variables)
            optimizer.apply_gradients(zip(grads, q_network_main.trainable_variables))

        if state == 47: #Checking if episode is over
            break

    #if episode % target_update_frequency == 0:
    epsilon = max(epsilon * epsilon_decay, epsilon_min)

    print(f"Episodio {episode + 1}: Ricompensa totale = {total_reward}")

# Utilizzo del modello addestrato
state = env.reset()
done = False
while not done:
    action = np.argmax(q_network_main.predict(np.expand_dims(state, axis=0)))
    next_state, _, done, _ = env.step(action)
    state = next_state
env.close()