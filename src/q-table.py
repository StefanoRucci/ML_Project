import gym
import numpy as np
import random
import matplotlib.pyplot as plt

# setup the environment
env=gym.make("CliffWalking-v0")#,render_mode='human')
env.reset()
env.render()

# observation space - states of the environment (they are 48) 
print("Observation space:", env.observation_space)

# actions: up->0, right->1, down->1, left->3.
print("Action space:", env.action_space)

#Creating a q-table and intialising all values as 0
state_space_size = env.observation_space.n
action_space_size = env.action_space.n

q_table = np.zeros((state_space_size,action_space_size))
print(q_table)

#Number of episodes
num_episodes = 500
#Max number of steps per episode
max_steps_per_episode = 100

learning_rate = 0.1
discount_rate = 0.99

#Greedy strategy
exploration_rate = 1
max_exploration_rate = 1
min_exploration_rate = 0.01
exploration_decay_rate = 0.01

# Lista per contenere le ricompense cumulative di tutti gli episodi
rewards_all_episodes = []

# Lista per contenere la media delle ricompense ogni N episodi
avg_rewards_per_thousand_episodes = []

# Lista per contenere l'esplorazione ogni episodio
exploration_rates = []

#Q-Learning Algorithm
for episode in range(num_episodes): #Contains that happens in an episode
    state = env.reset()
    state=state[0]
    print("NUOVO EPISODIO\nEpisodio numero: ", episode)
    
    done = False #Tells whether episode is finished
    rewards_current_episode = 0 # start with reward 0 each episode

    for step in range(max_steps_per_episode): #moves for each episode
        
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
    #print("Expolarion rate:", exploration_rate)
    print("Episode reward:", rewards_current_episode)
    print('-' * 40)

    rewards_all_episodes.append(rewards_current_episode)

    # Aggiungi la media delle ricompense ogni N episodi alla lista
    if episode % 100 == 0:
        avg_reward = np.mean(rewards_all_episodes[-100:])
        avg_rewards_per_thousand_episodes.append(avg_reward)

        # Aggiungi l'esplorazione corrente alla lista
        exploration_rates.append(exploration_rate)

# Calculate and print the average reward for all episodes
rewards_per_thousand_episodes = np.split(np.array(rewards_all_episodes),num_episodes/500)
count = 500

print("********Average reward per all episodes********\n")
for r in rewards_per_thousand_episodes:
    print(count, ": ", str(sum(r/500)))
    count += 500#Print the updates Q-Table
#print("\n\n*******Q-Table*******\n")
#print(q_table)

# Plot delle ricompense cumulative per episodio
plt.plot(rewards_all_episodes)
plt.xlabel('Episodio')
plt.ylabel('Ricompensa Cumulativa')
plt.title('Ricompensa Cumulativa per Episodio')
plt.show()

# Plot della media delle ricompense ogni N episodi
plt.plot(avg_rewards_per_thousand_episodes)
plt.xlabel('Episodio (ogni 100 episodi)')
plt.ylabel('Media Ricompense')
plt.title('Media Ricompense per 100 Episodi')
plt.show()

# Plot dell'esplorazione durante gli episodi
plt.plot(exploration_rates)
plt.xlabel('Episodio')
plt.ylabel('Tasso di Esplorazione')
plt.title('Tasso di Esplorazione per Episodio')
plt.show()

#close the environment
env.close()