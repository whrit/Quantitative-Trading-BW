
import torch
import matplotlib.pyplot as plt
from Data.Stock_data import data
from tradeEnv import portfolio_tradeEnv
from Model.Deep_Q_Network import Q_Net, DQN_Agent
import numpy as np
import tqdm
from Prioritized_replay import PrioritizedReplayBuffer

def Normalize(state):
    # State normalization
    state = (state - state.mean()) / (state.std())
    return state

def DQN_train(episode, ticker, minimum_experience, batch_size):
    # Initialize agent with new parameters
    agent = DQN_Agent(state_dim=150, hidden_dim=64, action_dim=3, lr=0.0005, device="cuda:0", gamma=0.99,
                      epsilon=0.1, target_update=100)
    
    # Initialize Prioritized Experience Replay buffer
    memory = PrioritizedReplayBuffer(capacity=100000, alpha=0.6, beta=0.4)
    
    # Training data
    train_df = data(ticker=ticker, window_length=15, t=2000).train_data()
    
    # Training environment
    env = portfolio_tradeEnv(day=0, balance=100000, stock=train_df, cost=0.001)
    
    returns = []
    
    for ep in tqdm(range(episode), desc='Training Episodes'):
        state = env.reset()
        state = torch.tensor(Normalize(state).values, dtype=torch.float32).reshape(1, -1).to(device="cuda:0")
        done = False
        episode_return = 0
        
        while not done:
            action = agent.take_action(state)
            next_state, reward, done, _ = env.step(action)
            next_state = torch.tensor(Normalize(next_state).values, dtype=torch.float32).reshape(1, -1).to(device="cuda:0")
            
            memory.add(state, action, reward, next_state, done)
            
            state = next_state
            episode_return += reward
            
            if len(memory.buffer) > minimum_experience:
                experiences, indices, weights = memory.sample(batch_size)
                loss = agent.update(experiences, weights)
                
                # Update priorities in buffer
                td_errors = loss.detach().cpu().numpy()
                new_priorities = np.abs(td_errors) + 1e-6  # small constant to avoid zero priority
                memory.update_priorities(indices, new_priorities)
            
            if done:
                returns.append(episode_return)
                
                # Reset noise for NoisyLinear layers
                agent.Q_Net.reset_noise()
                agent.Target_Q_Net.reset_noise()
        
        # Decay epsilon
        agent.epsilon = max(0.01, agent.epsilon * 0.995)
        
        # Log performance every 10 episodes
        if ep % 10 == 0:
            avg_return = np.mean(returns[-10:])
            print(f"Episode {ep}, Avg Return: {avg_return:.2f}, Epsilon: {agent.epsilon:.2f}")
    
    # Save the trained model
    torch.save(agent.state_dict(), f'../Result/agent_dqn_{ticker}.pt')
    
    # Plot returns
    plt.plot(returns)
    plt.title('Training Returns')
    plt.xlabel('Episode')
    plt.ylabel('Return')
    plt.show()

if __name__ == '__main__':
    import time

    # Improvement: Periodically clear the experience buffer and sample a fixed number of experience tuples
    start = time.time()
    DQN_train(episode=100, ticker='SPY', minimum=1500)
    end = time.time()
    print('Training time', end - start)
