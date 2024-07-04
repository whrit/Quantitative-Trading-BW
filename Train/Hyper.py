import optuna
import torch
import numpy as np
from Data.Stock_data import data
from tradeEnv import portfolio_tradeEnv
from Model.Deep_Q_Network import DQN_Agent
from tqdm import tqdm
import matplotlib.pyplot as plt
import random

def Normalize(state):
    return (state - state.mean()) / (state.std())

def objective(trial):
    # Define hyperparameters to optimize
    lr = trial.suggest_loguniform('lr', 1e-5, 1e-2)
    hidden_dim = trial.suggest_int('hidden_dim', 32, 256)
    batch_size = trial.suggest_int('batch_size', 32, 256)
    gamma = trial.suggest_uniform('gamma', 0.9, 0.99)
    target_update = trial.suggest_int('target_update', 10, 1000)
    epsilon = trial.suggest_uniform('epsilon', 0.01, 0.2)
    epsilon_decay = trial.suggest_uniform('epsilon_decay', 0.95, 0.999)
    replay_buffer_size = trial.suggest_int('replay_buffer_size', 10000, 100000)

    # Create agent with trial hyperparameters
    agent = DQN_Agent(state_dim=150, hidden_dim=hidden_dim, action_dim=3, lr=lr, device="cuda:0",
                      gamma=gamma, epsilon=epsilon, target_update=target_update)

    # Training data
    train_df = data(ticker='SPY', window_length=15, t=2000).train_data()

    # Training environment
    env = portfolio_tradeEnv(day=0, balance=100000, stock=train_df, cost=0.001)

    # Experience replay buffer
    replay_buffer = []

    returns = []
    for episode in range(100):  # Run for 100 episodes
        state = env.reset()
        state = torch.tensor(Normalize(state).values, dtype=torch.float32).reshape(1, -1).to("cuda:0")
        done = False
        episode_return = 0

        while not done:
            action = agent.take_action(state)
            next_state, reward, done, _ = env.step(action)
            next_state = torch.tensor(Normalize(next_state).values, dtype=torch.float32).reshape(1, -1).to("cuda:0")

            replay_buffer.append((state, action, reward, next_state, done))
            if len(replay_buffer) > replay_buffer_size:
                replay_buffer.pop(0)

            state = next_state
            episode_return += reward

            if len(replay_buffer) > batch_size:
                batch = random.sample(replay_buffer, batch_size)
                agent.update(batch)

        returns.append(episode_return)
        agent.epsilon = max(0.01, agent.epsilon * epsilon_decay)

    # Return the mean of the last 10 episode returns as the objective value
    return np.mean(returns[-10:])

# Create a study object and optimize the objective function
study = optuna.create_study(direction='maximize')
study.optimize(objective, n_trials=100)

print('Best trial:')
trial = study.best_trial
print('Value: ', trial.value)
print('Params: ')
for key, value in trial.params.items():
    print('    {}: {}'.format(key, value))

# Plot optimization history
plt.figure(figsize=(10, 6))
optuna.visualization.plot_optimization_history(study)
plt.title('Optimization History')
plt.show()

# Plot parameter importances
plt.figure(figsize=(10, 6))
optuna.visualization.plot_param_importances(study)
plt.title('Parameter Importances')
plt.show()

# Use the best hyperparameters to train a final model
best_params = study.best_params
best_agent = DQN_Agent(state_dim=150, hidden_dim=best_params['hidden_dim'], action_dim=3, 
                       lr=best_params['lr'], device="cuda:0", gamma=best_params['gamma'], 
                       epsilon=best_params['epsilon'], target_update=best_params['target_update'])

# Train the best agent (you might want to train for more episodes here)
train_df = data(ticker='SPY', window_length=15, t=2000).train_data()
env = portfolio_tradeEnv(day=0, balance=100000, stock=train_df, cost=0.001)

returns = []
for episode in tqdm(range(1000), desc='Training Best Agent'):
    state = env.reset()
    state = torch.tensor(Normalize(state).values, dtype=torch.float32).reshape(1, -1).to("cuda:0")
    done = False
    episode_return = 0

    while not done:
        action = best_agent.take_action(state)
        next_state, reward, done, _ = env.step(action)
        next_state = torch.tensor(Normalize(next_state).values, dtype=torch.float32).reshape(1, -1).to("cuda:0")

        best_agent.update([(state, action, reward, next_state, done)])

        state = next_state
        episode_return += reward

    returns.append(episode_return)
    best_agent.epsilon = max(0.01, best_agent.epsilon * best_params['epsilon_decay'])

# Plot training returns
plt.figure(figsize=(10, 6))
plt.plot(returns)
plt.title('Training Returns of Best Agent')
plt.xlabel('Episode')
plt.ylabel('Return')
plt.show()

# Save the best model
torch.save(best_agent.state_dict(), '../Result/best_agent_dqn_SPY.pt')