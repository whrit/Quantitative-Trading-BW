import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from torch.optim.lr_scheduler import StepLR
import math

class NoisyLinear(nn.Module):
    def __init__(self, in_features, out_features, std_init=0.5):
        super(NoisyLinear, self).__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.std_init = std_init

        self.weight_mu = nn.Parameter(torch.FloatTensor(out_features, in_features))
        self.weight_sigma = nn.Parameter(torch.FloatTensor(out_features, in_features))
        self.register_buffer('weight_epsilon', torch.FloatTensor(out_features, in_features))

        self.bias_mu = nn.Parameter(torch.FloatTensor(out_features))
        self.bias_sigma = nn.Parameter(torch.FloatTensor(out_features))
        self.register_buffer('bias_epsilon', torch.FloatTensor(out_features))

        self.reset_parameters()
        self.reset_noise()

    def reset_parameters(self):
        mu_range = 1 / math.sqrt(self.in_features)
        self.weight_mu.data.uniform_(-mu_range, mu_range)
        self.weight_sigma.data.fill_(self.std_init / math.sqrt(self.in_features))
        self.bias_mu.data.uniform_(-mu_range, mu_range)
        self.bias_sigma.data.fill_(self.std_init / math.sqrt(self.out_features))

    def reset_noise(self):
        epsilon_in = self._scale_noise(self.in_features)
        epsilon_out = self._scale_noise(self.out_features)
        self.weight_epsilon.copy_(epsilon_out.ger(epsilon_in))
        self.bias_epsilon.copy_(self._scale_noise(self.out_features))

    def _scale_noise(self, size):
        x = torch.randn(size)
        x = x.sign().mul(x.abs().sqrt())
        return x

    def forward(self, input):
        if self.training:
            return F.linear(input, self.weight_mu + self.weight_sigma * self.weight_epsilon,
                            self.bias_mu + self.bias_sigma * self.bias_epsilon)
        else:
            return F.linear(input, self.weight_mu, self.bias_mu)

# Use NoisyLinear in Q_Net
class Q_Net(nn.Module):
    def __init__(self, state_dim, hidden_dim, action_dim, device):
        super(Q_Net, self).__init__()
        self.fc1 = NoisyLinear(state_dim, hidden_dim).to(device)
        self.fc2 = NoisyLinear(hidden_dim, hidden_dim).to(device)
        self.fc3 = NoisyLinear(hidden_dim, action_dim).to(device)
        
    def forward(self, x):
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = self.fc3(x)
        return x

    def reset_noise(self):
        self.fc1.reset_noise()
        self.fc2.reset_noise()
        self.fc3.reset_noise()

class Q_Net(nn.Module):
    def __init__(self, state_dim, hidden_dim, action_dim, device):
        super(Q_Net, self).__init__()
        self.fc1 = nn.Linear(state_dim, hidden_dim).to(device)
        self.fc2 = nn.Linear(hidden_dim, hidden_dim).to(device)
        self.fc3 = nn.Linear(hidden_dim, hidden_dim).to(device)
        self.fc4 = nn.Linear(hidden_dim, action_dim).to(device)
        
    def forward(self, x):
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = F.relu(self.fc3(x))
        x = self.fc4(x)
        return x

class DQN_Agent(nn.Module):
    def __init__(self, state_dim, hidden_dim, action_dim, lr, device, gamma, epsilon, target_update):
        super(DQN_Agent, self).__init__()
        self.state_dim = state_dim
        self.action_dim = action_dim
        self.epsilon = epsilon
        self.gamma = gamma
        self.device = device
        self.target_update = target_update

        # Q networks (now using NoisyLinear layers)
        self.Q_Net = Q_Net(state_dim, hidden_dim, action_dim, device)
        self.Target_Q_Net = Q_Net(state_dim, hidden_dim, action_dim, device)
        
        self.Q_optimizer = torch.optim.Adam(self.Q_Net.parameters(), lr=lr)
        self.scheduler = StepLR(self.Q_optimizer, step_size=1000, gamma=0.1)

        self.count = 0

    def take_action(self, state):
        if np.random.random() < self.epsilon:
            action = np.random.randint(self.action_dim)
        else:
            state = torch.tensor(state, dtype=torch.float32).unsqueeze(0).to(self.device)
            q_values = self.Q_Net(state)
            action = q_values.argmax().item()
        return action

    def update(self, experiences, weights):
        states, actions, rewards, next_states, dones = zip(*experiences)
        
        states = torch.tensor(states, dtype=torch.float32).to(self.device)
        actions = torch.tensor(actions, dtype=torch.int64).unsqueeze(-1).to(self.device)
        rewards = torch.tensor(rewards, dtype=torch.float32).unsqueeze(-1).to(self.device)
        next_states = torch.tensor(next_states, dtype=torch.float32).to(self.device)
        dones = torch.tensor(dones, dtype=torch.float32).unsqueeze(-1).to(self.device)
        weights = torch.tensor(weights, dtype=torch.float32).to(self.device)

        # Double DQN
        q_values = self.Q_Net(states).gather(1, actions)
        next_actions = self.Q_Net(next_states).max(1)[1].unsqueeze(-1)
        next_q_values = self.Target_Q_Net(next_states).gather(1, next_actions)
        expected_q_values = rewards + self.gamma * next_q_values * (1 - dones)

        loss = (weights * F.mse_loss(q_values, expected_q_values.detach(), reduction='none')).mean()

        self.Q_optimizer.zero_grad()
        loss.backward()
        self.Q_optimizer.step()
        self.scheduler.step()

        if self.count % self.target_update == 0:
            self.Target_Q_Net.load_state_dict(self.Q_Net.state_dict())
        
        self.count += 1

        return loss