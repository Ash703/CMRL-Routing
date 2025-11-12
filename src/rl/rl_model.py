# src/rl/rl_model.py
# Simple actor-critic (vanilla) for discrete actions (path indices).
# Uses PyTorch. Designed for small online updates from Ryu controller.

import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np

class Actor(nn.Module):
    def __init__(self, input_dim, hidden=64, num_actions=2):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, hidden),
            nn.ReLU(),
            nn.Linear(hidden, hidden),
            nn.ReLU(),
            nn.Linear(hidden, num_actions)
        )

    def forward(self, x):
        return self.net(x)  # raw logits

class Critic(nn.Module):
    def __init__(self, input_dim, hidden=64):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, hidden),
            nn.ReLU(),
            nn.Linear(hidden, hidden),
            nn.ReLU(),
            nn.Linear(hidden, 1)
        )

    def forward(self, x):
        return self.net(x).squeeze(-1)  # scalar value

class ActorCritic:
    def __init__(self, input_dim, num_actions, actor_lr=1e-4, critic_lr=5e-4, gamma=0.99, device='cpu'):
        self.device = torch.device(device)
        self.actor = Actor(input_dim, 64, num_actions).to(self.device)
        self.critic = Critic(input_dim, 64).to(self.device)
        self.actor_optim = optim.Adam(self.actor.parameters(), lr=actor_lr)
        self.critic_optim = optim.Adam(self.critic.parameters(), lr=critic_lr)
        self.gamma = gamma

    def policy(self, state_np):
        """Return (probs np.array, torch log_prob of sampled action, action index)"""
        state = torch.tensor(state_np, dtype=torch.float32, device=self.device).unsqueeze(0)
        logits = self.actor(state)                # (1, num_actions)
        probs = torch.softmax(logits, dim=-1).squeeze(0)  # (num_actions,)
        m = torch.distributions.Categorical(probs)
        a = m.sample()
        logp = m.log_prob(a)
        return probs.detach().cpu().numpy(), int(a.item()), logp

    def value(self, state_np):
        state = torch.tensor(state_np, dtype=torch.float32, device=self.device).unsqueeze(0)
        v = self.critic(state)
        return float(v.detach().cpu().item())

    def update(self, transitions):
        """
        transitions: list of (state_np, action_idx, reward, next_state_np, done, logp_torch)
        We'll do a simple 1-step actor-critic style update for each transition.
        """
        if not transitions:
            return

        # batching - convert everything to tensors
        states = torch.tensor(np.vstack([t[0] for t in transitions]), dtype=torch.float32, device=self.device)
        actions = torch.tensor([t[1] for t in transitions], dtype=torch.long, device=self.device)
        rewards = torch.tensor([t[2] for t in transitions], dtype=torch.float32, device=self.device)
        next_states = torch.tensor(np.vstack([t[3] for t in transitions]), dtype=torch.float32, device=self.device)
        dones = torch.tensor([t[4] for t in transitions], dtype=torch.float32, device=self.device)
        logps = torch.stack([t[5] for t in transitions]).to(self.device)

        # Critic targets: r + gamma * V(next) * (1-done)
        with torch.no_grad():
            v_next = self.critic(next_states).squeeze(-1)
            target = rewards + self.gamma * v_next * (1.0 - dones)

        vals = self.critic(states).squeeze(-1)
        critic_loss = nn.MSELoss()(vals, target)

        # Policy (actor) loss: -logpi * advantage
        advantage = (target - vals).detach()
        # We already have log probabilities stored as tensors in logps
        actor_loss = -(logps * advantage).mean()

        # Optimize critic
        self.critic_optim.zero_grad()
        critic_loss.backward()
        self.critic_optim.step()

        # Optimize actor
        self.actor_optim.zero_grad()
        actor_loss.backward()
        self.actor_optim.step()

        return float(actor_loss.item()), float(critic_loss.item())
