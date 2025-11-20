# src/rl/rl_model.py
# Fixed Actor-Critic: recompute log-probs at update time (robust, correct gradients).
# Includes optional entropy regularization and gradient clipping.

# inside ActorCritic class

import os
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
        return self.net(x)  # raw logits (no softmax here)

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
        return self.net(x).squeeze(-1)  # scalar value per batch item

class ActorCritic:
    def __init__(self, input_dim, num_actions,
                 actor_lr=1e-4, critic_lr=5e-4, gamma=0.99,
                 device='cpu', entropy_coef=0.001, max_grad_norm=0.5):
        self.device = torch.device(device)
        self.actor = Actor(input_dim, 64, num_actions).to(self.device)
        self.critic = Critic(input_dim, 64).to(self.device)
        self.actor_optim = optim.Adam(self.actor.parameters(), lr=actor_lr)
        self.critic_optim = optim.Adam(self.critic.parameters(), lr=critic_lr)
        self.gamma = gamma
        self.entropy_coef = entropy_coef
        self.max_grad_norm = max_grad_norm
        self.num_actions = num_actions

    def policy(self, state_np):
        """
        Sample an action from the current policy given a single state (numpy array).
        Returns (probs_np, action_idx_int).
        NOTE: we do NOT return a logp to be stored; log-prob is recomputed at update time.
        """
        state = torch.tensor(state_np, dtype=torch.float32, device=self.device).unsqueeze(0)  # (1, state_dim)
        with torch.no_grad():
            logits = self.actor(state)                   # (1, num_actions)
            probs = torch.softmax(logits, dim=-1).squeeze(0)  # (num_actions,)
            m = torch.distributions.Categorical(probs)
            a = m.sample()
        return probs.detach().cpu().numpy(), int(a.item())

    def value(self, state_np):
        state = torch.tensor(state_np, dtype=torch.float32, device=self.device).unsqueeze(0)
        with torch.no_grad():
            v = self.critic(state)
        return float(v.detach().cpu().item())

    def update(self, transitions):
        """
        transitions: list of (state_np, action_idx, reward, next_state_np, done, logp_placeholder)
        Note: the last element (we accept it for backward compatibility) is ignored here.
        We'll recompute log-probs from the current actor using the batched states/actions.

        Performs a simple 1-step bootstrap actor-critic update:
         - Critic target: r + gamma * V(next) * (1 - done)
         - Critic loss: MSE(V(s), target)
         - Actor loss: -logpi(a|s) * advantage  - entropy_coef * entropy(probs)
        """
        if not transitions:
            return

        # Batch data
        states = torch.tensor(np.vstack([t[0] for t in transitions]), dtype=torch.float32, device=self.device)
        # actions = torch.tensor([t[1] for t in transitions], dtype=torch.long, device=self.device)
        # t[1] is now the probability vector array
        actions_probs = torch.tensor(
            np.vstack([t[1] for t in transitions]),
            dtype=torch.float32,
            device=self.device
        )  # shape (batch, num_actions)
        rewards = torch.tensor([t[2] for t in transitions], dtype=torch.float32, device=self.device)
        next_states = torch.tensor(np.vstack([t[3] for t in transitions]), dtype=torch.float32, device=self.device)
        dones = torch.tensor([t[4] for t in transitions], dtype=torch.float32, device=self.device)

        # Critic targets
        with torch.no_grad():
            v_next = self.critic(next_states).squeeze(-1)  # (batch,)
            target = rewards + self.gamma * v_next * (1.0 - dones)

        vals = self.critic(states).squeeze(-1)  # (batch,)
        critic_loss = nn.MSELoss()(vals, target)

        # Compute current policy log-probs and entropy
        logits = self.actor(states)                    # (batch, num_actions)
        probs_pred = torch.softmax(logits, dim=-1)          # (batch, num_actions)
        # m = torch.distributions.Categorical(probs)
        # logps = m.log_prob(actions)                    # (batch,)
        entropy = -(probs_pred * torch.log(probs_pred + 1e-8)).sum(dim=1).mean()
        log_probs_pred = torch.log(probs_pred + 1e-8)       # (batch, num_actions)
        logps = (actions_probs * log_probs_pred).sum(dim=1)  # batch of log-prob expectations
        # entropy = m.entropy().mean()

        # Advantage
        advantage = (target - vals).detach()           # (batch,)

        # Actor loss
        actor_loss = -(logps * advantage).mean() - self.entropy_coef * entropy

        # Optimize critic
        self.critic_optim.zero_grad()
        critic_loss.backward()
        # optional grad clip for critic
        torch.nn.utils.clip_grad_norm_(self.critic.parameters(), self.max_grad_norm)
        self.critic_optim.step()

        # Optimize actor
        self.actor_optim.zero_grad()
        actor_loss.backward()
        torch.nn.utils.clip_grad_norm_(self.actor.parameters(), self.max_grad_norm)
        self.actor_optim.step()

        return float(actor_loss.item()), float(critic_loss.item())
    

    def save_checkpoint(self, path, optimizer_state=True, extra=None):
        """
        Save actor+critic and optimizer state plus optional extra metadata.
        path: final file path (atomic save will be used)
        optimizer_state: whether to store optimizers' state_dicts
        extra: dict of extra python-serializable metadata (steps, config, rng states, etc.)
        """
        tmp = path + ".tmp"
        checkpoint = {
            'actor_state_dict': self.actor.state_dict(),
            'critic_state_dict': self.critic.state_dict(),
            'num_actions': self.num_actions,
        }
        if optimizer_state:
            checkpoint['actor_optim_state_dict'] = self.actor_optim.state_dict()
            checkpoint['critic_optim_state_dict'] = self.critic_optim.state_dict()
        if extra is not None:
            checkpoint['extra'] = extra

        # PyTorch save (atomic by tmp -> replace)
        torch.save(checkpoint, tmp)
        os.replace(tmp, path)

    def load_checkpoint(self, path, map_location='cpu', load_optim=True):
        """
        Load checkpoint saved by save_checkpoint.
        Returns extra dict if present.
        """
        checkpoint = torch.load(path, map_location=map_location)
        self.actor.load_state_dict(checkpoint['actor_state_dict'])
        self.critic.load_state_dict(checkpoint['critic_state_dict'])
        if load_optim and 'actor_optim_state_dict' in checkpoint:
            try:
                self.actor_optim.load_state_dict(checkpoint['actor_optim_state_dict'])
            except Exception:
                # optimizer shape mismatch -> skip
                pass
        if load_optim and 'critic_optim_state_dict' in checkpoint:
            try:
                self.critic_optim.load_state_dict(checkpoint['critic_optim_state_dict'])
            except Exception:
                pass
        return checkpoint.get('extra', None)

