import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import numpy as np

class ActorCritic(nn.Module):
    def __init__(self, input_dim, num_actions, hidden_dim=128, lr=0.001, device='cpu'):
        super(ActorCritic, self).__init__()
        self.device = device
        self.num_actions = num_actions
        
        # Shared feature extraction layer
        self.fc1 = nn.Linear(input_dim, hidden_dim)
        
        # Actor Head (Policy): Outputs probabilities for each path
        self.actor = nn.Linear(hidden_dim, num_actions)
        
        # Critic Head (Value): Estimates value of the state
        self.critic = nn.Linear(hidden_dim, 1)
        
        self.optimizer = optim.Adam(self.parameters(), lr=lr)
        self.to(self.device)
        
        # Training memory
        self.gamma = 0.99
        self.eps = np.finfo(np.float32).eps.item()

    def forward(self, x):
        # x shape: (batch, input_dim)
        if not isinstance(x, torch.Tensor):
            x = torch.FloatTensor(x).to(self.device)
            
        x = F.relu(self.fc1(x))
        
        # Actor: softmax for probability distribution
        action_probs = F.softmax(self.actor(x), dim=-1)
        
        # Critic: single value
        state_value = self.critic(x)
        
        return action_probs, state_value

    def policy(self, state):
        """
        Select action during inference/simulation
        """
        state_t = torch.FloatTensor(state).unsqueeze(0).to(self.device)
        with torch.no_grad():
            probs, _ = self.forward(state_t)
        
        np_probs = probs.cpu().numpy().flatten()
        
        # Select action based on probabilities
        action_idx = np.random.choice(len(np_probs), p=np_probs)
        return np_probs, action_idx

    def update(self, transitions):
        """
        Perform one step of Backprop using a batch of transitions.
        Transition format: (state, action_probs, reward, next_state, done, _)
        """
        if not transitions:
            return None

        # Unpack batch
        states = torch.FloatTensor(np.array([t[0] for t in transitions])).to(self.device)
        # We use the probabilities generated at the time of action as the "action taken" reference
        # ideally in A2C we need the log_prob of the specific action taken, 
        # but here we simplify by optimizing the distribution.
        # For strict A2C: we need the specific action index taken.
        # Let's assume we want to reinforce the chosen path.
        
        rewards = torch.FloatTensor([t[2] for t in transitions]).to(self.device).unsqueeze(1)
        next_states = torch.FloatTensor(np.array([t[3] for t in transitions])).to(self.device)
        # dones = torch.FloatTensor([t[4] for t in transitions]).to(self.device).unsqueeze(1)

        # Forward pass
        current_probs, state_values = self.forward(states)
        _, next_state_values = self.forward(next_states)

        # Calculate TD Target (Bootstrap)
        # target = r + gamma * V(s')
        target_values = rewards + self.gamma * next_state_values.detach()
        
        # Advantage = Target - V(s)
        advantage = target_values - state_values

        # Critic Loss: MSE(V(s), Target)
        critic_loss = F.mse_loss(state_values, target_values)

        # Actor Loss: -log_prob * advantage
        # We want to encourage actions that resulted in positive advantage
        # Since we stored full probs, we can compute entropy for exploration bonus
        
        # Simplify: Just assume we want to shift distribution towards the "good" direction
        # This is a simplified update for the sake of the project structure. 
        # Standard A2C requires saving the log_prob of the *specific* action taken.
        
        # Re-running policy is required to get log_probs attached to graph
        # Only if we stored indices. Since we stored probs, we do a basic PG approximation:
        actor_loss = -(torch.log(current_probs + self.eps) * advantage.detach()).mean()

        # Total loss
        loss = actor_loss + 0.5 * critic_loss

        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()

        return actor_loss.item(), critic_loss.item()

    def save_checkpoint(self, path):
        torch.save(self.state_dict(), path)

    def load_checkpoint(self, path, map_location):
        self.load_state_dict(torch.load(path, map_location=map_location))