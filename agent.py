import numpy as np
import random
import copy
from collections import namedtuple, deque

import model

import torch
import torch.nn.functional as F
import torch.optim as optim

import RAdam

BUFFER_SIZE = int(1e6)  # replay buffer size
BATCH_SIZE = 128        # minibatch size
GAMMA = 0.99            # discount factor
TAU = 1e-3              # for soft update of target parameters
LR_ACTOR = 0.001         # learning rate of the actor
LR_CRITIC = 0.001        # learning rate of the critic
WEIGHT_DECAY = 0        # L2 weight decay
UPDATE_EVERY = 1        # learning timestep interval
UPDATE_NUM = 2          # number of learning passes
RANDOM_SEED = 0         # Random Seed
NOISE_START = 1.0       # Noise value at start of training
T_NOISE_STOP = 1000     # t step where noise stops
NOISE_DECAY = 0.9999    # Noise decay every t step


device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
print(device)

class MADDPG:
    
    def __init__(self, action_size=2, state_size=24, seed=RANDOM_SEED, num_agents=2, buffer_size=BUFFER_SIZE, batch_size=BATCH_SIZE, gamma=GAMMA, update_every=UPDATE_EVERY, update_num=UPDATE_NUM, noise_start=NOISE_START, noise_decay=NOISE_DECAY, t_noise_stop=T_NOISE_STOP):
        
        self.action_size = action_size
        self.state_size = state_size
        self.buffer_size = buffer_size
        self.batch_size = batch_size
        self.update_every = update_every
        self.update_num = update_num
        self.gamma = gamma
        self.num_agents = num_agents
        self.noise_weight = noise_start
        self.noise_decay = noise_decay
        self.t_step = 0
        self.noise_on = True
        self.t_noise_stop = t_noise_stop

        # Shared Replay Memory
        self.memory = ReplayBuffer(buffer_size, batch_size, random_seed=seed)
        models = [model.AgentModels(num_agents=2) for i in range(num_agents)]
        self.agents = [DDPGAgent(num_agent, models[num_agent]) for num_agent in range(num_agents)]
        
        
        
    def step(self, all_states, all_actions, all_rewards, all_next_states, all_dones):
        # 2x24 > 1x48
        all_states = all_states.reshape(1, -1) 
        all_next_states = all_next_states.reshape(1, -1)
        
        # Add to memory
        self.memory.add(all_states, all_actions, all_rewards, all_next_states, all_dones)
        
        if self.t_step > self.t_noise_stop:
            self.noise_on = False
        
        self.t_step += 1
        
        if self.t_step % self.update_every == 0:
            if len(self.memory) > self.batch_size:
                self.learn()
    
    def act(self, all_states, add_noise=True):
        all_actions = []
        for agent, state in zip(self.agents, all_states):
            action = agent.act(state, noise_weight=self.noise_weight, add_noise=self.noise_on)
            self.noise_weight *= self.noise_decay
            all_actions.append(action)
        return np.array(all_actions).reshape(1, -1)
    
    def learn(self):
        for update_n in range(self.update_num):
            all_actions_next, all_actions = [], []
            experiences = [self.memory.sample() for agent_n in range(self.num_agents)]
            for agent_n, agent in enumerate(self.agents):
                states, _, _, next_states, _ = experiences[agent_n]
                agent_id = torch.tensor([agent_n]).to(device)
                state = states.reshape(-1, self.action_size, self.state_size).index_select(1, agent_id).squeeze(1)
                action = agent.actor_local(state)
                all_actions.append(action)
                next_state = next_states.reshape(-1, self.action_size, self.state_size).index_select(1, agent_id).squeeze(1)
                next_action = agent.actor_target(next_state)
                all_actions_next.append(next_action)
            for agent_n, agent in enumerate(self.agents):
                agent.learn(agent_n, experiences[agent_n], self.gamma, all_actions_next, all_actions)
    
    def save_checkpoint(self):
        for i, agent in enumerate(self.agents):
            torch.save(agent.actor_local.state_dict(), f'agent_{i}_a.pth')
            torch.save(agent.critic_local.state_dict(), f'agent_{i}_c.pth')
            

class DDPGAgent():
    """Interacts with and learns from the environment."""

    def __init__(self, agent_num, model, action_size=2, random_seed=0, tau=1e-3, lr_actor=LR_ACTOR, lr_critic=LR_CRITIC, weight_decay=WEIGHT_DECAY):
        """Initialize an Agent object.
        Params
        ======
            state_size (int): dimension of each state
            action_size (int): dimension of each action
            n_agents (int): number of agents
            random_seed (int): random seed
        """
        self.agent_num = agent_num
        self.action_size = action_size
        self.seed = random.seed(random_seed)
        
        # Actor Network (w/ Target Network)
        self.actor_local = model.actor_local
        self.actor_target = model.actor_target
        self.actor_optimizer = RAdam.RAdam(self.actor_local.parameters(), lr=LR_ACTOR)

        # Critic Network (w/ Target Network)
        self.critic_local = model.critic_local
        self.critic_target = model.critic_target
        self.critic_optimizer = RAdam.RAdam(self.critic_local.parameters(), lr=LR_CRITIC, weight_decay=WEIGHT_DECAY)
        
        self.soft_update(self.critic_local, self.critic_target, 0)
        self.soft_update(self.actor_local, self.actor_target, 0)

        # Noise process
        self.noise = OUNoise(action_size, random_seed)
        self.epsilon = 1.0

    def act(self, state, noise_weight=1.0, add_noise=True):
        """Returns actions for given state as per current policy."""
        state = torch.from_numpy(state).float().to(device)
        self.actor_local.eval()
        with torch.no_grad():
            action = self.actor_local(state).cpu().data.numpy()
        self.actor_local.train()
        if add_noise:
            action += self.noise.sample() * noise_weight
        return np.clip(action, -1, 1)

    def reset(self):
        self.noise.reset()

    def learn(self, agent_num, experiences, gamma, all_next_actions, all_actions):

        states, actions, rewards, next_states, dones = experiences

        # ---------------------------- update critic ---------------------------- #
        # Get predicted next-state actions and Q values from target models
        self.critic_optimizer.zero_grad()
        agent_num = torch.tensor([agent_num]).to(device)
        actions_next = torch.cat(all_next_actions, dim=1).to(device)
        
        with torch.no_grad():
            Q_targets_next = self.critic_target(next_states, actions_next)
        # Compute Q targets for current states (y_i)
        Q_targets = rewards.index_select(1, agent_num) + (gamma * Q_targets_next * (1 - dones.index_select(1, agent_num)))
        # Compute critic loss
        Q_expected = self.critic_local(states, actions)
        critic_loss = F.mse_loss(Q_expected, Q_targets.detach())
        # Minimize the loss
        critic_loss.backward()
        # Normalize the gradients 
        torch.nn.utils.clip_grad_norm_(self.critic_local.parameters(), 1)
        self.critic_optimizer.step()

        # ---------------------------- update actor ---------------------------- #
        # Compute actor loss
        self.actor_optimizer.zero_grad()
        actions_pred = [actions if i == self.agent_num else actions.detach() for i, actions in enumerate(all_actions)]
        actions_pred = torch.cat(actions_pred, dim=1).to(device)
        actor_loss = -self.critic_local(states, actions_pred).mean()
        # Minimize the loss
        actor_loss.backward()
        self.actor_optimizer.step()

        # ----------------------- update target networks ----------------------- #
        self.soft_update(self.critic_local, self.critic_target, TAU)
        self.soft_update(self.actor_local, self.actor_target, TAU)

    def soft_update(self, local_model, target_model, tau):
        """Soft update model parameters.
        θ_target = τ*θ_local + (1 - τ)*θ_target
        Params
        ======
            local_model: PyTorch model (weights will be copied from)
            target_model: PyTorch model (weights will be copied to)
            tau (float): interpolation parameter
        """
        for target_param, local_param in zip(target_model.parameters(), local_model.parameters()):
            target_param.data.copy_(tau*local_param.data + (1.0-tau)*target_param.data)
            
            
class OUNoise:
    """Ornstein-Uhlenbeck process."""

    def __init__(self, size, seed, mu=0., theta=0.15, sigma=0.2):
        """Initialize parameters and noise process."""
        self.mu = mu * np.ones(size)
        self.theta = theta
        self.sigma = sigma
        self.seed = random.seed(seed)
        self.size = size
        self.reset()
        

    def reset(self):
        """Reset the internal state (= noise) to mean (mu)."""
        self.state = copy.copy(self.mu)

    def sample(self):
        """Update internal state and return it as a noise sample."""
        x = self.state
        dx = self.theta * (self.mu - x) + self.sigma * np.random.randn(self.size)
        self.state = x + dx
        return self.state

class ReplayBuffer:
    """Fixed-size buffer to store experience tuples."""

    def __init__(self, buffer_size, batch_size, random_seed):
        """Initialize a ReplayBuffer object.
        Params
        ======
            buffer_size (int): maximum size of buffer
            batch_size (int): size of each training batch
        """

        self.memory = deque(maxlen=buffer_size)  # internal memory (deque)
        self.batch_size = batch_size
        self.experience = namedtuple("Experience", field_names=["states", "actions", "rewards", "next_states", "dones"])
        self.seed = random.seed(random_seed)

    def add(self, state, action, reward, next_state, done):
        """Add a new experience to memory."""
        e = self.experience(state, action, reward, next_state, done)
        self.memory.append(e)

    def sample(self):
        """Randomly sample a batch of experiences from memory."""
        experiences = random.sample(self.memory, k=self.batch_size)
        states = torch.from_numpy(np.vstack([e.states for e in experiences if e is not None])).float().to(device)
        actions = torch.from_numpy(np.vstack([e.actions for e in experiences if e is not None])).float().to(device)
        rewards = torch.from_numpy(np.vstack([e.rewards for e in experiences if e is not None])).float().to(device)
        next_states = torch.from_numpy(np.vstack([e.next_states for e in experiences if e is not None])).float().to(device)
        dones = torch.from_numpy(np.vstack([e.dones for e in experiences if e is not None]).astype(np.uint8)).float().to(device)
        return (states, actions, rewards, next_states, dones)

    def __len__(self):
        """Return the current size of internal memory."""
        return len(self.memory)