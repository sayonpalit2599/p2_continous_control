import numpy as np
import random
import copy
from collections import deque, namedtuple
from model import Actor, Critic
import torch
import torch.nn.functional as F
import torch.optim as optim
BUFFER_SIZE = int(1e6)       # Replay buffer size
BATCH_SIZE  = 128            # Mini-batch size
GAMMA       = 0.99           # Discount factor 
TAU         = 1e-3           # Soft-update target parameters
LR_ACTOR    = 1e-3           # Learning Rate of Actor
LR_CRITIC   = 1e-3           # Learning Rate of Critic
WEIGHT_DECAY= 0              # L2 Weight Decay
LEAK_FACTOR = 0.01           # Leak factor of leaky_relu
LEARN_EVERY = 20             # Learning timestep interval
LEARN_NUM   = 10             # Number of learning passes
GRAD_CLIPPING= 1.            # Gradient Clipping

# Ornstein-Uhlenbeck noise
OU_SIGMA = 0.2
OU_THETA = 0.15
EPSILON  = 1.     # for epsilon in noise
EPSILON_DECAY = 1e-6
device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
class Agent():
    """Agent which interacts and learns from environment"""
    def __init__(self, state_size, action_size, random_seed=0):
        """Initialize an Agent
        Params
        =======
            state_size (int): dimensions of each state
            action_size (int): dimensions of each action
            random_seed (int): random seed
        """
        self.state_size = state_size
        self.action_size = action_size
        self.seed = random.seed(random_seed)
        self.epsilon = EPSILON
        
        # Actor Network
        self.actor_local = Actor(state_size, action_size, random_seed, leakiness=LEAK_FACTOR).to(device)
        self.actor_target = Actor(state_size, action_size, random_seed, leakiness=LEAK_FACTOR).to(device)
        self.actor_optimizer = optim.Adam(self.actor_local.parameters(), lr=LR_ACTOR)
        
        # Critic Network    
        self.critic_local = Critic(state_size, action_size, random_seed, leakiness=LEAK_FACTOR).to(device)
        self.critic_target = Critic(state_size, action_size, random_seed, leakiness=LEAK_FACTOR).to(device)
        self.critic_optimizer = optim.Adam(self.critic_local.parameters(), lr=LR_CRITIC, weight_decay=WEIGHT_DECAY)
        
        self.noise = OUNoise(action_size, random_seed)
        
        # Replay memory
        self.memory = ReplayBuffer(action_size, BUFFER_SIZE, BATCH_SIZE, random_seed)
    
    def add_to_memory(self, states, actions, rewards, next_states, dones):
        """Save experience in replay memory"""
        self.memory.add(states, actions, rewards, next_states, dones)
    
    def learn_from_memory(self, timestep):
        """Sample experience tuples from the replay memory every LEARN_EVERY timesteps"""
        if len(self.memory) > BATCH_SIZE and timestep % LEARN_EVERY == 0:
            for _ in range(LEARN_NUM):
                experiences = self.memory.sample()
                self.learn(experiences, GAMMA)
    
    def step(self, states, actions, rewards, next_states, dones, timestep):
        """Save experience in replay memory, and use random sample from buffer to learn."""
        self.add_to_memory(states, actions, rewards, next_states, dones)
        self.learn_from_memory(timestep)
    
    def act(self, state, add_noise=True):
        """Returns actions for given state as per current policy."""
        state = torch.from_numpy(state).float().to(device)
        self.actor_local.eval()
        with torch.no_grad():
            action = self.actor_local(state).cpu().data.numpy()
        self.actor_local.train()
        
        if add_noise:
            action+=self.epsilon * self.noise.sample()
        
        return np.clip(action, -1, 1)
    
    def reset(self):
        """resets the current noise value"""
        self.noise.reset()
        
    def learn(self, experiences, gamma):
        """Update policy and value parameters given batch of experience tuples
            Q_targets = r + gamma * critic_target(next_state, actor_target(next_state))
            where:
                actor_target(state)  -> action
                critic_target(state) -> Q_value
            Params
            =======
                experiences (Tuple[torch.Tensor]): tuple of (s, a, r, s', done) tuples
                gamma (float): discount factor
        """
        
        states, actions, rewards, next_states, dones = experiences
        
        #----------------update critic-------------------------------#
        # Get predicted next-state actions and Q values from target models
        actions_next = self.actor_target(next_states)
        Q_targets_next = self.critic_target(next_states, actions_next)
        
        # Compute Q_targets for current state
        Q_targets = rewards + (gamma * Q_targets_next * (1-dones))
        
        # Compute critic loss
        Q_expected = self.critic_local(states, actions)
        critic_loss = F.mse_loss(Q_expected, Q_targets)
        
        # Minimize loss
        self.critic_optimizer.zero_grad()
        critic_loss.backward()
        
        # Clipping gradients
        if GRAD_CLIPPING>0:
            torch.nn.utils.clip_grad_norm_(self.critic_local.parameters(), GRAD_CLIPPING)
        self.critic_optimizer.step()
        
        #--------------update actor-------------------------------#
        
        # Compute actor loss
        actions_pred = self.actor_local(states)
        actor_loss = -self.critic_local(states, actions_pred).mean()
        
        # Minimize loss
        self.actor_optimizer.zero_grad()
        actor_loss.backward()
        self.actor_optimizer.step()
        
        
        #------------update target networks---------------------#
        self.soft_update(self.critic_local, self.critic_target, TAU)
        self.soft_update(self.actor_local, self.actor_target, TAU)
        
        #-----------update epsilon decay------------------------#
        self.epsilon*=EPSILON_DECAY
        self.noise.reset()
        
    def soft_update(self, local_model, target_model, tau):
        """ Soft updating target model's parameters
            theta_target = tau*theta_local + (1-tau)*theta_target
            
            Params
            =======
                local_model: Pytorch model
                target_model: Pytorch model
                tau: interpolation parameter
        """
        for target_param, local_param in zip(target_model.parameters(), local_model.parameters()):
            target_param.data.copy_(tau*local_param.data + (1-tau)*target_param.data)

class OUNoise:
    """Ornstein-Uhlenbeck noise"""
    
    def __init__(self, size, seed, mu=0., theta=OU_THETA, sigma=OU_SIGMA):
        """Initialize parameters and noise process.
        Params
        ======
            mu: long-running mean
            theta: the speed of mean reversion
            sigma: the volatility parameter
        """
        self.mu = mu * np.ones(size)
        self.theta = theta
        self.sigma = sigma
        self.seed = random.seed(seed)
        self.reset()
    
    def reset(self):
        """Reset the internal state(noise) to mean(mu)"""
        self.state = copy.copy(self.mu)
    
    def sample(self):
        """Update noise and return it as sample"""
        x = self.state
        #dx = self.theta * (self.mu - x) + self.sigma * np.array([random.random() for i in range(len(x))])
        dx = self.theta * (self.mu - x) + self.sigma * np.random.standard_normal(self.size)
        self.state = x + dx
        return self.state
    

class ReplayBuffer:
    """Fixed-size buffer to store experience tuples."""

    def __init__(self, action_size, buffer_size, batch_size, seed):
        """Initialize a ReplayBuffer object.
        Params
        ======
            buffer_size (int): maximum size of buffer
            batch_size (int): size of each training batch
        """
        self.action_size = action_size
        self.memory = deque(maxlen=buffer_size)  # internal memory (deque)
        self.batch_size = batch_size
        self.experience = namedtuple("Experience", field_names=["state", "action", "reward", "next_state", "done"])
        self.seed = random.seed(seed)
    
    def add(self, state, action, reward, next_state, done):
        """Add a new experience to memory."""
        e = self.experience(state, action, reward, next_state, done)
        self.memory.append(e)
    
    def sample(self):
        """Randomly sample a batch of experiences from memory."""
        experiences = random.sample(self.memory, k=self.batch_size)

        states = torch.from_numpy(np.vstack([e.state for e in experiences if e is not None])).float().to(device)
        actions = torch.from_numpy(np.vstack([e.action for e in experiences if e is not None])).float().to(device)
        rewards = torch.from_numpy(np.vstack([e.reward for e in experiences if e is not None])).float().to(device)
        next_states = torch.from_numpy(np.vstack([e.next_state for e in experiences if e is not None])).float().to(device)
        dones = torch.from_numpy(np.vstack([e.done for e in experiences if e is not None]).astype(np.uint8)).float().to(device)

        return (states, actions, rewards, next_states, dones)

    def __len__(self):
        """Return the current size of internal memory."""
        return len(self.memory)
