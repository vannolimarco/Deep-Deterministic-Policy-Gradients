import torch
import torch.autograd
import torch.optim as optim
import torch.nn as nn
from class_ import actor_critic
from class_ import replay_buffer


class DDPGagent:
    """
       Agent ddpg class. it set parameters of DDPG algorithm and starts the learning with relative updates of networks.
    """
    def __init__(self, env, hidden_size=256, actor_learning_rate=1e-4, critic_learning_rate=1e-3, gamma=0.99, tau=1e-2,
                 max_memory_size=50000):

        # Hyperparameters
        self.num_states = env.observation_space.shape[0]
        self.num_actions = env.action_space.shape[0]
        self.gamma = gamma
        self.tau = tau

        # Networks
        self.actor = actor_critic.Actor(self.num_states, hidden_size, self.num_actions,name= 'Actor')
        self.actor_target = actor_critic.Actor(self.num_states, hidden_size, self.num_actions,name= 'TargetActor')
        self.critic = actor_critic.Critic(self.num_states + self.num_actions, hidden_size, self.num_actions,name='Critic')
        self.critic_target = actor_critic.Critic(self.num_states + self.num_actions, hidden_size, self.num_actions,name='TargetCritic')

        # Create target networks as copies of actor-critic networks.Thus,
        # make a copy of the target network parameters and have them slowly track those of the learned networks via “soft updates"
        for target_param, param in zip(self.actor_target.parameters(), self.actor.parameters()):
            target_param.data.copy_(param.data)

        for target_param, param in zip(self.critic_target.parameters(), self.critic.parameters()):
            target_param.data.copy_(param.data)

        # Training
        self.memory = replay_buffer.ReplayBuffer(max_memory_size)
        self.critic_criterion = nn.MSELoss()

        # optimizers
        self.actor_optimizer = optim.Adam(self.actor.parameters(), lr=actor_learning_rate)
        self.critic_optimizer = optim.Adam(self.critic.parameters(), lr=critic_learning_rate)

    # takes state and return action by actor-critic networks.
    def get_action(self, state):
        state = actor_critic.Variable(torch.from_numpy(state).float().unsqueeze(0))
        action = self.actor.forward(state)
        action = action.detach().numpy()[0, 0]
        return action

    # make updates of networks and target networks taking random batch of transiction from replay buffer
    def update(self, batch_size):
        states, actions, rewards, next_states, _ = self.memory.sample(batch_size)
        states = torch.FloatTensor(states)
        actions = torch.FloatTensor(actions)
        rewards = torch.FloatTensor(rewards)
        next_states = torch.FloatTensor(next_states)

        # The Critic loss
        Qvals = self.critic.forward(states, actions)
        next_actions = self.actor_target.forward(next_states)
        next_Q = self.critic_target.forward(next_states, next_actions.detach())
        Qprime = rewards + self.gamma * next_Q
        critic_loss = self.critic_criterion(Qvals, Qprime)

        # The Actor loss. The derivative of the objective function is taken  with respect to the policy parameter.
        policy_loss = -self.critic.forward(states, self.actor.forward(states)).mean()

        # update networks actor and critic:
        #update actor
        self.actor_optimizer.zero_grad()
        policy_loss.backward()
        self.actor_optimizer.step()

        #update ciritc
        self.critic_optimizer.zero_grad()
        critic_loss.backward()
        self.critic_optimizer.step()

        # update target networks with tau parameter (soft update)
        for target_param, param in zip(self.actor_target.parameters(), self.actor.parameters()):
            target_param.data.copy_(param.data * self.tau + target_param.data * (1.0 - self.tau))

        for target_param, param in zip(self.critic_target.parameters(), self.critic.parameters()):
            target_param.data.copy_(param.data * self.tau + target_param.data * (1.0 - self.tau))

    # save models
    def save_models(self):
        self.actor.save_checkpoint()
        self.actor_target.save_checkpoint()
        self.critic.save_checkpoint()
        self.critic_target.save_checkpoint()

    # load models
    def load_models(self):
        self.actor.load_checkpoint()
        self.actor_target.load_checkpoint()
        self.critic.load_checkpoint()
        self.critic_target.load_checkpoint()