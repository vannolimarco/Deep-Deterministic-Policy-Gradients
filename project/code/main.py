import sys
import gym
import numpy as np
import pandas as pd
import argparse
import matplotlib.pyplot as plt
from class_ import ddpg
import utils
from class_ import noise
from class_ import normalization
import gym
from class_ import pathconfig
import mujoco_py

paths = pathconfig.paths()

#
# The learning of testing environment. It performs the DDPG algorithm steps.
#
def train_testing_environment(episodes=100, batch_size=128, env = 'MountainCarContinuous-v0', name_image = 'ContinuousCar-1', mode = 'testing', steps= 500, save_models= False):
    """
    This method takes as parameter:
    :param episodes: the number of episodes for training
    :param batch_size:  the number for batch size
    :param env:  the name of environment in which start the learning of agent  (default 'MountainCarContinuous-v0)
    :param name_image:  the name for saving the plots
    :param mode: the modality (testing or experimental)
    :param steps: the number of steps
    :param save_models:  True if we want to save models or False if we dont want to save models of DDPG algo
    :return: None, start the learning process and then save plots on own directories
    """
    environment = normalization.NormalizedEnvironment(gym.make(env))
    agent = ddpg.DDPGagent(environment)
    noise_factor = noise.OrnsteinUhlenbeckNoise(environment.action_space)
    rewards = []       #rewards
    avg_rewards = []   #the average among rewards
    random_seed = 123  #random seed
    environment.action_space.seed(random_seed)
    mean_rewards = -10
    steps_save_models = 25

    #Info
    print('|-----INFO ABOUT %s ENVIRONMENT-----|' % mode.upper())
    print('|. name-env : {}'.format(env))
    print('|. action-space : {}'.format(environment.action_space.shape[0]))
    print('|. observation-space : {} '.format(environment.observation_space))
    print('|. Reward range: %s' % (str(environment.reward_range)))
    for i in range(len(environment.observation_space.low)):
        print('|. Observation range, dimension %i: (%.3f to %.3f)' %
              (i, environment.observation_space.low[i], environment.observation_space.high[i]))
    print('|-----START TRAINING OF %s WITH %s EPISODES-----|' % (env, episodes))

    # start the learning, application of the algorithm
    for episode in range(episodes):
        state = environment.reset()
        noise_factor.reset()
        episode_reward = 0
        done = False
        for steps in range(steps): # for each step
            while not done:        # while the state is not terminal
                action = agent.get_action(state)                          # get action from actor
                action = noise_factor.get_action(action, steps)           # get action from adding noise,exploration with noise
                new_state, reward, done, _ = environment.step(action)
                agent.memory.push(state, action, reward, new_state, done) # push the transiction in the  replay buffer

                if len(agent.memory) > batch_size: # update the agent, so actor-critic netowrk and target networks
                    agent.update(batch_size)

                state = new_state         # update state variable with new state
                episode_reward += reward  # sum of rewards
                environment.render()      # render of environment
        sys.stdout.write("episode: {}, reward: {}, average_reward: {} \n".format(episode, np.round(episode_reward, decimals=2),
                                                                     str(np.mean(rewards[mean_rewards:]))))
        rewards.append(episode_reward)
        avg_rewards.append(np.mean(rewards[mean_rewards:]))
        # save models
        if save_models:
           if episode % steps_save_models == 0:
              agent.save_models()
    # close environment
    environment.close()

    # plots
    path_plots = paths.MOUNTAIN_CAR_CONTINUOUS + name_image
    utils.plot_episode_reward(path_file=path_plots,avg_rewards=avg_rewards,mean = abs(mean_rewards),rewards=rewards)
    utils.plot_learning_reward_100(scores=rewards, path_file=path_plots, window=100)

#
# The learning of experimental environment with the DDPG algorithm.
#
def train_experimentation_environment(episodes=100, batch_size=128, env = 'Swimmer-v2', name_image = 'Swimmer-1', mode = 'experimentation', steps=500,save_models=False):
    """
        this method takes as parameter:
        :param episodes: the number of episodes for training
        :param batch_size:  the number for batch size
        :param env:  the name of environment in which start the learning of agent (default 'Swimmer-v2)
        :param name_image:  the name for saving the plots
        :param mode: the modality (testing or experimental)
        :param steps: the number of steps
        :return: None, start the learning process and then save plots on own directories
    """
    environment = normalization.NormalizedEnvironment(gym.make(env))
    agent = ddpg.DDPGagent(environment)
    noise_factor = noise.OrnsteinUhlenbeckNoise(environment.action_space)
    rewards = []       # rewards of each episodes
    avg_rewards = []   # the average among rewards
    random_seed = 123  # random seed
    mean_rewards = -10
    steps_save_models = 25
    environment.action_space.seed(random_seed)

    #Info
    print('|-----INFO ABOUT %s ENVIRONMENT ------|' % mode.upper())
    print('|. name-env : {}'.format(env))
    print('|. action-space : {}'.format(environment.action_space.shape[0]))
    print('|. observation-space : {} '.format(environment.observation_space))
    print('|. Reward range: %s' % (str(environment.reward_range)))
    for i in range(len(environment.observation_space.low)):
        print('|. Observation range, dimension %i: (%.3f to %.3f)' %
              (i, environment.observation_space.low[i], environment.observation_space.high[i]))
    print('|-----START TRAINING OF %s WITH %s EPISODES-----|' % (env, episodes))

    # start the learning, application of the algorithm
    for episode in range(episodes):
        state = environment.reset()
        noise_factor.reset()
        episode_reward = 0
        done = False
        for steps in range(steps):  #for each step

            while not done:    # while the state is not terminal
                action = agent.get_action(state)                            # get action from actor
                action = noise_factor.get_action(action, steps)             # get action from adding noise,exploration with noise
                new_state, reward, done, _ = environment.step(action)
                agent.memory.push(state, action, reward, new_state, done)  # push the transiction in the  replay buffer

                if len(agent.memory) > batch_size:  # update the agent, so actor-critic netowrk and target networks
                    agent.update(batch_size)

                state = new_state        # update state variable with new state
                episode_reward += reward # sum of rewards
                environment.render()     # render of environment
        sys.stdout.write(
            "episode: {}, reward: {}, average_reward: {} \n".format(episode, np.round(episode_reward, decimals=2),
                                                                    str(np.mean(rewards[mean_rewards:]))))
        rewards.append(episode_reward)
        avg_rewards.append(np.mean(rewards[mean_rewards:]))
        # save models
        if save_models:
            if episode % steps_save_models == 0:
                agent.save_models()
    # close environment
    environment.close()

    # plots
    path_plots = paths.SWIMMER + name_image
    utils.plot_episode_reward(path_file=path_plots, avg_rewards=avg_rewards, mean = abs(mean_rewards), rewards=rewards)
    utils.plot_learning_reward_100(scores=rewards, path_file=path_plots, window=100)

# main
if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='PyTorch on TORCS with Multi-modal')
    parser.add_argument('--mode', default='experimentation', type=str, help='support option: train/test')
    parser.add_argument('--env',default='MountainCarContinuous-v0',type=str,help='support option: kind of environment from gym (see all with: envs.registry.all()')
    parser.add_argument('--episodes', default=1000, type=str, help='support option: numbers of episodes with which train model')
    parser.add_argument('--batch_size', default=128, type=str, help='support option: the batch size')
    parser.add_argument('--name_image', default='MountainCarContinuous-11', type=str, help='support option: name image saved')
    parser.add_argument('--steps', default=1000, type=str, help='support option: name image saved')
    parser.add_argument('--save_models', default=False, type=str, help='support option: train/test')
    args = parser.parse_args()

    if args.mode == 'testing':
        args.env = 'MountainCarContinuous-v0'
        train_testing_environment(episodes=args.episodes, env=args.env, batch_size=args.batch_size,name_image=args.name_image, mode=args.mode, steps=args.steps, save_models = args.save_models)
    elif args.mode == 'experimentation':
        args.env = 'Swimmer-v2'
        train_experimentation_environment(episodes=args.episodes, env=args.env, batch_size=args.batch_size,name_image=args.name_image, mode=args.mode, steps=args.steps, save_models = args.save_models)
    else:
        raise RuntimeError('undefined mode {}. mode args can be: testing or experimentation'.format(args.mode))