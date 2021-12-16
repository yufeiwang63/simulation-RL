import yaml
import os
import random
import numpy as np
import torch
from soft_actor_critic.sac import SAC
from replay_memory import ReplayMemory
from acrobot import AcrobotEnv
from soft_actor_critic.utils import save_numpy_as_gif
import argparse

parser = argparse.ArgumentParser()
parser.add_argument('--integrator', type=str, default='rk4')
parser.add_argument('--dt', type=float, default=0.2)
parser.add_argument('--horizon', type=int, default=200)
parser.add_argument('--load_path', type=str, default=None)
args = parser.parse_args()


config = yaml.safe_load(open('./soft_actor_critic/configs/acrobot.yml')) # custom hyperparams
print(config)
print(os.getpid())

integrator = args.integrator
save_path = args.load_path

# Environment
env = AcrobotEnv(integrator, args.dt, args.horizon)
torch.manual_seed(config['seed'])
np.random.seed(config['seed'])
random.seed(config['seed'])
env.seed(config['seed'])
env.action_space.np_random.seed(config['seed'])
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False

# Agent
suffix = '1000000v2'
env_name = 'Acrobot-v1'

print("{} evaluating {} {}".format('=' * 20, save_path, '=' * 20))

agent = SAC(env.observation_space.shape[0], env.action_space, config)
agent.load_model(save_path, env_name, suffix)

# eval
rgb_arrays = []
episodes = 1
avg_reward = 0
actions = []
for ep_id  in range(episodes):
    print("running test episode: ", ep_id)
    state = env.reset()
    if ep_id == 0:
        img = env.render(mode='rgb_array')
        rgb_arrays.append(img)

    episode_reward = 0
    done = False
    t = 0
    while not done:
        action = agent.select_action(state, eval=True)
        actions.append(action)

        next_state, reward, done, _ = env.step(action)
        if ep_id == 0:
            img = env.render(mode='rgb_array')
            rgb_arrays.append(img)

        episode_reward += reward

        t += 1

        state = next_state
    avg_reward += episode_reward
avg_reward /= episodes

print("----------------------------------------")
print("Test Episodes: {}, Avg. Reward: {}".format(episodes, round(avg_reward, 2)))
if config['automatic_entropy_tuning']:
    print("Test Log Alpha: {}".format(agent.log_alpha.item()))
print("----------------------------------------")
print("saving gif to: ", os.path.join(save_path, 'visual-{}-{}-{}.gif'.format(integrator, args.dt, args.horizon)))
np.save("tmp_action.npy", np.asarray(actions))
save_numpy_as_gif(np.asarray(rgb_arrays), os.path.join(save_path, 'visual-{}-{}-{}.gif'.format(integrator, args.dt, args.horizon)))

env.close()
