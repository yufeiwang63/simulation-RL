import acrobot_maxi as acrobot
import matplotlib.pyplot as plt
import numpy as np
from tqdm import tqdm
import torch
from soft_actor_critic.utils import save_numpy_as_gif
import os

n = 200

acrob = acrobot.AcrobotMaxiEnv()
actions = np.array([acrob.action_space.sample() for _ in range(n)])
actions[::2] = 1
actions[1::2] = -1
statess = {}

for intgt_name in acrobot.AcrobotMaxiEnv.INTEGRATOR_NAMES:
    print(intgt_name)
    acrob = acrobot.AcrobotMaxiEnv(integrator_name = intgt_name)
    acrob.reset()

    states = []
    rgb_arrays_maxi = []
    rgb_arrays_maxi.append(acrob.render(mode='rgb_array'))
    for action in tqdm(actions):
        # print(states)
        state = acrob.step(action)[0]
        states.append(state)
        rgb_arrays_maxi.append(acrob.render(mode='rgb_array'))

    statess[intgt_name] = np.array(torch.stack(states))

save_numpy_as_gif(np.asarray(rgb_arrays_maxi), os.path.join(os.getcwd(), 'visual_maxi_{}.gif'.format(n)))
for intgt_name, states in statess.items():
    plt.plot(np.asarray(states).mean(1), label=intgt_name)
plt.legend()
plt.show()
