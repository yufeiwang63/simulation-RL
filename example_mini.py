# run different simulations with different integrators
import acrobot
import matplotlib.pyplot as plt
import numpy as np
import os
from soft_actor_critic.utils import save_numpy_as_gif

n = 500

acrob = acrobot.AcrobotEnv()
actions = [acrob.action_space.sample() for _ in range(n)]
statess = {}

for intgt_name in acrobot.AcrobotEnv.INTEGRATOR_NAMES:
    print(intgt_name)
    acrob = acrobot.AcrobotEnv(integrator_name = intgt_name)
    acrob.reset()
    
    states = []
    rgb_arrays = []
    rgb_arrays.append(acrob.render(mode='rgb_array'))
    for action in actions:
        state = acrob.step(action)[0]
        states.append(state)
        rgb_arrays.append(acrob.render(mode='rgb_array'))
    statess[intgt_name] = states
    print(len(statess))

    save_numpy_as_gif(np.asarray(rgb_arrays), os.path.join(os.getcwd(), 'visual_mini_{}.gif'.format(intgt_name)))


for intgt_name, states in statess.items():
    plt.plot(np.asarray(states).mean(1), label=intgt_name)
plt.legend()
plt.show()