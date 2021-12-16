# Simulation Method Choices for Reinforcement Learning

This repo contains the code implementations for a CMU course project (16715 Fall 2021). Using double pendulum as a case study, we stuided how different simulation choices including integrators and coordinate representations would affect downstream reinforcement learning, in terms of learning efficiency and sim-2-real transfer performance.

## File Description
- `./soft_actor_critic/` contains the soft-actor-critic reinforcement learning algorithm
- `acrobot.py` implements minimal coordinate acrobot simulation, with all integrator choices
- `acrobot_maxi.py` implements maximal coordinate acrobot simulation
- `util.py` implements utility functions for maximal coordinate acrobot simulation
- `example_mini.py` and `example_maxi.py` implement extremely simple simulations to smoketest the corresponding simulations. They will also dump a gif visual of the simulation under the current directory.


## Train SAC agent
Our implementation of Soft Actor-Critic (SAC) is based on the popular [public github repo](https://github.com/pranz24/pytorch-soft-actor-critic/tree/d8ba7370e574340e9e0e9dd0276dbd2241ff3fd1).

To train SAC on the Acrobot env, run:
```bash
python soft_actor_critic/main.py {COORDINATE_REPRESENTATION} {INTEGRATOR_NAME}
```
where coordinate_representation can be 'minmal' or 'maximal', and integrator name can be in {'rk4', 'explicit_euler', 'explicit_midpoint', 'implicit_midpoint'}  
An example would be 
```bash
python soft_actor_critic/main.py minimal explicit_euler
```
Trained models will be saved to `models/acrobot/{time-stamp}/`.

## Evaluate a trained sac policy 
To evalaute a trained sac policy, e.g., on a different integrator, run
```bash
python soft_actor_critic/eval_policy.py --load_path {PATH_TO_SAVED_SAC_POLICY} --integrator {INTEGRATOR_NAME} --dt {TIME_STEP} --horizon {TASK_HORIZON}
```
E.g., to load the pretrained sac policy on the minimal coordinate representation with rk4:
```bash
python soft_actor_critic/eval_policy.py --load_path pretrained-models/rk4/ --integrator rk4 --dt 0.02 --horizon 2000
```
Gif visuals of the policy will be saved to the `load_path` directory.


## Pretrained models
We provide pretrained sac policies under `pretrained-models/{INTEGRATOR_NAME}`. To test the pretrained models, run the command in the last section.



