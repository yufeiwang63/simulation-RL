"""classic Acrobot task"""
import numpy as np
from numpy import sin, cos, pi
import scipy.optimize
import torch, copy

from gym import core, spaces
from gym.utils import seeding

__copyright__ = "Copyright 2013, RLPy http://acl.mit.edu/RLPy"
__credits__ = [
    "Alborz Geramifard",
    "Robert H. Klein",
    "Christoph Dann",
    "William Dabney",
    "Jonathan P. How",
]
__license__ = "BSD 3-Clause"
__author__ = "Christoph Dann <cdann@cdann.de>"

# SOURCE:
# https://github.com/rlpy/rlpy/blob/master/rlpy/Domains/Acrobot.py

from util import *
from acrobot import wrap

class AcrobotMaxiEnv(core.Env):

    """
    Acrobot with maximal coordinate.
    """

    metadata = {"render.modes": ["human", "rgb_array"], "video.frames_per_second": 15}

    LINK_LENGTH_1 = 1.0  # [m]
    LINK_LENGTH_2 = 1.0  # [m]
    LINK_MASS_1 = 1.0  #: [kg] mass of link 1
    LINK_MASS_2 = 1.0  #: [kg] mass of link 2
    LINK_COM_POS_1 = 0.5  #: [m] position of the center of mass of link 1
    LINK_COM_POS_2 = 0.5  #: [m] position of the center of mass of link 2
    LINK_MOI = 1.0  #: moments of inertia for both links
    real_small_MOI = 0.1 #: moments of inertia along the rod's thin axis, does not affect the 2d dynamics here
    J1 = torch.tensor(np.diag([real_small_MOI,LINK_MOI,LINK_MOI])).float()
    J2 = torch.tensor(np.diag([real_small_MOI,LINK_MOI,LINK_MOI])).float()
    MAX_VEL_1 = 4 * pi
    MAX_VEL_2 = 9 * pi
    INTEGRATOR_NAMES = ["midpoint"]
    _max_episode_steps = 200

    def __init__(self, integrator_name="midpoint", dt=0.2, horizon=200):
        self.viewer = None
        high = np.array(
            [-1.5, -1.5, 0, np.inf, np.inf, np.inf, np.inf, 
            -1.5, -1.5, 0, np.inf, np.inf, np.inf, np.inf,
            np.inf, np.inf, np.inf, np.inf, np.inf, np.inf, 
            np.inf, np.inf, np.inf, np.inf, np.inf, np.inf], dtype=np.float32
        )
        low = -high
        action_high = np.array(
            [1.0] , dtype=np.float32
        )
        action_low = -action_high 
        self.observation_space = spaces.Box(low=low, high=high) #, dtype=np.float32)
        self.action_space = spaces.Box(low=action_low, high=action_high) #, dtype=np.float32)
        self.dt = dt
        self._max_episode_steps = horizon
        self.T = get_T()
        self.H = get_H()
        self.integrator_name = integrator_name
        assert self.integrator_name in self.INTEGRATOR_NAMES
      
        self.reset()
        self.seed()

    def seed(self, seed=None):
        self.np_random, seed = seeding.np_random(seed)
        return [seed]

    def reset(self):
        init_state = torch.zeros(14)
        # zac's initial
        # init_state[3] = 1
        # init_state[10] = 1
        # init_state[7] = 1

        # our initials, on negative y axis
        init_state[10:] = torch.tensor([0.7071068, 0, 0, -0.7071068])
        init_state[3:7] = torch.tensor([0.7071068, 0, 0, -0.7071068])
        init_state[1] = -self.LINK_LENGTH_1*0.5
        init_state[8] = -self.LINK_LENGTH_1-0.5*self.LINK_LENGTH_2
        self.state = init_state
        self.prev_state = init_state
        self.prev_a = 0
        self.t = 0
        return self._get_ob()

    def get_rQ(self, q):
        return q[:3], q[3:7], q[7:10], q[10:]

    def DEL(self, q1,q2,q3,lamb,F1,F2):
        """
        q1/q2/q3: (14,) tensor
        lamb: (8,) tensor
        F1/F2: (12,) tensor
        return (12,) tensor
        """

        H = self.H
        T = self.T
        m1 = self.LINK_MASS_1
        m2 = self.LINK_MASS_2
        J1 = self.J1
        J2 = self.J2
        h = self.dt

        r1_1, Q1_1, r2_1, Q2_1 = self.get_rQ(q1)
        r1_2, Q1_2, r2_2, Q2_2 = self.get_rQ(q2)
        r1_3, Q1_3, r2_3, Q2_3 = self.get_rQ(q3)
         
        first_row = (1./h)*m1*(r1_2-r1_1) - (1./h)*m1*(r1_3-r1_2) #3,
        second_row = (2./h)*G(Q1_2).T@L(Q1_1)@H@J1@H.T@L(Q1_1).T@Q1_2 \
            + (2./h)*G(Q1_2).T@T@R(Q1_3).T@H@J1@H.T@L(Q1_2).T@Q1_3
        third_row = (1./h)*m2*(r2_2-r2_1) -(1./h)*m2*(r2_3-r2_2)
        forth_row = (2.0/h)*G(Q2_2).T@L(Q2_1)@H@J2@H.T@L(Q2_1).T@Q2_2 \
                    + (2.0/h)*G(Q2_2).T@T@R(Q2_3).T@H@J2@H.T@L(Q2_2).T@Q2_3
        cat = torch.cat((first_row, second_row, third_row, forth_row)) # 12
        force_terms = (h/2.0)*F1 + (h/2.0)*F2 + h*self.Dc(q2).T@lamb # 12
        stacked_return = cat+force_terms # 12
        return stacked_return

    def dq3DEL(self, q_1,q_2,q_3,lamb,F1,F2):
        """
        return (12,12) tensor
        """
        return torch.autograd.functional.jacobian(lambda dq : self.DEL(q_1,q_2,dq,lamb,F1,F2), q_3)@G_bar(q_3)    
    
    # defines the kinematic constraints
    def Dc(self,q):
        """
        q : (14,) tensor
        return: (8,12) tensor
        """
        return torch.autograd.functional.jacobian(lambda dq : self.c(dq), q)@G_bar(q) 

    def c(self,q):
        """
        q : (14,) tensor
        return (8,) tensor
        """
        H = self.H
        ell1 = self.LINK_LENGTH_1
        ell2 = self.LINK_LENGTH_2
        r1, Q1, r2, Q2 = self.get_rQ(q)
        x = torch.Tensor([0.5*ell1, 0, 0])
        y = torch.Tensor([-0.5*ell2, 0, 0])
        z = torch.Tensor([[0, 1, 0, 0], [0, 0, 1, 0]])
        c1 = r1 - H.T@R(Q1).T@L(Q1)@H@x
        c2 = r1 + H.T@R(Q1).T@L(Q1)@H@x - r2 - H.T@R(Q2).T@L(Q2)@H@y
        c3 =  z@L(Q1).T@Q2
        cq = torch.cat([c1,c2,c3]) # (8,)
        return cq

    def maximal_coord_nextstate(self,s1,s2,F1,F2):
        """
        Args:
            s1: (14,) tensor
            s2: (14,) tensor
            F1: (12,) tensor
            F2: (12,) tensor
        Returns:
            s3: (14,) tensor
        """
        h = self.dt
        s3 = copy.deepcopy(s2)
        lamb = torch.zeros(8)
        del_result = self.DEL(s1,s2,s3,lamb,F1,F2) # (12,)
        constraint = self.c(s3) # (8,) 
        e = torch.cat((del_result,constraint)) #(20,)
        while torch.max(torch.abs(e)).item() > 1e-5:
            D = self.dq3DEL(s1,s2,s3,lamb,F1,F2) #12,12
            C2 = self.Dc(s2) #8,12
            C3 = self.Dc(s3) #8,12
            Amat_row1 = torch.cat((D, self.dt*C2.T),dim=1) #12,20
            Amat_row2 = torch.cat((C3,torch.zeros(8,8)),dim=1) #8,20
            Amat = torch.cat((Amat_row1,Amat_row2),dim=0) #20,20
            delta = -torch.inverse(Amat)@e #20,20,20,1 ->20,1

            # update s3
            s3[:3] = s3[:3]+delta[:3]
            row1_link1 = torch.tensor(torch.sqrt(1-delta[3:6].T@delta[3:6])).unsqueeze(0) #1
            rot_mat_link1 = torch.cat((row1_link1,delta[3:6])) #4 
            s3[3:7] = L(s3[3:7])@rot_mat_link1 #4
            
            s3[7:10] = s3[7:10] + delta[6:9]
            row1_link2 = torch.tensor(torch.sqrt(1-delta[9:12].T@delta[9:12])).unsqueeze(0) #1
            rot_mat_link2 = torch.cat((row1_link2,delta[9:12])) #4 
            s3[10:14] = L(s3[10:14])@rot_mat_link2

            # update lambda
            lamb = lamb + delta[12:20]

            # udpate e
            del_result = self.DEL(s1,s2,s3,lamb,F1,F2) # (12,)
            constraint = self.c(s3) # (8,) 
            e = torch.cat((del_result,constraint)) #(20,)
        return s1, s2, s3 

    def step(self, a):
        """
        Args:
            a: (1,), torque
        """
        
        q1 = self.prev_state
        q2 = self.state
        prev_a = self.prev_a

        g = 9.8
        F1 = torch.Tensor([0,-g * self.LINK_MASS_1,0,0,0,-prev_a,0,-g * self.LINK_MASS_1,0,0,0,+prev_a]) #(12,)
        F2 = torch.Tensor([0,-g * self.LINK_MASS_2,0,0,0,-a,0,-g * self.LINK_MASS_2,0,0,0,+a]) #(12,)

        prev_s, cur_s , ns = self.maximal_coord_nextstate(q1, q2, F1, F2) #(14,)

        self.state = ns.detach().clone()
        self.prev_state = cur_s.detach().clone()
        self.prev_a = a
        self.t += 1

        # compute reward based on maximal coordinate
        y = torch.Tensor([0.5*self.LINK_LENGTH_2, 0, 0])
        _, _, r2, Q2 = self.get_rQ(self.state.detach())
        H = self.H
        tip = r2 + H.T@R(Q2).T@L(Q2)@H@y
        reward_max = 1 if tip[1].item() > 1 else -1

        s = self.convert_max2min(self.state).data.numpy()
        if -cos(s[0]) - cos(s[1] + s[0]) > 1.0:
            reward = 1
        else:
            reward = -1

        terminal = self.t >= self._max_episode_steps
        return (self._get_ob(), reward, terminal, {})

    # convert max coordinates to min coordinates used by acrobot before
    def convert_max2min(self,maxq):
        """
        Args:
            maxq:(14,) [r1,Q1,r2,Q2]
        Returns:
            minq(2,) [theta1,theta2]
        """
        H = self.H
        ell1 = self.LINK_LENGTH_1
        minq = torch.zeros(2)
        r1, Q1, r2, Q2 = self.get_rQ(maxq)
        x1, y1, x2, y2 = r1[0], r1[1], r2[0], r2[1]

        theta1 = get_theta_from_negative_y_ccw(x1,y1)
        link1_tip_pos = r1 + H.T@R(Q1).T@L(Q1)@H@torch.Tensor([0.5*ell1, 0, 0])
        r2_new = r2 - link1_tip_pos # as if the origin is at the joint for link2
        x2_new = r2_new[0]
        y2_new = r2_new[1]
        
        theta2 = get_theta_from_negative_y_ccw(x2_new,y2_new) - theta1
        theta1 = wrap(theta1, -np.pi, np.pi)
        theta2 = wrap(theta2, -np.pi, np.pi)
        minq = torch.cat((torch.tensor([theta1]),torch.tensor([theta2])))
        return minq

    def _get_ob(self):
        s = self.state.detach()
        prev_s = self.prev_state.detach()
        r1_1, Q1_1, r2_1, Q2_1 = self.get_rQ(prev_s)
        r1_2, Q1_2, r2_2, Q2_2 = self.get_rQ(s)
        v1 = (r1_2 - r1_1) / self.dt
        v2 = (r2_2 - r2_1) / self.dt
        w1 = 2.0 / self.dt * self.H.T @ L(Q1_1) @ Q1_2
        w2 = 2.0 / self.dt * self.H.T @ L(Q2_1) @ Q2_2
        return np.hstack([s.data.numpy(), v1.data.numpy(), w1.data.numpy(), v2.data.numpy(), w2.data.numpy()])

    def render(self, mode="human"):
        from gym.envs.classic_control import rendering

        s = self.convert_max2min(self.state) #[theta1, theta2]

        if self.viewer is None:
            self.viewer = rendering.Viewer(500, 500)
            bound = self.LINK_LENGTH_1 + self.LINK_LENGTH_2 + 0.2  # 2.2 for default
            self.viewer.set_bounds(-bound, bound, -bound, bound)

        if s is None:
            return None

        p1 = [-self.LINK_LENGTH_1 * cos(s[0]), self.LINK_LENGTH_1 * sin(s[0])]

        p2 = [
            p1[0] - self.LINK_LENGTH_2 * cos(s[0] + s[1]),
            p1[1] + self.LINK_LENGTH_2 * sin(s[0] + s[1]),
        ]

        xys = np.array([[0, 0], p1, p2])[:, ::-1]
        thetas = [s[0] - pi / 2, s[0] + s[1] - pi / 2]
        link_lengths = [self.LINK_LENGTH_1, self.LINK_LENGTH_2]

        self.viewer.draw_line((-2.2, 1), (2.2, 1))
        for ((x, y), th, llen) in zip(xys, thetas, link_lengths):
            l, r, t, b = 0, llen, 0.1, -0.1
            jtransform = rendering.Transform(rotation=th, translation=(x, y))
            link = self.viewer.draw_polygon([(l, b), (l, t), (r, t), (r, b)])
            link.add_attr(jtransform)
            link.set_color(0, 0.8, 0.8)
            circ = self.viewer.draw_circle(0.1)
            circ.set_color(0.8, 0.8, 0)
            circ.add_attr(jtransform)

        return self.viewer.render(return_rgb_array=mode == "rgb_array")
        
    def close(self):
        if self.viewer:
            self.viewer.close()
            self.viewer = None
