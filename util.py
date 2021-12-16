import torch
import numpy as np

def get_H():
    return torch.cat((torch.zeros((1,3)), torch.eye(3)), dim=0) # 4,3

def get_T():
    return torch.diag(torch.Tensor([1,-1,-1,-1])) # 4,4

def hat(w):
    """
    w: (3,) tensor
    return (3,3) tensor
    """
    ret = torch.cat([
        torch.cat([torch.tensor([0]).view(1, 1), -w[2].view(1, 1), w[1].view(1, 1)], dim=1), 
        torch.cat([w[2].view(1, 1), torch.tensor([0]).view(1, 1), -w[0].view(1, 1)], dim=1), 
        torch.cat([-w[1].view(1, 1), w[0].view(1, 1), torch.tensor([0]).view(1, 1)], dim=1)
        ], dim=0)
    return ret
    
def L(Q): 
    """
    Q: (4,) tensor
    return (4,4) tensor
    """
    row1 = torch.cat((Q[0:1],-Q[1:4].T)).unsqueeze(0) # 1,4
    block_21 = Q[1:4].unsqueeze(-1) # 3,1
    block_22 = Q[0:1]*torch.eye(3)+hat(Q[1:4]) # 3,3
    row24 = torch.cat((block_21,block_22),dim=1) # 3,4
    Lmat = torch.cat((row1,row24),dim=0) # 4,4
    return Lmat

def R(Q):
    """
    Q: (4,) tensor
    return (4,4) tensor
    """
    row1 = torch.cat((Q[0:1],-Q[1:4].T)).unsqueeze(0) # 1,4
    block_21 = Q[1:4].unsqueeze(-1) # 3,1
    block_22 = Q[0:1]*torch.eye(3)-hat(Q[1:4]) # 3,3
    row24 = torch.cat((block_21,block_22),dim=1) # 3,4
    Rmat = torch.cat((row1,row24),dim=0) # 4,4
    return Rmat
    
def G(Q):
    """
    Q: (4,) tensor
    return (4,3) tensor
    """
    return L(Q)@get_H()

def G_bar(q):
    """
    q: (14,) tensor
    return (14,12) tensor
    """
    Q1 = q[3:7]
    Q2 = q[10:14]
    r1 = torch.cat((torch.eye(3), torch.zeros((3,9),requires_grad=True)),dim=1) # 3,12
    r2 = torch.cat((torch.zeros((4,3)), G(Q1), torch.zeros((4,6))),dim=1) # 4,12
    r3 = torch.cat((torch.zeros((3,6)), torch.eye(3), torch.zeros((3,3))),dim=1) #3,12
    r4 = torch.cat((torch.zeros((4,9)), G(Q2)),dim=1) # 4,12
    return torch.cat((r1,r2,r3,r4),dim=0) # 14,12

def get_theta_from_negative_y_cw(x,y):
    """
    positive theta defined as going from negative y axis clockwise
    """
    if x <= 1e-5:
        if y>0:
            theta = np.pi
        else:
            theta = 0
    elif y <= 1e-5:
        if x>0:
            theta = 3*np.pi/2
        else:
            theta = np.pi/2
    else:
        alpha = torch.atan(y/x)
        if x>0: 
            theta = (3*np.pi/2)-alpha # in first/forth quadrant
        elif x<0:
            theta = (np.pi/2)-alpha # in third/second quadrant
    return theta

def get_theta_from_negative_y_ccw(x,y):
    """
    positive theta defined as going from negative y axis counter clockwise
    """
    if abs(x.item()) < 1e-10:
        if y>0:
            theta = np.pi
        else:
            theta = 0
    elif abs(y.item()) < 1e-10:
        if x>0:
            theta = np.pi/2
        else:
            theta = 3*np.pi/2
    else:
        # theta = torch.atan(x/-y)
        alpha = torch.atan(y/x)
        if x>0: 
            theta = alpha + (np.pi/2)# in first/forth quadrant
        else:
            theta = alpha + (3*np.pi/2) # in third/second quadrant
    return theta