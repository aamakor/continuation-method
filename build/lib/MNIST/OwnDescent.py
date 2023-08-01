from email.errors import InvalidMultipartContentTransferEncodingDefect
import numpy as np
import torch
from torch.optim import Optimizer
from importlib import resources
import io


from torch import Tensor
from typing import List, Optional

class OwnDescent(Optimizer):
    
    def __init__(self, params, lr, sr, alpha):
        if lr is None or lr < 0.0:
            raise ValueError("Invalid learning rate: {}".format(lr))

        defaults = dict(lr=lr, sr=sr)

        super(OwnDescent, self).__init__(params, defaults)

        self.alpha = alpha
        # parameters for nesterov acceleration
        self.k = 0
        # our model has only on param_group so this is okay
        self.last_p = self.param_groups[0]['params'].copy()

        # #random start velocity
        #for i, p in enumerate(self.last_p):
        #    self.last_p[i] = p + torch.randn_like(p)

    def __setstate__(self, state):
        super(OwnDescent, self).__setstate__(state)

    @torch.no_grad()
    def step_direction(self):
        """Performs a single optimization step.
        Args:
            closure (callable, optional): A closure that reevaluates the model
                and returns the loss.
        """
        
        for group in self.param_groups:
            
            lr = group['lr']

            step_dir = list()

            for p in group['params']:

                step_dir.append(torch.clone(p.grad))
                
        return step_dir

    @torch.no_grad()
    def step(self, closure=None):
        """Performs a single optimization step.
        Args:
            closure (callable, optional): A closure that reevaluates the model
                and returns the loss.
        """

        for group in self.param_groups:
            
            lr = group['lr']

            for p in group['params']:

                p.add_(p.grad, alpha=-lr)

    @torch.no_grad()
    def fixed_step(self, step):
        """Performs a single optimization step.
        Args:
            closure (callable, optional): A closure that reevaluates the model
            and returns the loss.
        """

        for group in self.param_groups:
            
            lr = group['lr']

            for i, p in enumerate(group['params']):
                
                p.add_(step[i], alpha=-lr)

    @torch.no_grad()
    def acceleration_step(self):

        # store l1 norm of current parmeters
        p_ = [torch.clone(p) for p in self.param_groups[0]['params']]
        l1norm_p_current = sum([torch.sum(torch.abs(p__)) for p__ in p_])

        p_old = []

        for group in self.param_groups:
            
            lr = group['lr']

            for i, p in enumerate(group['params']):
                
                #save last iteration in case of restart
                p_old.append(torch.clone(p))

                #acceleration
                acc_step = torch.add(p, self.last_p[i], alpha=-1)
                acc_alpha = (self.k -1)/(self.alpha+self.k)
                acc_step = torch.mul(acc_step, acc_alpha)
                p.add_(acc_step, alpha=1)

                #update for acceleration
                #use torch.clone() to create a copy
                self.last_p[i] = torch.clone(p)
        
        #l1norm_p = sum([torch.])

        l1norm_p_acc = sum([torch.sum(torch.abs(p__)) for p__ in self.param_groups[0]['params']])
        
        # restart acceleration scheme if l1 norm is increased by acceleration step
        if l1norm_p_acc > l1norm_p_current:
            # undo acceleration step
        
            for group in self.param_groups:

                for i, p in enumerate(group['params']):
                                       
                    p.copy_(p_old[i])

                    #use torch.clone() to create a copy
                    self.last_p[i] = torch.clone(p)

            self.k = 0

        else:

            # update acceleration 
            self.k = self.k + 1
        
    @torch.no_grad()
    def shrinkage(self):

        """Performs a single optimization step.
        Args:
            closure (callable, optional): A closure that reevaluates the model
                and returns the loss.
        """

        for group in self.param_groups:
            
            sr = group['sr']

            for p in group['params']:

                # shrinkage operator
                c = torch.mul(torch.sign(p), torch.max(torch.add(torch.abs(p), torch.ones_like(p), alpha = -sr), torch.zeros_like(p)))  # the alpha multiples the torch.ones_like(p) before adding it to torch.abs(p) 
                p.add_(p, alpha = -1)
                p.add_(c, alpha=1)

    @torch.no_grad()
    def proximalgradientstep(self):

        """Performs a single optimization step.
        Args:
            closure (callable, optional): A closure that reevaluates the model
                and returns the loss.
        """

        for group in self.param_groups:
            
            lr = group['lr']
            sr = group['sr']

            for p in group['params']:

                p.add_(p.grad, alpha = -lr)
                c = torch.mul(torch.sign(p), torch.max(torch.add(torch.abs(p), torch.ones_like(p), alpha = -sr), torch.zeros_like(p)))
                p.add_(p, alpha = -1)
                p.add_(c, alpha=1)

                #p.copy_(c)


    @torch.no_grad()
    def MOOproximalgradientstep(self):

        """Performs a single optimization step.
        Args:
            closure (callable, optional): A closure that reevaluates the model
                and returns the loss.
        """

        for group in self.param_groups:
            
            lr = group['lr']

            x = [torch.tensor(p) for p in group['params']]
            d = [torch.tensor(p.grad) for p in group['params']]

            x, structure = self.stackTensor(x)
            d = self.stackTensor(d)[0]

            Y = self.MOOproximalgradientUpdate(x, d, lr)
            Y = self.convertStackedTensorToStructuredTensor(Y, structure)

            for p, y in zip(group['params'], Y):

                p.add_(p, alpha = -1)
                p.add_(y, alpha=1)

    @torch.no_grad()
    def MOOproximalgradientstep_st(self):

        """Performs a single optimization step.
        Args:
            closure (callable, optional): A closure that reevaluates the model
                and returns the loss.
        """

        for group in self.param_groups:
            
            lr = group['lr']

            x = [torch.tensor(p) for p in group['params']]
            d = [torch.tensor(p.grad) for p in group['params']]

            x, structure = self.stackTensor(x)
            d = self.stackTensor(d)[0]

            Y = self.MOOproximalgradientUpdate_st(x, d, lr)
            Y = self.convertStackedTensorToStructuredTensor(Y, structure)

            for p, y in zip(group['params'], Y):

                p.add_(p, alpha = -1)
                p.add_(y, alpha=1)           

    @torch.no_grad()
    def MOOproximalgradientUpdate_st(self, x, d, h):
        
        # step should be a torch tensor of dimension 1 times 
        
        y_ = lambda alpha : torch.mul(torch.sign(x.add(d, alpha = -h*alpha)), torch.max(torch.add(torch.abs(x.add(d, alpha = -h*alpha)), torch.ones_like(x), alpha = -h*(1-alpha)), torch.zeros_like(x)))

        omega_1 = lambda alpha : torch.dot(d, y_(alpha) - x)

        l1_x = torch.norm(x, p = 1)

        omega_2 = lambda alpha : torch.norm(y_(alpha), p = 1) - l1_x 


        alpha = .5
 
        for j in range(10):

            if omega_1(alpha) > omega_2(alpha):

                alpha = alpha + (.5)**(j+2)

            else:

                alpha = alpha - (.5)**(j+2)

        y = y_(alpha)

        if alpha > .25 and alpha < .75:
            print('yo')

        return y
    

    @torch.no_grad()
    def MOOproximalgradientUpdate(self, x, d, h):
        
        # step should be a torch tensor of dimension 1 times 
        
        y_ = lambda alpha : torch.mul(torch.sign(x.add(d, alpha = -h*alpha)), torch.max(torch.add(torch.abs(x.add(d, alpha = -h*alpha)), torch.ones_like(x), alpha = -h*(1-alpha)), torch.zeros_like(x)))

        omega_1 = lambda alpha : torch.dot(d, y_(alpha) - x)

        l1_x = torch.norm(x, p = 1)

        omega_2 = lambda alpha : torch.norm(y_(alpha), p = 1) - l1_x 


        alpha = .5
 
        for j in range(100):

            if omega_1(alpha) > omega_2(alpha):

                alpha = alpha + (.5)**(j+2)

            else:

                alpha = alpha - (.5)**(j+2)

        y = y_(alpha)

        if alpha > .25 and alpha < .75:
            print('yo')

        return y

    def stackTensor(self, tensor):

        # save size of each tensor in a list for reconstruction purposes
        structure = [tensor_.size() for tensor_ in tensor]

        stacked_tensor = torch.cat([tensor_.reshape(-1) for tensor_ in tensor])

        return stacked_tensor, structure

    def convertStackedTensorToStructuredTensor(self, tensor_direction, structure):
        
        # create empty list with length matching the structure
        tensor = []

        for s in structure:
            
            if len(s) == 1:

                J = s[0]
                tensor_t = tensor_direction[0:J]

                tensor_t = torch.tensor(tensor_t)

                tensor.append(tensor_t)

                tensor_direction = tensor_direction[J::]

            elif len(s) == 2:

                J = s[0]*s[1]
                tensor_t = tensor_direction[0:J]

                tensor_t = torch.tensor(tensor_t)
                tensor_t = torch.reshape(tensor_t, s)

                tensor.append(tensor_t)

                tensor_direction = tensor_direction[J::]

            else:

                print('Warning: Tensor with 3 axis!')

        if len(tensor_direction) > 0:
            print('Conversion of tensor direction from list to structured tensor failed. There are remaining elements in the list!')

        return tensor

