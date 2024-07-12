import torch
from torch.optim import Optimizer



class AdamWithDiff(Optimizer):
    """
    Adam optimizer with gradient differences for stepsize calculation.
    """
    def __init__(self, params, lr=1e-3, betas=(0.9, 0.999), eps=1e-8, weight_decay=0, correct_bias=True):
        """
        Initializes the optimizer with hyperparameters.

        Args:
            params (iterable): Iterable of parameters to optimize.
            lr (float, optional): Learning rate. Defaults to 1e-3.
            betas (tuple, optional): Coefficients used for computing running averages
                of gradient and its square (default: (0.9, 0.999)).
            eps (float, optional): Epsilon for numerical stability. Defaults to 1e-8.
            amsgrad (bool, optional): Whether to use the AMSGrad variant. Defaults to False.
        """
        if not 0.0 <= lr:
            raise ValueError("Invalid learning rate: {}".format(lr))
        if not 0.0 <= eps:
            raise ValueError("Invalid epsilon value: {}".format(eps))
        if not 0.0 <= betas[0] < 1.0:
            raise ValueError("Invalid beta parameter at index 0: {}".format(betas[0]))
        if not 0.0 <= betas[1] < 1.0:
            raise ValueError("Invalid beta parameter at index 1: {}".format(betas[1]))

        defaults = dict(lr=lr, betas=betas, eps=eps, weight_decay=weight_decay, correct_bias=correct_bias)
        super(AdamWithDiff, self).__init__(params, defaults)

        # State initialization
        for group in self.param_groups:
            for p in group['params']:
                state = self.state[p]
                state['step'] = 0                               # Steps for optimization
                state['m'] = torch.zeros_like(p.data)      # Exponential moving average of gradient values
                state['v'] = torch.zeros_like(p.data)   # Exponential moving average of squared gradient values
                state['prev_grad'] = torch.zeros_like(p.data)   # Previous gradient
    
    def __setstate__(self, state):
        super(AdamWithDiff, self).__setstate__(state)

    def step(self, closure=None):
        """
        Performs a single optimization step.

        Args:
            closure (callable, optional): A closure that reevaluates the model
                and returns the loss. Defaults to None.

        Returns:
            Optional[Tensor]: None if closure is None, otherwise
                the closure return value.
        """
        loss = None
        if closure is not None:
            loss = closure()

        for group in self.param_groups:
            for p in group['params']:
                if p.grad is None:
                    continue
                grad = p.grad.data
                if grad.is_sparse:
                    raise RuntimeError('AdamWithDiff does not support sparse gradients, please consider other optimizers.')
                state = self.state[p]

                state['step'] += 1 
                m, v, prev_grad, = state['m'], state['v'], state['prev_grad']
                beta1, beta2 = group['betas']


                # Just adding the square of the weights to the loss function is *not*
                # the correct way of using L2 regularization/weight decay with Adam,
                # since that will interact with the m and v parameters in strange ways.
                #
                # Instead we want to decay the weights in a manner that doesn't interact
                # with the m/v parameters. This is equivalent to adding the square
                # of the weights to the loss with plain (non-momentum) SGD.
                # Add weight decay at the end (fixed version)
                if group['weight_decay'] != 0:
                    grad.add_(group['weight_decay'], p.data)

                grad_diff = abs(grad - prev_grad)

                m.mul_(beta1).add_(1 - beta1, grad)
                v.mul_(beta2).addcmul_(1 - beta2, grad_diff, grad_diff)

                # if group['correct_bias']:
                m_hat = m / (1 - beta1 ** state['step'])
                v_hat = v / (1 - beta2 ** state['step'])
                denom = v_hat.sqrt().add_(group['eps'])

                p.data.addcdiv_(-group['lr'], m_hat, denom)

                state['prev_grad'] = grad.clone()
                state['m'], state['v'] = m.clone(), v.clone()
                
        return loss, denom
            
class Adam(Optimizer):
    """
    Adam optimizer with gradient differences for stepsize calculation.
    """
    def __init__(self, params, lr=1e-3, betas=(0.9, 0.999), eps=1e-8, weight_decay=0, correct_bias=True):
        """
        Initializes the optimizer with hyperparameters.

        Args:
            params (iterable): Iterable of parameters to optimize.
            lr (float, optional): Learning rate. Defaults to 1e-3.
            betas (tuple, optional): Coefficients used for computing running averages
                of gradient and its square (default: (0.9, 0.999)).
            eps (float, optional): Epsilon for numerical stability. Defaults to 1e-8.
            amsgrad (bool, optional): Whether to use the AMSGrad variant. Defaults to False.
        """
        if not 0.0 <= lr:
            raise ValueError("Invalid learning rate: {}".format(lr))
        if not 0.0 <= eps:
            raise ValueError("Invalid epsilon value: {}".format(eps))
        if not 0.0 <= betas[0] < 1.0:
            raise ValueError("Invalid beta parameter at index 0: {}".format(betas[0]))
        if not 0.0 <= betas[1] < 1.0:
            raise ValueError("Invalid beta parameter at index 1: {}".format(betas[1]))

        defaults = dict(lr=lr, betas=betas, eps=eps, weight_decay=weight_decay, correct_bias=correct_bias)
        super(AdamWithDiff, self).__init__(params, defaults)

        # State initialization
        for group in self.param_groups:
            for p in group['params']:
                state = self.state[p]
                state['step'] = 0                               # Steps for optimization
                state['m'] = torch.zeros_like(p.data)      # Exponential moving average of gradient values
                state['v'] = torch.zeros_like(p.data)   # Exponential moving average of squared gradient values
    
    def __setstate__(self, state):
        super(AdamWithDiff, self).__setstate__(state)

    def step(self, closure=None):
        """
        Performs a single optimization step.

        Args:
            closure (callable, optional): A closure that reevaluates the model
                and returns the loss. Defaults to None.

        Returns:
            Optional[Tensor]: None if closure is None, otherwise
                the closure return value.
        """
        loss = None
        if closure is not None:
            loss = closure()

        for group in self.param_groups:
            for p in group['params']:
                if p.grad is None:
                    continue
                grad = p.grad.data
                if grad.is_sparse:
                    raise RuntimeError('AdamWithDiff does not support sparse gradients, please consider other optimizers.')
                state = self.state[p]

                state['step'] += 1 
                m, v = state['m'], state['v']
                beta1, beta2 = group['betas']

                # Just adding the square of the weights to the loss function is *not*
                # the correct way of using L2 regularization/weight decay with Adam,
                # since that will interact with the m and v parameters in strange ways.
                #
                # Instead we want to decay the weights in a manner that doesn't interact
                # with the m/v parameters. This is equivalent to adding the square
                # of the weights to the loss with plain (non-momentum) SGD.
                # Add weight decay at the end (fixed version)
                if group['weight_decay'] != 0:
                    grad.add_(group['weight_decay'], p.data)

                # grad_diff = abs(grad - prev_grad)

                m.mul_(beta1).add_(1 - beta1, grad)
                v.mul_(beta2).addcmul_(1 - beta2, grad, grad)

                # if group['correct_bias']:
                m_hat = m / (1 - beta1 ** state['step'])
                v_hat = v / (1 - beta2 ** state['step'])
                denom = v_hat.sqrt().add_(group['eps'])

                p.data.addcdiv_(-group['lr'], m_hat, denom)

                state['m'], state['v'] = m.clone(), v.clone()
                
        return loss, denom


class AdaGradWithDiff(Optimizer):
    """
    AdaGrad optimizer with gradient differences for stepsize calculation.
    """
    def __init__(self, params, lr=1e-2, eps=1e-8):
        """
        Initializes the optimizer with hyperparameters.

        Args:
            params (iterable): Iterable of parameters to optimize.
            lr (float, optional): Learning rate. Defaults to 1e-2.
            eps (float, optional): Epsilon for numerical stability. Defaults to 1e-8.
        """
        if not 0.0 <= lr:
            raise ValueError("Invalid learning rate: {}".format(lr))
        if not 0.0 <= eps:
            raise ValueError("Invalid epsilon value: {}".format(eps))

        defaults = dict(lr=lr, eps=eps)
        super(AdaGradWithDiff, self).__init__(params, defaults)

        # State initialization
        for group in self.param_groups:
            for p in group['params']:
                state = self.state[p]
                state['step'] = 0                                   # Steps for optimization
                state['sum_grad_diffs'] = torch.zeros_like(p.data)  # Sum of gradient difference
                state['prev_grad'] = torch.zeros_like(p.data)       # Previous gradient

    def __setstate__(self, state):
        super(AdaGradWithDiff, self).__setstate__(state)

    def step(self, closure=None):
        """
        Performs a single optimization step.

        Args:
            closure (callable, optional): A closure that reevaluates the model
                and returns the loss. Defaults to None.

        Returns:
            Optional[Tensor]: None if closure is None, otherwise
                the closure return value.
        """
        loss = None
        if closure is not None:
            loss = closure()
        for group in self.param_groups:
            for p in group['params']:
                if p.grad is None:
                    continue
                grad = p.grad.data
                if grad.is_sparse:
                    raise RuntimeError('AdaGradWithDiff does not support sparse gradients, please consider other optimizers.')
                
                state = self.state[p]

                state['step'] += 1
                sum_grad_diffs, prev_grad = state['sum_grad_diffs'], state['prev_grad']

                grad_diff = abs(prev_grad - grad)

                sum_grad_diffs.addcmul_(grad_diff, grad_diff)
                denom = sum_grad_diffs.sqrt().add_(group['eps'])

                p.data.addcdiv_(-group['lr'], grad, denom)

                # At the end, change previous gradient to current gradient
                state['prev_grad'] = grad.clone()
                state['sum_grad_diffs'] = sum_grad_diffs.clone()

        return loss, denom

class AdaGrad(Optimizer):
    def __init__(self, params, lr=1e-2, eps=1e-10):
        """
        Initializes the optimizer with hyperparameters.

        Args:
            params (iterable): Iterable of parameters to optimize.
            lr (float, optional): Learning rate. Defaults to 1e-2.
            eps (float, optional): Epsilon for numerical stability. Defaults to 1e-8.
        """
        if not 0.0 <= lr:
            raise ValueError("Invalid learning rate: {}".format(lr))
        if not 0.0 <= eps:
            raise ValueError("Invalid epsilon value: {}".format(eps))

        defaults = dict(lr=lr, eps=eps)
        super(AdaGrad, self).__init__(params, defaults)

        # State initialization
        for group in self.param_groups:
            for p in group['params']:
                state = self.state[p]
                state['step'] = 0                                   # Steps for optimization
                state['sum_grads'] = torch.zeros_like(p.data)  # Sum of gradient difference

    def __setstate__(self, state):
        super(AdaGrad, self).__setstate__(state)
    
    def step(self, closure=None):
        """
        Performs a single optimization step.

        Args:
            closure (callable, optional): A closure that reevaluates the model
                and returns the loss. Defaults to None.

        Returns:
            Optional[Tensor]: None if closure is None, otherwise
                the closure return value.
        """
        loss = None
        if closure is not None:
            loss = closure()
        for group in self.param_groups:
            for p in group['params']:
                if p.grad is None:
                    continue
                grad = p.grad.data
                if grad.is_sparse:
                    raise RuntimeError('AdaGrad does not support sparse gradients, please consider other optimizers.')
                
                state = self.state[p]

                state['step'] += 1
                sum_grads = state['sum_grads']

                sum_grads.addcmul_(grad, grad)
                denom = sum_grads.sqrt().add_(group['eps'])

                p.data.addcdiv_(-group['lr'], grad, denom)

                state['sum_grads'] = sum_grads.clone()

        return loss, denom