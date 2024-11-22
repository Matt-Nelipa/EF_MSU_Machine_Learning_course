import numpy as np
from typing import Callable, Tuple, Union, List

class f1:
    def __call__(self, x: float):
        """
        Args:
            x: float
        Returns:
            float
        """
        return x**2

    def grad(self, x: float):
        """
        Args:
            x: float
        Returns:
            float
        """
        return 2 * x

    def hess(self, x: float):
        """
        Args:
            x: float
        Returns:
            float
        """
        return 2


class f2:
    def __call__(self, x: float):
        """
        Args:
            x: float
        Returns:
            float
        """
        return np.sin(3 * x**(3/2) + 2) + x**2

    def grad(self, x: float):
        """
        Args:
            x: float
        Returns:
            float
        """
        return np.cos(3 * x**(3/2) + 2) * (9 / 2) * np.sqrt(x) + 2 * x

    def hess(self, x: float):
        """
        Args:
            x: float
        Returns:
            float
        """
        return np.cos(3 * x**(3/2) + 2) * 9 / 4 * x**(-1/2) - np.sin(3 * x**(3/2) + 2) * (9 / 2 * x**(1/2))**2 + 2


class f3:
    def __call__(self, x: np.ndarray):
        """
        Args:
            x: numpy array of shape (2,)
        Returns:
            float
        """
        return (((x[0] - 3.3) ** 2 ) / 4) + (((x[1] + 1.7) ** 2) / 15)

    def grad(self, x: np.ndarray):
        """
        Args:
            x: numpy array of shape (2,)
        Returns:
            np.ndarray of shape (2,)
        """
        #grad_f = np.zeros(2)
        grad_f = np.array([(x[0] - 3.3) / 2, (x[1] + 1.7) / 7.5])
        #grad_f[0] = (x[0] - 3.3) / 2
        #grad_f[1] = (x[1] + 1.7) / 7.5
        return grad_f

    def hess(self, x: np.ndarray):
        """
        Args:
            x: numpy array of shape (2,)
        Returns:
            np.ndarray of shape (2, 2)
        """
        hess_f = np.zeros((2, 2))
        hess_f[0,0] = 1/2
        hess_f[1,0] = 0
        hess_f[0,1] = 0
        hess_f[1,1] = 1 / 7.5
        return hess_f

class SquaredL2Norm:
    def __call__(self, x: np.ndarray):
        """
        Args:
            x: numpy array of shape (n,)
        Returns:
            float
        """
        return np.sum(x ** 2)

    def grad(self, x: np.ndarray):
        """
        Args:
            x: numpy array of shape (n,)
        Returns:
            np.ndarray of shape (n,)
        """
        return 2 * x

    def hess(self, x: np.ndarray):
        """
        Args:
            x: numpy array of shape (n,)
        Returns:
            np.ndarray of shape (n, n)
        """
        n = len(x)
        hess = np.zeros((n, n))
        for i in range(n):
            for j in range(n):
                if i == j:
                   hess[i, j] = 2
                else:
                   hess[i, j] = 0

        return hess


class Himmelblau:
    def __call__(self, x: np.ndarray):
        """
        Args:
            x: numpy array of shape (2,)
        Returns:
            float
        """
        return (x[0] ** 2 + x[1] - 11) ** 2 + (x[0] + x[1] ** 2 - 7) ** 2

    def grad(self, x: np.ndarray):
        """
        Args:
            x: numpy array of shape (2,)
        Returns:
            numpy array of shape (2,)
        """
        grad_f = np.zeros(2)
        grad_f[0] = 2 * (x[0] ** 2 + x[1] - 11) * 2 * x[0] + \
        2 * (x[0] + x[1] ** 2 - 7)
        grad_f[1] = 2 * (x[0] ** 2 + x[1] - 11) + \
        2 * (x[0] + x[1] ** 2 - 7) * 2 * x[1]
        return grad_f

    def hess(self, x: np.ndarray):
        """
        Args:
            x: numpy array of shape (2,)
        Returns:
            numpy array of shape (2, 2)
        """
        hess_f = np.zeros((2, 2))
        hess_f[0, 0] = 12 * x[0] ** 2 + 4 * x[1] - 44 + 2
        hess_f[1, 0] = 4 * x[0] + 4 * x[1]
        hess_f[0, 1] = 4 * x[0] + 4 * x[1]
        hess_f[1, 1] = 2 + 12 * x[1] ** 2 + 4 * x[0] - 28
        return hess_f


class Rosenbrok:
    def __call__(self, x: np.ndarray):

        """
        Args:
            x: numpy array of shape (n,) (n >= 2)
        Returns:
            float
        """

        assert x.shape[0] >= 2, "x.shape[0] must be >= 2"
        f = [(100 * (x[i + 1] - x[i]**2)**2 + (1 - x[i])**2) for i in range(len(x) - 1)]
        return sum(f)

    def grad(self, x: np.ndarray):

        """
        Args:
            x: numpy array of shape (n,) (n >= 2)
        Returns:
            numpy array of shape (n,)
        """

        len_x = x.shape[0]

        assert len_x >= 2, "x.shape[0] must be >= 2"
        grad_f = np.zeros(len_x)
        for i in range(len_x):
            if i == 0:
                grad_f[i] = -100 * 2 * x[i] * 2 * (x[i + 1] - x[i]**2) - 2 * (1 - x[i])
            elif i == len_x - 1:
                grad_f[i] = 100 * 2 * (x[i] - x[i - 1]**2)
            else:
                grad_f[i] = 100 * 2 * (x[i] - x[i - 1]**2) + 100 * 2 * (-2) * x[i] * (x[i + 1] - x[i]**2) - 2 * (1 - x[i])
        return grad_f
    
    def hess(self, x: np.ndarray):

        """
        Args:
            x: numpy array of shape (n,) (n >= 2)
        Returns:
            numpy array of shape (n, n)
        """

        len_x = x.shape[0]

        assert len_x >= 2, "x.shape[0] must be >= 2"

        hess = np.zeros([len_x, len_x])
        for i in range(len_x):
            if i == 0:
                hess[i, i] = -400 * x[i + 1] + 1200 * x[i]**2 + 2
                hess[i, i+1] = -400 * x[i]
            
            elif i == len_x - 1:
                hess[i, i] = 200
                hess[i, i-1] = -400 * x[i-1]
                
            else:
                hess[i, i] = 200 - 400 * x[i+1] + 1200 * x[i]**2 + 2
                hess[i, i-1] = -400 * x[i-1]
                hess[i, i+1] = -400 * x[i]

        return hess

def minimize(
        func: Callable,
        x_init: np.ndarray,
        learning_rate: Callable = lambda x: 0.1,
        method: str = 'gd',
        max_iter: int = 10_000,
        stopping_criteria: str = 'function',
        tolerance: float = 1e-2,
) -> Tuple:
    """
    Args:
        func: a function that needs to find a minimum (an object of the class we just wrote)
            (it has to contain the following methods: __call__, grad, hess)
        x_init: initial point
        learning_rate: descent rate
        method:
            "gd" - gradient descent
            "newtone" - Newtone's method
        max_iter: maximum possible number of iterations for the algorithm
        stopping_criteria: the condition under which the algorithm should be stopped
            'points' - stopping by the norm of point difference on neighboring iterations
            'function' - stopping by the norm of the difference of function values on neighboring iterations
            'gradient' - stopping by function gradient norm
        tolerance: how accurately to search for a solution (involved in the stopping criterion)
    Returns:
        x_opt: local minimum point found
        points_history_list: (list) point history list
        functions_history_list: (list) function value history list
        grad_history_list: (list) list with a history of function gradient values
    """

    assert max_iter > 0, 'max_iter must be > 0'
    assert method in ['gd', 'newtone'], 'method can be "gd" or "newtone"!'
    assert stopping_criteria in ['points', 'function', 'gradient'], \
        'stopping_criteria can be "points", "function" or "gradient"!'
    
    x_opt = x_init
    points_history_list = [x_init]
    functions_history_list = [func(x_init)]
    grad_history_list = [func.grad(x_init)]
    
    if method == 'gd':
        
        if stopping_criteria == 'points':
            
            for i in range(max_iter):
                x_prev = x_opt
                x_opt = x_prev - learning_rate(i) * func.grad(x_prev)
                points_history_list.append(x_opt)
                functions_history_list.append(func(x_opt))
                grad_history_list.append(func.grad(x_opt))
                    
                if np.sum(abs(x_prev - x_opt)) < tolerance:
                    return (x_opt, points_history_list, functions_history_list, grad_history_list)
            
            return (x_opt, points_history_list, functions_history_list, grad_history_list)
            
        elif stopping_criteria == 'function':
            
            for i in range(max_iter):
                x_prev = x_opt
                x_opt = x_prev - learning_rate(i) * func.grad(x_prev)
                points_history_list.append(x_opt)
                functions_history_list.append(func(x_opt))
                grad_history_list.append(func.grad(x_opt))
                    
                if np.sum(abs(func(x_prev) - func(x_opt))) < tolerance:
                    return (x_opt, points_history_list, functions_history_list, grad_history_list)
                    
            return (x_opt, points_history_list, functions_history_list, grad_history_list)
            
        else:
            
            for i in range(max_iter):
                x_prev = x_opt
                x_opt = x_prev - learning_rate(i) * func.grad(x_prev)
                points_history_list.append(x_opt)
                functions_history_list.append(func(x_opt))
                grad_history_list.append(func.grad(x_opt))
                    
                if np.sum(abs(func.grad(x_prev) - func.grad(x_opt))) < tolerance:
                    return (x_opt, points_history_list, functions_history_list, grad_history_list)
                    
            return (x_opt, points_history_list, functions_history_list, grad_history_list)
            
    else:
        
        if stopping_criteria == 'points':
            
            for i in range(max_iter):
                x_prev = x_opt
                x_opt = x_prev - learning_rate(i) * func.grad(x_prev) / func.hess(x_prev) if np.shape(x_prev) == () else x_prev - learning_rate(i) * np.linalg.inv(func.hess(x_prev)) @ func.grad(x_opt)
                points_history_list.append(x_opt)
                functions_history_list.append(func(x_opt))
                grad_history_list.append(func.grad(x_opt))
                    
                if np.sum(abs(x_prev - x_opt)) < tolerance:
                    return (
                        x_opt, points_history_list, 
                        functions_history_list, 
                        grad_history_list)
            
            return (
                x_opt, points_history_list, 
                functions_history_list, 
                grad_history_list)
        
        elif stopping_criteria == 'function':
            
            for i in range(max_iter):
                x_prev = x_opt
                x_opt = x_prev - learning_rate(i) * func.grad(x_prev) / func.hess(x_prev) if np.shape(x_prev) == () else x_prev - learning_rate(i) * np.linalg.inv(func.hess(x_prev)) @ func.grad(x_opt)
                points_history_list.append(x_opt)
                functions_history_list.append(func(x_opt))
                grad_history_list.append(func.grad(x_opt))
                    
                if np.sum(abs(func(x_prev) - func(x_opt))) < tolerance:
                    return (
                        x_opt, points_history_list, 
                        functions_history_list, 
                        grad_history_list)
            
            return (
                x_opt, points_history_list, 
                functions_history_list, 
                grad_history_list)
        
        else:
            
            for i in range(max_iter):
                x_prev = x_opt
                x_opt = x_prev - learning_rate(i) * func.grad(x_prev) / func.hess(x_prev) if np.shape(x_prev) == () else x_prev - learning_rate(i) * np.linalg.inv(func.hess(x_prev)) @ func.grad(x_opt)
                points_history_list.append(x_opt)
                functions_history_list.append(func(x_opt))
                grad_history_list.append(func.grad(x_opt))
                    
                if np.sum(abs(func.grad(x_prev) - func.grad(x_opt))) < tolerance:
                    return (
                        x_opt, points_history_list, 
                        functions_history_list, 
                        grad_history_list)
            
            return (
                x_opt, points_history_list, 
                functions_history_list, 
                grad_history_list)
            