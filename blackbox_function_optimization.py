import numpy as np
from typing import Tuple, Union

coord = 0
delta = lambda step: ((-1)**counter_opt) * (step**0.1 / (step + 100) + 1)
step = 0
counter_opt = 1
iteration = 0

def blackbox_optimize(
        args_history: np.ndarray,
        func_vals_history: np.ndarray
) -> Union[np.ndarray, str]:

    """
    by the history of checked points and blackbox function values in them, 
    returns the point to be checked next or the “stop” line.

    Args:
        args_history: arguments history (args_history.shape = (n, 10))
        func_vals_history: history of function values in the corresponding arguments
    Returns:
        next point (size 10 np.ndarray)
    """
    
    global step, delta, counter_opt, coord, iteration
    
    if func_vals_history.shape[0] > 1:
        best_arg_index = np.argmin(func_vals_history)
        best_arg = args_history[best_arg_index]
    
    if args_history.shape[0] > 1:
        if iteration == 0:
            best_arg[coord] += delta(step)
            iteration += 1
            step += 1
            return best_arg
        elif func_vals_history[-1] == func_vals_history[best_arg_index]:
            iteration += 1
            step += 1
            best_arg[coord] += delta(step)
            return best_arg
        elif func_vals_history[-1] > func_vals_history[best_arg_index]:
            iteration += 1
            step += 1
            if counter_opt % 2 == 0:
                if coord == 9:
                    return 'stop'
                else:
                    counter_opt += 1
                    coord += 1
                    step = 0
                    best_arg[coord] += delta(step)
                    return best_arg
            else:
                counter_opt += 1
                best_arg[coord] += delta(step)
                return best_arg
    elif args_history.shape[0] == 1:
        iteration += 1
        step += 1
        args_history[0][0] += delta(step)
        return args_history[0]
    
    else:
        iteration +=1
        step += 1
        return np.array([1.0]*10)