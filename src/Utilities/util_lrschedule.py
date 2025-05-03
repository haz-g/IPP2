import numpy as np
from config import args

def update_learning_rate(optimizer, iteration):
    if args["use_clr"]:
        if args["clr_policy"] == "triangular":
            cycle = np.floor(1 + iteration / (2 * args["clr_stepsize"]))
            x = np.abs(iteration / args["clr_stepsize"] - 2 * cycle + 1)
            lr = args["clr_base_lr"] + (args["clr_max_lr"] - args["clr_base_lr"]) * max(0, (1 - x))
            
        elif args["clr_policy"] == "triangular2":
            cycle = np.floor(1 + iteration / (2 * args["clr_stepsize"]))
            x = np.abs(iteration / args["clr_stepsize"] - 2 * cycle + 1)
            lr = args["clr_base_lr"] + (args["clr_max_lr"] - args["clr_base_lr"]) * max(0, (1 - x)) / (2 ** (cycle - 1))
            
        elif args["clr_policy"] == "exp_range":
            cycle = np.floor(1 + iteration / (2 * args["clr_stepsize"]))
            x = np.abs(iteration / args["clr_stepsize"] - 2 * cycle + 1)
            lr = args["clr_base_lr"] + (args["clr_max_lr"] - args["clr_base_lr"]) * max(0, (1 - x)) * (args["clr_gamma"] ** iteration)
        
        optimizer.param_groups[0]["lr"] = lr
        
    elif args["anneal_lr"]:
        # Your original annealing code
        frac = 1.0 - (iteration - 1.0) / args["num_iterations"]
        lr = frac * args["learning_rate"]
        optimizer.param_groups[0]["lr"] = lr
    
    return optimizer.param_groups[0]["lr"]  # Return current lr for logging