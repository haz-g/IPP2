import time

args = {
    # SETUP
    "seed": 2, # Choose from: [1, 42, 100, 2023, 2024]
    "torch_deterministic": True,
    "cuda": True,
    "track": True,
    "wandb_project_name": "IPP-second-paper-generalist",
    "wandb_entity": 'IPP-experiments',
    "capture_video": False,
    'train_envs_list': ['easy_envs_96', 'medium_easy_envs_96', 
                       'medium_envs_96','medium_hard_envs_96',
                       'hard_envs_96'],
    "test_log_freq": 400,
    'meta_ep_size': 64,
    'DREST_lambda_factor': 0.9,
    'DREST_agent_on': True,

    # ALGO
    "env_id": "easy_envs_96",
    "total_timesteps": 100_000_000,
    "num_envs": 2,
    "num_steps": 128,
    "gamma": 0.99,
    "gae_lambda": 0.95,
    "num_minibatches": 4,
    "update_epochs": 4,
    "norm_adv": True,
    "clip_coef": 0.2,
    "clip_vloss": True,
    "ent_coef": 0.02,
    "vf_coef": 0.5,
    "max_grad_norm": 0.5,
    "target_kl": None,
    "learning_rate": 3e-5,
    "anneal_lr": False,
    "test_train_split": False,
    
    # CLR Parameters
    "use_clr": True,  # Set this to True to enable CLR
    "clr_policy": "triangular",
    "number_of_cycles": 16,  # how many complete cycles over training
    "clr_base_lr": 7.5e-5,  # 1/4 of max_lr
    "clr_max_lr": 3e-4,    # Matches your original learning_rate
}

args["model_save_name"] = f"ppo_CL5x96_SA24_THDec_penwalls2_{args['seed']}_{args['DREST_lambda_factor']}_{args['meta_ep_size']}_{args['total_timesteps']}"

RUN_TAG = time.strftime('%d%H%M')

# Additional Configurations Auto-Complete
args['run_name'] = f'D{RUN_TAG}'
args["batch_size"] = int(args["num_envs"] * args["num_steps"])
args["minibatch_size"] = int(args["batch_size"] // args["num_minibatches"])
args["num_iterations"] = args["total_timesteps"] // args["batch_size"]
args["clr_stepsize"] = args["num_iterations"] / (2 * args["number_of_cycles"])