import time

args = {
    # SETUP
    "seed": 1, # Choose from: [1, 42, 100, 2023, 2024]
    "torch_deterministic": True,
    "cuda": True,
    "track": True,
    "wandb_project_name": "IPP-second-paper-generalist",
    "wandb_entity": 'IPP-experiments',
    'curriculum_learning_on': False, # Define if you want the model to train using a curriculum of environments
    'curriculum_learning_range': [0,4], # Define the range of tiers over which the model should train (0 = T1, 1 = T2, etc.)
    'single_env_training': 0, # Define the environment tier you'd like to focus model training on if "curriculum_learning_on" == False (0 = T1, 1 = T2, etc.)
    'load_existing_model': False,
    'existing_model': 'src/models/D212213T0U84N55.pt',
    "test_log_freq": 400,
    'meta_ep_size': 128,
    'DREST_lambda_factor': 0.9,
    'DREST_agent_on': True,
    'meta_ep_on': True,

    # ALGO
    "total_timesteps": 100_000_000,
    "num_envs": 4,
    "num_steps": 128,
    "gamma": 0.99,
    "gae_lambda": 0.95,
    "num_minibatches": 4,
    "update_epochs": 4,
    "norm_adv": True,
    "clip_coef": 0.2,
    "clip_vloss": True,
    "ent_coef": 0.01,
    "vf_coef": 0.5,
    "max_grad_norm": 0.5,
    "target_kl": None,
    "learning_rate": 3e-4,
    "anneal_lr": True,
    "test_train_split": False,
    
    # CLR Parameters
    "use_clr": True,  # Set this to True to enable CLR
    "clr_policy": "triangular",
    "number_of_cycles": 16,  # how many complete cycles over training
    "clr_base_lr": 7.5e-5,  # 1/4 of max_lr
    "clr_max_lr": 3e-4,    # Matches your original learning_rate
}

RUN_TAG = time.strftime('%d%H%M')
train_envs = ['easy_envs_96', 'medium_easy_envs_96', 'medium_envs_96','medium_hard_envs_96', 'hard_envs_96']

# Additional Configurations Auto-Complete
args['run_name'] = f'D{RUN_TAG}'
args["batch_size"] = int(args["num_envs"] * args["num_steps"])
args["minibatch_size"] = int(args["batch_size"] // args["num_minibatches"])
args["num_iterations"] = args["total_timesteps"] // args["batch_size"]
args["clr_stepsize"] = args["num_iterations"] / (2 * args["number_of_cycles"])
if args['curriculum_learning_on']:
    args["train_envs_list"] = train_envs[args["curriculum_learning_range"][0]:args["curriculum_learning_range"][1]]
else:
    args["train_envs_list"] = [train_envs[args["single_env_training"]]]

print(f"LOADED {len(args['train_envs_list'])} TIER(S) FOR TRAINING")
if args["load_existing_model"]:
    print(f"LOADING EXISTING MODEL: {args['existing_model']}")