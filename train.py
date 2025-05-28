import random
import time
import gymnasium as gym
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import pickle as pkl
import wandb
import time
import numpy as np
import random
import timeout_decorator
import matplotlib.pyplot as plt
from config import args
from math import isnan
from src.Utilities.evals_utils import evaluate_agent
from ppo_agent import Agent
from src.Generalist.grid_env import RandomEnvWrapper
from src.Utilities.util_lrschedule import update_learning_rate
import sys
from src.Utilities.draw_gridworld import draw_policy
from src.Utilities.evals_utils import normalise_ratio_with_exp

if __name__ == "__main__":
    sys.path.append('.')

    training_envs = []

    for env_path in args['train_envs_list']:
        with open(f'{env_path}.pkl', "rb") as f:
            envs = pkl.load(f)
            training_envs.extend(envs)

    if args['track']:
        wandb.init(
            project=args['wandb_project_name'],
            entity=args['wandb_entity'],
            config=args,
            name=args['run_name'],
            monitor_gym=True,
            save_code=True,
        )

    # TRY NOT TO MODIFY: seeding
    random.seed(args['seed'])
    np.random.seed(args['seed'])
    torch.manual_seed(args['seed'])
    torch.backends.cudnn.deterministic = args['torch_deterministic']

    device = torch.device("cuda" if torch.cuda.is_available() and args['cuda'] else "cpu")

    # env setup
    cur_lb = 0
    cur_ub = 80
    cur_best_avr_usefulness = 0
    cur_best_avr_neutrality = 0
    CUR_BEST_SCORE = [0,0,0,0,0]
    envs = gym.vector.SyncVectorEnv([lambda: RandomEnvWrapper(training_envs, cur_lb, cur_ub) for _ in range(args['num_envs'])],)

    assert isinstance(envs.single_action_space, gym.spaces.Discrete), "only discrete action space is supported"

    agent = Agent(envs).to(device)
    optimizer = optim.Adam(agent.parameters(), lr=args['learning_rate'], eps=1e-5)

    # ALGO Logic: Storage setup
    obs = torch.zeros((args['num_steps'], args['num_envs']) + envs.single_observation_space.shape).to(device)
    actions = torch.zeros((args['num_steps'], args['num_envs']) + envs.single_action_space.shape).to(device)
    logprobs = torch.zeros(args['num_steps'], args['num_envs']).to(device)
    rewards = torch.zeros(args['num_steps'], args['num_envs']).to(device)
    dones = torch.zeros(args['num_steps'], args['num_envs']).to(device)
    values = torch.zeros(args['num_steps'], args['num_envs']).to(device)

    # TRY NOT TO MODIFY: start the game
    global_step = 0
    start_time = time.time()
    next_obs, _ = envs.reset(seed=args['seed'])
    next_obs = torch.Tensor(next_obs).to(device)
    next_done = torch.zeros(args['num_envs']).to(device)
    
    for iteration in range(1, args['num_iterations'] + 1):

        update_learning_rate(optimizer, iteration)

        for step in range(0, args['num_steps']):
            global_step += args['num_envs']
            obs[step] = next_obs
            dones[step] = next_done

            # ALGO LOGIC: action logic
            with torch.no_grad():
                action, logprob, _, value = agent.get_action_and_value(next_obs)
                values[step] = value.flatten()
            actions[step] = action
            logprobs[step] = logprob

            # TRY NOT TO MODIFY: execute the game and log data.
            next_obs, reward, terminations, truncations, infos = envs.step(action.cpu().numpy())
            next_done = np.logical_or(terminations, truncations)
            #print(rewards)
            rewards[step] = torch.tensor(reward).to(device).view(-1)
            next_obs, next_done = torch.Tensor(next_obs).to(device), torch.Tensor(next_done).to(device)

        # bootstrap value if not done
        with torch.no_grad():
            next_value = agent.get_value(next_obs).reshape(1, -1)
            advantages = torch.zeros_like(rewards).to(device)
            lastgaelam = 0
            for t in reversed(range(args['num_steps'])):
                if t == args['num_steps'] - 1:
                    nextnonterminal = 1.0 - next_done
                    nextvalues = next_value
                else:
                    nextnonterminal = 1.0 - dones[t + 1]
                    nextvalues = values[t + 1]
                delta = rewards[t] + args['gamma'] * nextvalues * nextnonterminal - values[t]
                advantages[t] = lastgaelam = delta + args['gamma'] * args['gae_lambda'] * nextnonterminal * lastgaelam
            returns = advantages + values

        # flatten the batch
        b_obs = obs.reshape((-1,) + envs.single_observation_space.shape)
        b_logprobs = logprobs.reshape(-1)
        b_actions = actions.reshape((-1,) + envs.single_action_space.shape)
        b_advantages = advantages.reshape(-1)
        b_returns = returns.reshape(-1)
        b_values = values.reshape(-1)

        # Optimizing the policy and value network
        b_inds = np.arange(args['batch_size'])
        clipfracs = []

        for epoch in range(args['update_epochs']):
            np.random.shuffle(b_inds)
            for start in range(0, args['batch_size'], args['minibatch_size']):
                end = start + args['minibatch_size']
                mb_inds = b_inds[start:end]
                _, newlogprob, entropy, newvalue = agent.get_action_and_value(b_obs[mb_inds], b_actions.long()[mb_inds])
                logratio = newlogprob - b_logprobs[mb_inds]
                ratio = logratio.exp()

                with torch.no_grad():
                    # calculate approx_kl http://joschu.net/blog/kl-approx.html
                    old_approx_kl = (-logratio).mean()
                    approx_kl = ((ratio - 1) - logratio).mean()
                    clipfracs += [((ratio - 1.0).abs() > args['clip_coef']).float().mean().item()]

                mb_advantages = b_advantages[mb_inds]
                if args['norm_adv']:
                    mb_advantages = (mb_advantages - mb_advantages.mean()) / (mb_advantages.std() + 1e-8)

                # Policy loss
                pg_loss1 = -mb_advantages * ratio
                pg_loss2 = -mb_advantages * torch.clamp(ratio, 1 - args['clip_coef'], 1 + args['clip_coef'])
                pg_loss = torch.max(pg_loss1, pg_loss2).mean()

                # Value loss
                newvalue = newvalue.view(-1)
                if args['clip_vloss']:
                    v_loss_unclipped = (newvalue - b_returns[mb_inds]) ** 2
                    v_clipped = b_values[mb_inds] + torch.clamp(
                        newvalue - b_values[mb_inds],
                        -args['clip_coef'],
                        args['clip_coef'],
                    )
                    v_loss_clipped = (v_clipped - b_returns[mb_inds]) ** 2
                    v_loss_max = torch.max(v_loss_unclipped, v_loss_clipped)
                    v_loss = 0.5 * v_loss_max.mean()
                else:
                    v_loss = 0.5 * ((newvalue - b_returns[mb_inds]) ** 2).mean()

                entropy_loss = entropy.mean()
                loss = pg_loss - args['ent_coef'] * entropy_loss + v_loss * args['vf_coef']

                optimizer.zero_grad()
                loss.backward()
                nn.utils.clip_grad_norm_(agent.parameters(), args['max_grad_norm'])
                optimizer.step()

            if args['target_kl'] is not None and approx_kl > args['target_kl']:
                break

        y_pred, y_true = b_values.cpu().numpy(), b_returns.cpu().numpy()
        var_y = np.var(y_true)
        explained_var = np.nan if var_y == 0 else 1 - np.var(y_true - y_pred) / var_y

        if args['track']:
            wandb.log({
                "charts/learning_rate": optimizer.param_groups[0]["lr"],
                "losses/value_loss": v_loss.item(),
                "losses/policy_loss": pg_loss.item(),
                "losses/entropy": entropy_loss.item(),
                "losses/old_approx_kl": old_approx_kl.item(),
                "losses/approx_kl": approx_kl.item(), 
                "losses/clipfrac": np.mean(clipfracs), 
                "losses/explained_variance": explained_var, 
                'charts/env_steps': global_step,
                'charts/steps_per_second': int(global_step / (time.time() - start_time)),
                }, step=global_step)
        
        if iteration % int(args['num_iterations']/args['test_log_freq']) == 0:
            agent.eval()
            if args["DREST_agent_on"]:
                train_usefulness_list = []
                train_neutrality_list = []
                train_traj_short_list = []
                train_traj_long_list = []
                policy_plots = []
                unique_design_indices = range(cur_lb, cur_ub, 8)
                if args['test_train_split']:   
                    test_usefulness_list = []
                    test_neutrality_list = []
                    test_traj_short_list = []
                    test_traj_long_list = [] 
                    test_indices = range(cur_ub,cur_ub+16)
                    full_index_list = [*unique_design_indices] + [*test_indices]
                else:
                    full_index_list = [*unique_design_indices]

                with torch.no_grad():
                    for i in full_index_list:
                        env = training_envs[i]
                        env.reset()
                        try:
                            traj_ratio, useful, neutral = evaluate_agent(
                            env,
                            model=agent,
                            max_coins_by_trajectory=np.array([
                                env.max_coins[1],  # longer trajectory
                                env.max_coins[0]   # shorter trajectory
                            ]))

                            '''   
                            # Draw policy and create wandb image
                            plt.figure(figsize=(10, 5))
                            draw_policy(env, agent, device)
                            policy_plot = wandb.Image(plt.gcf(), caption=f'Policy for Env {i} at iteration {iteration}')
                            policy_plots.append(policy_plot)
                            plt.close()
                            ''' 

                            if isnan(neutral):
                                neutral = 0
                            if i < cur_ub:
                                train_usefulness_list.append(useful)
                                train_neutrality_list.append(neutral)
                                train_traj_short_list.append(traj_ratio[1])
                                train_traj_long_list.append(traj_ratio[0])
                            else:
                                test_usefulness_list.append(useful)
                                test_neutrality_list.append(neutral)
                                test_traj_short_list.append(traj_ratio[1])
                                test_traj_long_list.append(traj_ratio[0])
                                
                        except timeout_decorator.TimeoutError:
                            print(f"Skipping evaluation of train env at window index {i} - timed out after 4 seconds")
                            continue
                
                    usefulness = np.mean(train_usefulness_list)
                    neutrality = np.mean(train_neutrality_list)
                    mean_short_traj = np.mean(train_traj_short_list)
                    mean_long_traj = np.mean(train_traj_long_list)

                    print(f'\n----STEP {round(global_step,-3)} | TIER: {round((cur_lb/96)+1,2)}----')
                    print(f'TRAIN AVR. - USE: {round(np.mean(train_usefulness_list),2)} | NEU: {round(np.mean(train_neutrality_list),2)} | TRA: [{round(np.mean(train_traj_short_list),2)},{round(np.mean(train_traj_long_list),2)}]\n---')
                    idx = 0
                    for i in range(cur_lb, cur_ub, 8):
                        print(f'env{i} - USE: {round(train_usefulness_list[idx],2)} | NEU: {round(train_neutrality_list[idx],2)} | TRA: [{round(train_traj_short_list[idx],2)},{round(train_traj_long_list[idx],2)}]')
                        idx += 1
                    
                    if args['test_train_split']:
                        print(f'---\nTEST AVR. - USE: {round(np.mean(test_usefulness_list),2)} | NEU: {round(np.mean(test_neutrality_list),2)} | TRA: [{round(np.mean(test_traj_short_list),2)},{round(np.mean(test_traj_long_list),2)}]\n')
                    
                    if args['track']:
                        if (usefulness + neutrality)/2 > (cur_best_avr_usefulness + cur_best_avr_neutrality)/2:
                            artifact = wandb.Artifact(f"{args['run_name']}U{int(round(usefulness,2)*100)}N{int(round(neutrality,2)*100)}", type='model')
                            torch.save(agent.state_dict(), f"src/models/{args['run_name']}U{int(round(usefulness,2)*100)}N{int(round(neutrality,2)*100)}.pt")
                            artifact.add_file(f"src/models/{args['run_name']}U{int(round(usefulness,2)*100)}N{int(round(neutrality,2)*100)}.pt")
                            wandb.log_artifact(artifact)
                            cur_best_avr_usefulness = usefulness
                            cur_best_avr_neutrality = neutrality

                        wandb.log({
                            'train_metrics/Usefulness': usefulness,
                            'train_metrics/Neutrality': neutrality,
                            'train_metrics/Trajectory_Ratio': normalise_ratio_with_exp(mean_short_traj, mean_long_traj),
                            'curriculum/window_start': cur_lb,
                            'curriculum/window_end': cur_ub,
                            'curriculum/tier': cur_ub // (cur_ub-cur_lb)
                            #'train_metrics/PolicyVisualisations': policy_plots
                        }, step=global_step)
                
                window_size = (cur_ub-cur_lb)
                threshold = round(0.8 - 0.005, 2)  # Gradually lower threshold for higher tiers
                threshold = max(0.7, threshold)  # Don't go below 0.7

                #if min(usefulness, neutrality) >= threshold:
                if usefulness >= threshold:
                    shift_amount = 24 # int(window_size // 4)
                    new_lb = min(cur_lb + shift_amount, 383)
                    new_ub = min(cur_ub + shift_amount, 479)
                    if new_lb > cur_lb:
                        print(f"Advancing curriculum window to [{new_lb}:{new_ub}] (Tier {round((new_lb/96)+1,3)})")
                    cur_lb = new_lb
                    cur_ub = new_ub
                    envs = gym.vector.SyncVectorEnv([lambda: RandomEnvWrapper(training_envs, cur_lb, cur_ub) for _ in range(args['num_envs'])],)
                    next_obs, _ = envs.reset(seed=args['seed'])
                    next_obs = torch.Tensor(next_obs).to(device)
                    next_done = torch.zeros(args['num_envs']).to(device)
            else:
                agent_scores = []
                agent_scores_test = []
                policy_plots = []
                unique_design_indices = range(cur_lb, cur_ub, 8)
                if args['test_train_split']:
                    test_indices = range(cur_ub,cur_ub+16)
                    full_index_list = [*unique_design_indices] + [*test_indices]
                else:
                    full_index_list = [*unique_design_indices]

                with torch.no_grad():
                    for i in full_index_list:
                        env = training_envs[i]
                        env.reset()

                        try:
                            env_reward = 0
                            env = training_envs[i]
                            max_reward = max(env.max_coins)
                            x = 20

                            for _ in range(x):
                                cur_obs, _ = env.reset()
                                cur_obs = torch.Tensor(cur_obs).unsqueeze(0).to(device)
                                done = False
                                episode_reward = 0
                                
                                while not done:
                                    with torch.no_grad():
                                        action, _, _, _ = agent.get_action_and_value(cur_obs)
                                    
                                    new_obs, reward, terminated, truncated, _ = env.step(action.cpu().item())
                                    done = terminated or truncated
                                    episode_reward += reward
                                    cur_obs = torch.Tensor(new_obs).unsqueeze(0).to(device)

                                #print(f'EPISODE REW for: {episode_reward} - MAX Rew: {max_reward}')
                                env_reward += episode_reward/max_reward
                            
                            avr_reward = env_reward/x

                            if i < cur_ub:
                                agent_scores.append(avr_reward)
                            else:
                                agent_scores_test.append(avr_reward)

                            # Draw policy and create wandb image
                            plt.figure(figsize=(10, 5))
                            draw_policy(env, agent, device)
                            policy_plot = wandb.Image(plt.gcf(), caption=f'Policy for Env {i} at iteration {iteration}')
                            policy_plots.append(policy_plot)
                            plt.close() 

                        except timeout_decorator.TimeoutError:
                            print(f"Skipping evaluation of train env at window index {i} - timed out after 4 seconds")
                            continue

                    print(f'\n----STEP {round(global_step,-3)} | TIER: {round((cur_lb/96)+1,2)}----')
                    print(f'TRAIN AVR. - SCORE: {round(np.mean(agent_scores),2)} \n---')
                    idx = 0
                    for i in full_index_list:
                        print(f'env{i} - SCORE: {round(agent_scores[idx],2)}')
                        idx += 1
                    
                    if args['test_train_split']:
                        print(f'---\nTEST AVR. - SCORE: {round(np.mean(agent_scores_test),2)}\n')
                    
                    AGENT_SCORE = round(np.mean(agent_scores),2)
                    
                    if args['track']:
                        if AGENT_SCORE > CUR_BEST_SCORE[cur_lb//96]:
                            artifact = wandb.Artifact(f"{args['run_name']}T{(cur_lb//96)+1}R{int(round(AGENT_SCORE,2)*100)}", type='model')
                            torch.save(agent.state_dict(), f"src/models/{args['run_name']}T{(cur_lb//96)+1}R{int(round(AGENT_SCORE,2)*100)}.pt")
                            artifact.add_file(f"src/models/{args['run_name']}T{(cur_lb//96)+1}R{int(round(AGENT_SCORE,2)*100)}.pt")
                            wandb.log_artifact(artifact)
                            CUR_BEST_SCORE[cur_lb//96] = AGENT_SCORE

                        wandb.log({
                            'train_metrics/Score': AGENT_SCORE,
                            'curriculum/window_start': cur_lb,
                            'curriculum/window_end': cur_ub,
                            'curriculum/tier': cur_ub // (cur_ub-cur_lb),
                            'train_metrics/PolicyVisualisations': policy_plots
                        }, step=global_step)
                
                window_size = (cur_ub-cur_lb)

                threshold = round(0.8 - (0.05 * (cur_ub / window_size) if (cur_ub / window_size) > 1 else 0), 2)  # Gradually lower threshold for higher tiers
                threshold = max(0.6, threshold)  # Don't go below 0.6

                #if min(usefulness, neutrality) >= threshold:
                if AGENT_SCORE >= threshold:
                    shift_amount = int(window_size // 4) #change to level set size / 4
                    new_lb = min(cur_lb + shift_amount, 383)
                    new_ub = min(new_lb + window_size, 479)
                    if new_lb > cur_lb:
                        print(f"Advancing curriculum window to [{new_lb}:{new_ub}] | Tier {round((new_lb / 96)+1,2)} | THR: {threshold}")
                    cur_lb = new_lb
                    cur_ub = new_ub
                    envs = gym.vector.SyncVectorEnv([lambda: RandomEnvWrapper(training_envs, cur_lb, cur_ub) for _ in range(args['num_envs'])],)
                    next_obs, _ = envs.reset(seed=args['seed'])
                    next_obs = torch.Tensor(next_obs).to(device)
                    next_done = torch.zeros(args['num_envs']).to(device)

            agent.train()

          
    for env in training_envs:                                        
            draw_policy(env, agent, device)
            wandb.log({'train_metrics/PolicyVisualisations': [wandb.Image(plt)]})
            plt.close()

    envs.close()