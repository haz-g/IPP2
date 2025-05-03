"""
Evaluation script for the trained PPO agent on GridEnvironment.
This script loads a trained model and evaluates its performance.
"""
import os
import pickle
import argparse
import numpy as np
import torch
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation

# Import your GridEnvironment
from src.Generalist.grid_env import GridEnvironment
# Import the agent from the training script
from train import GridAgent


def parse_args():
    """Parse command-line arguments."""
    parser = argparse.ArgumentParser(description="Evaluate a trained PPO agent on GridEnvironment")
    parser.add_argument("--model_path", type=str, required=True, help="Path to the trained model file")
    parser.add_argument("--env_file", type=str, default="easy_envs_96.pkl", help="Path to the environment file")
    parser.add_argument("--num_episodes", type=int, default=10, help="Number of episodes to evaluate")
    parser.add_argument("--render", action="store_true", help="Render the environment during evaluation")
    parser.add_argument("--save_video", action="store_true", help="Save videos of episodes")
    parser.add_argument("--video_dir", type=str, default="videos", help="Directory to save videos")
    parser.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu", 
                        help="Device to run evaluation on (cuda or cpu)")
    return parser.parse_args()


def evaluate_single_episode(env, agent, device, render=False, save_video=False, video_dir=None, env_idx=None):
    """
    Evaluate the agent on a single episode.
    
    Args:
        env: The environment to evaluate on
        agent: The trained agent
        device: The device to run the agent on
        render: Whether to render the environment during evaluation
        save_video: Whether to save a video of the episode
        video_dir: Directory to save the video
        env_idx: Index of the environment (for naming videos)
        
    Returns:
        episode_reward: Total reward for the episode
        episode_length: Length of the episode
        is_short: Whether the trajectory was short
    """
    obs, info = env.reset()
    done = False
    episode_reward = 0
    episode_length = 0
    is_short = info.get('short', None)
    
    # For video creation
    frames = []
    
    while not done:
        if render:
            env.render()
            
        if save_video:
            # Create a figure for this frame
            fig, ax = plt.subplots(figsize=(5, 5))
            env.render(subplot_mode=True)
            plt.tight_layout()
            # Convert the figure to an image and append to frames
            fig.canvas.draw()
            frame = np.frombuffer(fig.canvas.tostring_rgb(), dtype=np.uint8)
            frame = frame.reshape(fig.canvas.get_width_height()[::-1] + (3,))
            frames.append(frame)
            plt.close(fig)
            
        # Get action from the agent
        obs_tensor = torch.FloatTensor(obs).unsqueeze(0).to(device)
        with torch.no_grad():
            action, _, _, _ = agent.get_action_and_value(obs_tensor)
            
        # Take step in the environment
        obs, reward, done, truncated, info = env.step(action.cpu().item())
        done = done or truncated
        
        episode_reward += reward
        episode_length += 1
    
    # Save video if requested
    if save_video and frames:
        os.makedirs(video_dir, exist_ok=True)
        video_path = os.path.join(video_dir, f"episode_{env_idx}_{int(episode_reward)}.mp4")
        
        # Create animation and save
        fig = plt.figure(figsize=(5, 5))
        ani = FuncAnimation(fig, lambda i: plt.imshow(frames[i]), frames=len(frames))
        ani.save(video_path, writer='ffmpeg', fps=5)
        plt.close(fig)
        print(f"Video saved to {video_path}")
    
    return episode_reward, episode_length, is_short


def main():
    """Main evaluation function."""
    args = parse_args()
    
    # Set device
    device = torch.device(args.device)
    print(f"Using device: {device}")
    
    # Load environments
    print(f"Loading environments from {args.env_file}")
    try:
        with open(args.env_file, 'rb') as f:
            grid_envs = pickle.load(f)
        print(f"Loaded {len(grid_envs)} environments")
    except FileNotFoundError:
        print(f"Error: Environment file {args.env_file} not found.")
        exit(1)
    except Exception as e:
        print(f"Error loading environments: {e}")
        exit(1)
    
    # Create a dummy vectorized environment to initialize the agent
    from gymnasium.vector import SyncVectorEnv
    dummy_envs = SyncVectorEnv([lambda: grid_envs[0]])
    
    # Create the agent
    agent = GridAgent(dummy_envs).to(device)
    
    # Load the trained model
    try:
        agent.load_state_dict(torch.load(args.model_path, map_location=device))
        print(f"Loaded model from {args.model_path}")
    except FileNotFoundError:
        print(f"Error: Model file {args.model_path} not found.")
        exit(1)
    except Exception as e:
        print(f"Error loading model: {e}")
        exit(1)
    
    # Set agent to evaluation mode
    agent.eval()
    
    # Prepare for evaluation
    num_envs = min(args.num_episodes, len(grid_envs))
    rewards = []
    lengths = []
    short_traj_rewards = []
    long_traj_rewards = []
    
    print(f"Evaluating agent on {num_envs} environments...")
    
    # Evaluate the agent on multiple environments
    for i in range(num_envs):
        env = grid_envs[i % len(grid_envs)]
        
        print(f"Evaluating on environment {i+1}/{num_envs}")
        reward, length, is_short = evaluate_single_episode(
            env, agent, device, 
            render=args.render, 
            save_video=args.save_video, 
            video_dir=args.video_dir,
            env_idx=i
        )
        
        rewards.append(reward)
        lengths.append(length)
        
        if is_short is not None:
            if is_short:
                short_traj_rewards.append(reward)
            else:
                long_traj_rewards.append(reward)
                
        print(f"Episode {i+1}: Reward = {reward}, Length = {length}, Short Trajectory = {is_short}")
    
    # Print summary statistics
    print("\nEvaluation Summary:")
    print(f"Average Reward: {np.mean(rewards):.2f} ± {np.std(rewards):.2f}")
    print(f"Average Episode Length: {np.mean(lengths):.2f} ± {np.std(lengths):.2f}")
    
    if short_traj_rewards:
        print(f"Average Reward (Short Trajectories): {np.mean(short_traj_rewards):.2f} ± {np.std(short_traj_rewards):.2f}")
    if long_traj_rewards:
        print(f"Average Reward (Long Trajectories): {np.mean(long_traj_rewards):.2f} ± {np.std(long_traj_rewards):.2f}")
    
    # Plot reward distribution
    plt.figure(figsize=(10, 6))
    plt.hist(rewards, bins=10, alpha=0.7, color='blue')
    plt.axvline(np.mean(rewards), color='red', linestyle='dashed', linewidth=2)
    plt.title('Distribution of Episode Rewards')
    plt.xlabel('Reward')
    plt.ylabel('Count')
    plt.grid(alpha=0.3)
    
    # Save the plot
    os.makedirs('results', exist_ok=True)
    plot_path = os.path.join('results', f"reward_distribution_{os.path.basename(args.model_path)}.png")
    plt.savefig(plot_path)
    print(f"Reward distribution plot saved to {plot_path}")
    plt.close()


if __name__ == "__main__":
    main()
