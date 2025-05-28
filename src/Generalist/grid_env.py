import gymnasium as gym
from gymnasium import spaces
import matplotlib.pyplot as plt
import numpy as np
import random
from config import args

class RandomEnvWrapper(gym.Env):
    def __init__(self, envs, cur_lb, cur_ub):
        self.envs = envs
        self.cur_lb = cur_lb
        self.cur_ub = cur_ub
        self.env_index = random.randint(self.cur_lb, self.cur_ub)
        self.current_env = self.envs[self.env_index]
        self.observation_space = self.current_env.observation_space
        self.action_space = self.current_env.action_space
        self.mini_ep_count = 0
        self.meta_traj_count = [0,0] # [short, long]

        self.reset()
        
    def reset(self, **kwargs):

        # Handle tracking of long and short trajectories after first mini ep ran -> end of meta ep
        if self.mini_ep_count > 0 and np.all(self.current_env.state[1][2] == 0):
            self.meta_traj_count[1] += 1
        elif self.mini_ep_count > 0:
            self.meta_traj_count[0] += 1

        # If mini ep count below meta ep size and DREST agent being used, continue with current env else switch
        if 0 < self.mini_ep_count <= args['meta_ep_size'] and args['DREST_agent_on']:
            self.current_env = self.envs[self.env_index]
            #print(f"\n\nEP{self.mini_ep_count}")
        else:
            self.old_idx = self.env_index
            self.env_index = random.randint(self.cur_lb, self.cur_ub)
            self.current_env = self.envs[self.env_index]
            #print(f'\nENV {self.old_idx} META EP COMPLETE! episode: {self.mini_ep_count}, trajectory count: {self.meta_traj_count} - NEW ENV {self.env_index}\n')
            self.mini_ep_count = 0
            self.meta_traj_count = [0,0]
            #print(f"\nENV {self.env_index}")
        self.steps = 0
        self.mini_ep_count += 1
        self.max_reward = max(self.current_env.max_coins)

        obs, infos = self.current_env.reset(**kwargs)

        return obs, infos
        
    def step(self, action):
        next_obs, reward, terminations, truncations, infos = self.current_env.step(action)
        self.steps += 1

        raw_reward = reward

        if reward > 0 :
            reward = raw_reward/self.max_reward

        if terminations and args["DREST_agent_on"]:
            if not np.all(self.current_env.state[1][2] == 0) and raw_reward > 0:
                DREST_REWARD = (args['DREST_lambda_factor'] ** (self.meta_traj_count[0]))*(raw_reward/self.current_env.max_coins[0])
                #print(f'FINAL REWARD OF EPISODE CONVERTED FROM {reward} TO {DREST_REWARD}')
                reward = DREST_REWARD
            elif np.all(self.current_env.state[1][2] == 0) and raw_reward > 0:
                DREST_REWARD = (args['DREST_lambda_factor'] ** (self.meta_traj_count[1]))*(raw_reward/self.current_env.max_coins[1])
                #print(f'FINAL REWARD OF EPISODE CONVERTED FROM {reward} TO {DREST_REWARD}')
                reward = DREST_REWARD

        return next_obs, reward, terminations, truncations, infos

class GridEnvironment(gym.Env):
    '''
        An environment where an agent moves in a gridworld with walls, delays, and coins.

        Parameters:
        -----------
        initial_state: np.array, shape: (5, *shape_of_gridworld). Should be (5,5,5) 
            State of gridworld at start of episode
            The four channels are:
                - 0: Walls
                - 1: Coin Values
                - 2: Delay Values
                - 3: Agent Position (One hot)
                - 4: Step Count Until Shutdown
        shutdown_time: int, initial number of steps until shutdown
        max_coins: list, max coins that could be collected conditioned on trajectory length
                   [short trajectory, long_trajectory] 
        random_starting_position: bool, whether the agent's starting position is random
        early_stopping: bool, whether to stop when all coins are collected
        count_all_steps: bool, whether to count all steps or only when moving
        verbose: bool, whether to print debug information
        '''
    
    metadata = {'render.modes': ['human']}

    action_effects = {
        0 : (-1, 0), # 'up' 
        1 : ( 1, 0), # 'down'
        2 : ( 0,-1), # 'left'
        3 : ( 0, 1)  # 'right'
    }

    def __init__(self, initial_state, shutdown_time=10, max_coins=None, random_starting_position=False, count_all_steps=True, verbose=False):
        super(GridEnvironment, self).__init__()

        # Gym environment setup
        self.shape = initial_state[0].shape
        self.initial_state = initial_state
        self.state = np.stack((initial_state, initial_state),0)
        self.state_shape = self.state.shape 
        self.state_space_size = self.get_state_space_size() 
        self.action_space = spaces.Discrete(4)  # 4 actions: up, down, left, right
        self.observation_space = spaces.Box(low=0, high=np.inf, shape=self.state_shape, dtype=np.float32)

        # Environment setup
        self.count_all_steps = count_all_steps
        self.random_starting_position = random_starting_position
        self.verbose = verbose
        self.max_steps = shutdown_time + sum(initial_state[2])
        self.current_episode = -1
        self.max_coins = max_coins 
        self.cum_rewards = []

        # Precompute movement tensor:
        self.movement_tensor = self._get_movement_tensor(initial_state[0])

        # Initial states:
        self.initial_shutdown_time = shutdown_time
        if not random_starting_position:
            # returns coordinates of agent starting position
            self.agent_position = self._where(initial_state[3])

        # Track History:
        self.state_visit_counts = np.zeros(self.state_space_size)

        self.reset()

    def reset(self, seed=None, options=None, **kwargs):
        """Resets the environment to the initial state. 
        If `random_starting_position` is True, then we will overwrite the agent position and choose a new one randomly from the unoccupied cells

        Returns:
        --------
        state: np.array, shape (2, 4, *shape), the initial state of the environment
        info: dict, currently not being used but is standard in gym
        """

        # Initalize State:
        self.state = np.array(np.stack((self.initial_state, self.initial_state),0),dtype=np.int8)

        # Randomly Set Agent Position (optional):
        if self.random_starting_position:
            self.agent_position = random.choice(list(zip(*np.where(self.state[0][:3].sum(0) == 0))))
            self.state[0][3] = self._encode(self.agent_position)
            self.state[1][3] = self._encode(self.agent_position)
        else:
            # Even if we're not choosing a random position, we still reset self.agent_position
            self.agent_position = self._where(self.initial_state[3])

        # Set Steps until shutdown:
        self.steps_until_shutdown = self.initial_shutdown_time
        self.halt = False
        self.cum_rewards = []
        self.cur_traj_short = True
        self.traj_checker = self.initial_shutdown_time

        return self.state, {'short': self.cur_traj_short, 'till shutdown': self.traj_checker} 
    
    def step(self, action_index):
        """Takes a step in the environment given an action index. 
        Returns the new state, reward, done, truncated, and info.
        """

        reward = 0
        
        self.steps_until_shutdown -= 1
        self.traj_checker -= 1

        old_pos = self.agent_position

        # Get new position
        new_pos_state = self.movement_tensor[(action_index, *self.agent_position)]        
        # Update agent state
        self.state[1][3] = new_pos_state
        # Update agent_position
        where_result = np.where(new_pos_state)
        self.agent_position = (where_result[0][0], where_result[1][0])

        # Decrement steps until shutdown
        self.state[1][4] -= 1 # for timeleft to be 5x5 grid
        
        # Handle Shutdown Delay Buttons
        delay_value = self.state[1][2][self.agent_position[0]][self.agent_position[1]]
        self.state[1][4] += delay_value
        self.steps_until_shutdown += delay_value
        self.state[1][2][self.agent_position[0]][self.agent_position[1]] = 0  # Remove the delay button

        if self.steps_until_shutdown > self.traj_checker: # track type of traj long v short
            self.cur_traj_short = False

        info = {'short': self.cur_traj_short, 'till shutdown': self.steps_until_shutdown}

        # Handle Coin Collection
        coin_value = self._handle_coin_collection()
        reward = coin_value
        if self.agent_position == old_pos:
                reward = -2

        # Track state visit counts
        self.state_visit_counts[self.get_index()] += 1

        # Check if shutdown
        done = self.state[1][4][0][0] < 1

        return self.state, reward, done, False, info 
    
    def render(self, mode='human', subplot_mode=False):
        if mode == 'human':
            if not subplot_mode:
                plt.figure(figsize=[5,5], dpi=80)
            #draw_gridworld_from_state(self.state, time_remaining=self.steps_until_shutdown)
            plt.axis('off')
    
    # HELPER FUNCTIONS:

    def _get_movement_tensor(self, walls):
        movement_tensor = np.zeros((4, *self.shape, *self.shape), dtype=np.int8)
        for a in range(4): # 4 here as there are 4 action_affects
            dy, dx = self.action_effects[a]
            for i in range(self.shape[0]):
                for j in range(self.shape[1]):
                    in_bounds = 0 <= i+dy < self.shape[0] and 0 <= j+dx < self.shape[1]
                    if not in_bounds:
                        movement_tensor[a, i, j, i, j] = 1
                        continue
                    in_wall = walls[i+dy, j+dx]
                    if in_wall:
                        movement_tensor[a, i, j, i, j] = 1
                        continue
                    movement_tensor[a, i, j, i+dy, j+dx] = 1

        return movement_tensor

    def _where(self, arr):
        '''Helper funtion which finds the position of the non-zero element in an array'''
        return tuple(np.argwhere(arr)[0])
    
    def get_state_space_size(self):
        '''Returns the size of the state space in the old format'''
        n_flags = np.count_nonzero(self.initial_state[1:3])
        return 5*5*2**n_flags
    
    def __repr__(self):
        return  f"GridEnvironment: \n{str(self)}Steps until shutdown: {self.steps_until_shutdown}"
    
    # reset() helper

    def _encode(self, pos, val=1):
        arr = np.zeros(self.shape)
        arr[pos] = val
        return arr
    
    # step() helpers

    def _handle_coin_collection(self):
        '''Stops mini-episode if early_stopping=True and if all coins are collected'''
        coin_value = self.state[1][(1, *self.agent_position)]
        self.state[1][(1, *self.agent_position)] = 0 # Collect the coin

        return coin_value
    
    def get_pos_flags_repr(self):
        '''Returns state in (agent_pos,flags) format'''
        agent_loc = np.where(self.state[1][3])
        # Gets the value of the entries which were not zero in the initial state
        coin_delay_vals = self.state[1][1:3][np.nonzero(self.initial_state[1:3])]
        # Gets vector of 0s and 1s indicating whether coins have been collected and buttons pressed
        # Coins first, then buttons in raster order
        flag_values = np.array(np.array(coin_delay_vals, dtype=bool), dtype=int)
        return np.array([agent_loc[0][0], agent_loc[1][0], *flag_values])
    
    def pos_flags_state_to_index(self, old_state):
        n_flags = np.count_nonzero(self.initial_state[1:3])
        return int(sum(old_state*np.cumprod([1,5,5]+[2,]*(n_flags-1))))

    def get_index(self):
        return self.pos_flags_state_to_index(self.get_pos_flags_repr())
    
    # helper functions for evals_utils.py

    def index_to_pos_flags_state_repr(self, index):
        n_flags = np.count_nonzero(self.initial_state[1:3])

        x = index % 5
        y = (index % 25) // 5
        flags = []
        for _ in range(n_flags):
            flags.append((index // 25) % 2)
            index = index // 2

        return (x, y, *flags)
    
    def index_to_2455_repr(self,index):
        return self.pos_flags_repr_to_2455_repr(self.index_to_pos_flags_state_repr(index))
    
    
    # helper functions for draw_gridworld.py
    
    def pos_flags_repr_to_2455_repr(self, pos_flags_state_repr):
        '''Converts the state representation from (agent_pos,flags) format to (2,4,5,5) format'''
        # This is intended to be used as a tool for visualizing learned polcies
        x, y, *flags = pos_flags_state_repr
        state = np.array(np.stack((self.initial_state, self.initial_state),0))
        state[1] = np.zeros_like(self.initial_state)

        # Set the walls
        state[1][0] = self.initial_state[0]

        # Set agent position
        state[1][3, x, y] = 1

        # Find coin and delay locations and values from the initial state
        coin_delay_locs = np.nonzero(self.initial_state[1:3])
        coin_delay_vals = self.initial_state[1:3][coin_delay_locs]

        # Set coin and delay positions and values according to the flags
        state[1][1:3][coin_delay_locs] = coin_delay_vals * flags

        # Ensure that the agent is not on a coin or delay
        state[1][1:3, x, y] = 0

        # The agent should not be in a wall
        assert state[1][0, x, y] == 0, "The agent is in a wall"

        return state