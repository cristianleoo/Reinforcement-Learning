import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation
import pickle
import os

# GridWorld Environment
class GridWorld:
    """GridWorld environment with obstacles and a goal.
    The agent starts at the top-left corner and has to reach the bottom-right corner.
    The agent receives a reward of -1 at each step, a reward of -0.01 at each step in an obstacle, and a reward of 1 at the goal.
    
    Args:
        size (int): The size of the grid.
        num_obstacles (int): The number of obstacles in the grid.
        
    Attributes:
        size (int): The size of the grid.
        num_obstacles (int): The number of obstacles in the grid.
        obstacles (list): The list of obstacles in the grid.
        state_space (numpy.ndarray): The state space of the grid.
        state (tuple): The current state of the agent.
        goal (tuple): The goal state of the agent.
    
    Methods:
        generate_obstacles: Generate the obstacles in the grid.
        step: Take a step in the environment.
        reset: Reset the environment.
    """
    def __init__(self, size=5, num_obstacles=5):
        self.size = size
        self.num_obstacles = num_obstacles
        self.obstacles = []
        self.generate_obstacles()
        self.state_space = np.zeros((self.size, self.size))
        self.state = (0, 0)
        self.goal = (self.size-1, self.size-1)

    def generate_obstacles(self):
        """
        Generate the obstacles in the grid.
        The obstacles are generated randomly in the grid, except in the top-left and bottom-right corners.
        
        Args:
            None
        
        Returns:
            None
        """
        for _ in range(self.num_obstacles):
            while True:
                obstacle = (np.random.randint(self.size), np.random.randint(self.size))
                if obstacle not in self.obstacles and obstacle != (0, 0) and obstacle != (self.size-1, self.size-1):
                    self.obstacles.append(obstacle)
                    break

    def step(self, action):
        """
        Take a step in the environment.
        The agent takes a step in the environment based on the action it chooses.

        Args:
            action (int): The action the agent takes.
                0: up
                1: right
                2: down
                3: left
        
        Returns:
            state (tuple): The new state of the agent.
            reward (float): The reward the agent receives.
            done (bool): Whether the episode is done or not.
        """
        x, y = self.state
        if action == 0:  # up
            x = max(0, x-1)
        elif action == 1:  # right
            y = min(self.size-1, y+1)
        elif action == 2:  # down
            x = min(self.size-1, x+1)
        elif action == 3:  # left
            y = max(0, y-1)
        self.state = (x, y)
        if self.state in self.obstacles:
            return self.state, -1, True
        if self.state == self.goal:
            return self.state, 1, True
        return self.state, -0.01, False

    def reset(self):
        """
        Reset the environment.
        The agent is placed back at the top-left corner of the grid.

        Args:
            None
        
        Returns:
            state (tuple): The new state of the agent.
        """
        self.state = (0, 0)
        return self.state

# Q-Learning
class QLearning:
    """
    Q-Learning agent for the GridWorld environment.

    Args:
        env (GridWorld): The GridWorld environment.
        alpha (float): The learning rate.
        gamma (float): The discount factor.
        epsilon (float): The exploration rate.
        episodes (int): The number of episodes to train the agent.
    
    Attributes:
        env (GridWorld): The GridWorld environment.
        alpha (float): The learning rate.
        gamma (float): The discount factor.
        epsilon (float): The exploration rate.
        episodes (int): The number of episodes to train the agent.
        q_table (numpy.ndarray): The Q-table for the agent.
    
    Methods:
        choose_action: Choose an action for the agent to take.
        update_q_table: Update the Q-table based on the agent's experience.
        train: Train the agent in the environment.
        save_q_table: Save the Q-table to a file.
        load_q_table: Load the Q-table from a file.
    """
    def __init__(self, env, alpha=0.5, gamma=0.95, epsilon=0.1, episodes=10):
        self.env = env
        self.alpha = alpha
        self.gamma = gamma
        self.epsilon = epsilon
        self.episodes = episodes
        self.q_table = np.zeros((self.env.size, self.env.size, 4))

    def choose_action(self, state):
        """
        Choose an action for the agent to take.
        The agent chooses an action based on the epsilon-greedy policy.
        
        Args:
            state (tuple): The current state of the agent.
        
        Returns:
            action (int): The action the agent takes.
                0: up
                1: right
                2: down
                3: left
        """
        if np.random.uniform(0, 1) < self.epsilon:
            return np.random.choice([0, 1, 2, 3])  # exploration
        else:
            return np.argmax(self.q_table[state])  # exploitation

    def update_q_table(self, state, action, reward, new_state):
        """
        Update the Q-table based on the agent's experience.
        The Q-table is updated based on the Q-learning update rule.

        Args:
            state (tuple): The current state of the agent.
            action (int): The action the agent takes.
            reward (float): The reward the agent receives.
            new_state (tuple): The new state of the agent.

        Returns:
            None
        """
        self.q_table[state][action] = (1 - self.alpha) * self.q_table[state][action] + \
            self.alpha * (reward + self.gamma * np.max(self.q_table[new_state]))

    def train(self):
        """
        Train the agent in the environment.
        The agent is trained in the environment for a number of episodes.
        The agent's experience is stored and returned.

        Args:
            None
        
        Returns:
            rewards (list): The rewards the agent receives at each step.
            states (list): The states the agent visits at each step.
            starts (list): The start of each new episode.
            steps_per_episode (list): The number of steps the agent takes in each episode.
        """
        rewards = []
        states = []  # Store states at each step
        starts = []  # Store the start of each new episode
        steps_per_episode = []  # Store the number of steps per episode
        steps = 0  # Initialize the step counter outside the episode loop
        episode = 0
        while episode < self.episodes:
            state = self.env.reset()
            total_reward = 0
            done = False
            while not done:
                action = self.choose_action(state)
                new_state, reward, done = self.env.step(action)
                self.update_q_table(state, action, reward, new_state)
                state = new_state
                total_reward += reward
                states.append(state)  # Store state
                steps += 1  # Increment the step counter
                if done and state == self.env.goal:  # Check if the agent has reached the goal
                    starts.append(len(states))  # Store the start of the new episode
                    rewards.append(total_reward)
                    steps_per_episode.append(steps)  # Store the number of steps for this episode
                    steps = 0  # Reset the step counter
                    episode += 1
        return rewards, states, starts, steps_per_episode
    
    def save_q_table(self, filename):
        """
        Save the Q-table to a file.
        
        Args:
            filename (str): The name of the file to save the Q-table to.
            
        Returns:
            None
        """
        filename = os.path.join(os.path.dirname(__file__), filename)
        with open(filename, 'wb') as f:
            pickle.dump(self.q_table, f)

    def load_q_table(self, filename):
        """
        Load the Q-table from a file.

        Args:
            filename (str): The name of the file to load the Q-table from.
        
        Returns:
            None
        """
        filename = os.path.join(os.path.dirname(__file__), filename)
        with open(filename, 'rb') as f:
            self.q_table = pickle.load(f)


if __name__ == '__main__':
    # Initialize environment and agent
    for i in range(10):
        env = GridWorld(size=5, num_obstacles=5)
        agent = QLearning(env)

        # Load the Q-table if it exists
        if os.path.exists(os.path.join(os.path.dirname(__file__), 'q_table.pkl')):
            agent.load_q_table('q_table.pkl')

        # Train the agent and get rewards
        rewards, states, starts, steps_per_episode = agent.train()  # Get starts and steps_per_episode as well

        # Save the Q-table
        agent.save_q_table('q_table.pkl')

        # Visualize the agent moving in the grid
        fig, ax = plt.subplots()

        def update(i):
            """
            Update the grid with the agent's movement.
            
            Args:
                i (int): The current step.
            
            Returns:
                None
            """
            ax.clear()
            # Calculate the cumulative reward up to the current step
            cumulative_reward = sum(rewards[:i+1])
            # Find the current episode
            current_episode = next((j for j, start in enumerate(starts) if start > i), len(starts)) - 1
            # Calculate the number of steps since the start of the current episode
            if current_episode < 0:
                steps = i + 1
            else:
                steps = i - starts[current_episode] + 1
            ax.set_title(f"Iteration: {current_episode+1}, Total Reward: {cumulative_reward:.2f}, Steps: {steps}")
            grid = np.zeros((env.size, env.size))
            for obstacle in env.obstacles:
                grid[obstacle] = -1
            grid[env.goal] = 1
            grid[states[i]] = 0.5  # Use states[i] instead of env.state
            ax.imshow(grid, cmap='cool')

        # Global scope
        # ani = None

        # Inside your function or loop
        global ani
        ani = animation.FuncAnimation(fig, update, frames=range(len(states)), repeat=False)

        # After the animation
        print(f"Environment number {i+1}")
        for i, steps in enumerate(steps_per_episode, 1):
            print(f"Iteration {i}: {steps} steps")
        print(f"Total reward: {sum(rewards):.2f}")
        print()

        # Save the animation
        ani.save(os.path.join(os.path.dirname(__file__), f'animation_{i+1}.gif'), writer='pillow', fps=2)