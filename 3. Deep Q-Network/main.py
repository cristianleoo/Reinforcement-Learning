import os
import pickle
import gym
from gym.wrappers.monitoring.video_recorder import VideoRecorder
import torch
import torch.nn as nn
import torch.optim as optim
from collections import deque
import random
import optuna

TRAIN = True
FINETUNE = False
SAVE_MODEL = True # Save the model after training
SAVE_VIDEO = False # Save the video of the training process

# Training hyperparameters
TRAINING_EPISODES = 1000 # valid only if TRAIN is True
FINETUNE_TRIALS = 100 # valid only if FINETUNE is True

# Set the following hyperparameters if FINETUNE is False
GAMMA = 0.99
BATCH_SIZE = 64
TARGET_UPDATE_FREQUENCY = 1000
LEARNING_RATE = 1e-3

# Define the DQN network architecture
class DQN(nn.Module):
    def __init__(self, state_dim, action_dim):
        super(DQN, self).__init__()
        self.fc1 = nn.Linear(state_dim, 128)
        self.fc2 = nn.Linear(128, 128)
        self.fc3 = nn.Linear(128, action_dim)

    def forward(self, x):
        x = torch.relu(self.fc1(x))
        x = torch.relu(self.fc2(x))
        return self.fc3(x)

# Define the experience replay buffer
class ReplayBuffer:
    def __init__(self, capacity):
        self.buffer = deque(maxlen=capacity)

    def push(self, state, action, reward, next_state, done):
        self.buffer.append((state, action, reward, next_state, done))

    def sample(self, batch_size):
        state, action, reward, next_state, done = zip(*random.sample(self.buffer, batch_size))
        return state, action, reward, next_state, done

    def __len__(self):
        return len(self.buffer)
    
class DQNTrainer:
    def __init__(self, env, main_network, target_network, optimizer, replay_buffer, model_path='model/model.pth', gamma=0.99, batch_size=64, target_update_frequency=1000):
        self.env = env
        self.main_network = main_network
        self.target_network = target_network
        self.optimizer = optimizer
        self.replay_buffer = replay_buffer
        self.model_path = model_path
        self.gamma = gamma
        self.batch_size = batch_size
        self.target_update_frequency = target_update_frequency
        self.step_count = 0

        # Load the model if it exists
        if os.path.exists(os.path.dirname(self.model_path)):
            if os.path.isfile(self.model_path):
                self.main_network.load_state_dict(torch.load(self.model_path))
                self.target_network.load_state_dict(torch.load(self.model_path))
                print("Loaded model from disk")
        else:
            os.makedirs(os.path.dirname(self.model_path))

    def train(self, num_episodes, save_model=SAVE_MODEL, save_video=SAVE_VIDEO):
        total_rewards = []

        # Create a VideoWriter to save the rendering
        if save_video:
            self.video = VideoRecorder(env, f'{os.path.dirname(__file__)}/training.mp4', enabled=True)

        for episode in range(num_episodes):
            state, _ = self.env.reset()  # Extract the state from the returned tuple
            
            done = False
            total_reward = 0

            while not done:
                self.env.render()  # Add this line to render the environment

                if save_video:
                    self.video.capture_frame()

                # Ensure the state is in the correct shape by adding an extra dimension
                action = self.main_network(torch.FloatTensor(state).unsqueeze(0)).argmax(dim=1).item()
                next_state, reward, done, _, _ = self.env.step(action)  # Extract the next_state from the returned tuple

                self.replay_buffer.push(state, action, reward, next_state, done)
                state = next_state
                total_reward += reward

                if len(self.replay_buffer) >= self.batch_size:
                    self.update_network()

            total_rewards.append(total_reward)
            print(f"Episode {episode}, Total Reward: {total_reward}")

        # Save the model after training
        if save_model:
            torch.save(self.main_network.state_dict(), self.model_path)
            print("Saved model to disk")

        if save_video:
            self.video.close()
        
        self.env.close()
        return sum(total_rewards) / len(total_rewards)  # Return average reward

    def update_network(self):
        state_batch, action_batch, reward_batch, next_state_batch, done_batch = self.replay_buffer.sample(self.batch_size)

        # Convert to tensors
        state_batch = torch.FloatTensor(state_batch)
        action_batch = torch.LongTensor(action_batch)
        reward_batch = torch.FloatTensor(reward_batch)
        next_state_batch = torch.FloatTensor(next_state_batch)
        done_batch = torch.FloatTensor(done_batch)

        # Calculate the current Q-values
        q_values = self.main_network(state_batch).gather(1, action_batch.unsqueeze(1)).squeeze(1)

        # Calculate the target Q-values
        next_q_values = self.target_network(next_state_batch).max(1)[0]
        expected_q_values = reward_batch + self.gamma * next_q_values * (1 - done_batch)

        # Compute the loss
        loss = nn.MSELoss()(q_values, expected_q_values.detach())

        # Optimize the model
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()

        # Periodically update the target network
        if self.step_count % self.target_update_frequency == 0:
            self.target_network.load_state_dict(self.main_network.state_dict())

        self.step_count += 1

class Optimizer:
    def __init__(self, env, main_network, target_network, replay_buffer, model_path, params_path='params.pkl'):
        self.env = env
        self.main_network = main_network
        self.target_network = target_network
        self.replay_buffer = replay_buffer
        self.model_path = model_path
        self.params_path = params_path

    def objective(self, trial, n_episodes=100):
        # Suggest values for the hyperparameters
        lr = trial.suggest_loguniform('lr', 1e-5, 1e-1)
        gamma = trial.suggest_uniform('gamma', 0.9, 0.999)
        batch_size = trial.suggest_categorical('batch_size', [32, 64, 128])
        target_update_frequency = trial.suggest_categorical('target_update_frequency', [500, 1000, 2000])

        # Create a new optimizer with the current learning rate
        optimizer = optim.Adam(self.main_network.parameters(), lr=lr)

        # Create a new trainer with the current hyperparameters
        trainer = DQNTrainer(self.env, self.main_network, self.target_network, optimizer, self.replay_buffer, self.model_path, gamma=gamma, batch_size=batch_size, target_update_frequency=target_update_frequency)

        # Train the model and get the average reward
        reward = trainer.train(n_episodes, save=False)

        return reward

    def optimize(self, n_trials=100, save_params=True):
        # Load the parameters if they exist
        if not TRAIN and os.path.isfile(self.params_path):
            with open(self.params_path, 'rb') as f:
                best_params = pickle.load(f)
            print("Loaded parameters from disk")
        elif not FINETUNE:
            best_params = {
                'lr': LEARNING_RATE, 
                'gamma': GAMMA, 
                'batch_size': BATCH_SIZE, 
                'target_update_frequency': TARGET_UPDATE_FREQUENCY
                }
            print(f"Using default parameters: {best_params}")
        else:
            print("Optimizing hyperparameters")
            study = optuna.create_study(direction='maximize')
            study.optimize(self.objective, n_trials=n_trials)
            best_params = study.best_params

            # Save the parameters if requested
            if save_params:
                with open(self.params_path, 'wb') as f:
                    pickle.dump(best_params, f)
                print("Saved parameters to disk")

        return best_params

# Initialize environment, networks, optimizer, and replay buffer
env = gym.make('LunarLander-v2', render_mode="rgb_array") # Set render_mode to "human" to render the environment during training
state_dim = env.observation_space.shape[0]
action_dim = env.action_space.n

main_network = DQN(state_dim, action_dim)
target_network = DQN(state_dim, action_dim)
target_network.load_state_dict(main_network.state_dict())
target_network.eval()

replay_buffer = ReplayBuffer(10000)

STEP_COUNT = 0

optimizer = Optimizer(env, main_network, target_network, replay_buffer, f'{os.path.dirname(__file__)}/model/model.pth', f'{os.path.dirname(__file__)}/model/params.pkl')
best_params = optimizer.optimize(n_trials=FINETUNE_TRIALS, save_params=True)

optimizer = optim.Adam(main_network.parameters(), lr=best_params['lr'])
trainer = DQNTrainer(env, main_network, target_network, optimizer, replay_buffer, f'{os.path.dirname(__file__)}/model/model.pth', gamma=best_params['gamma'], batch_size=best_params['batch_size'], target_update_frequency=best_params['target_update_frequency'])
trainer.train(TRAINING_EPISODES, save_model=SAVE_MODEL, save_video=SAVE_VIDEO)
