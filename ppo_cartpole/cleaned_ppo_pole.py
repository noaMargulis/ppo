import gymnasium as gym
import torch
import torch.nn as nn
import torch.optim as optim
from torch.distributions import Categorical
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation
from collections import deque
import os
import random

# Set seeds for reproducibility
torch.manual_seed(42)
np.random.seed(42)
random.seed(42)

# Make sure bool8 compatibility is preserved for older NumPy versions
np.bool8 = bool
gir
ENV_NAME = "CartPole-v1"  # Changed environment to CartPole-v1
GAMMA = 0.99
LAMBDA = 0.95
EPS_CLIP = 0.2
K_EPOCHS = 6
LR_ACTOR = 3e-4
LR_CRITIC = 3e-4
BATCH_SIZE = 64
TIMESTEPS_PER_BATCH = 256  # Reduced for CartPole as episodes are shorter
MAX_EPISODES = 1000  # Reduced max episodes as CartPole converges faster
EVAL_EVERY = 20  # Evaluate more frequently for faster feedback
MAX_GRAD_NORM = 0.5
VF_COEF = 0.5
ENT_COEF = 0.01
TARGET_KL = 0.01


class ActorCritic(nn.Module):
    """
    ActorCritic network for PPO.
    It shares a common feature extractor and has separate heads for policy (actor)
    and value (critic).
    """

    def __init__(self, state_dim, action_dim, hidden_dim=64):
        super().__init__()

        # Shared feature extractor
        self.shared = nn.Sequential(
            nn.Linear(state_dim, hidden_dim),
            nn.Tanh(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.Tanh()
        )

        # Policy head (outputs logits for action probabilities)
        self.policy = nn.Linear(hidden_dim, action_dim)

        # Value head (outputs a single value estimate)
        self.value = nn.Linear(hidden_dim, 1)

        # Initialize weights for better training stability
        self.apply(self._init_weights)

    def _init_weights(self, m):
        """
        Initializes weights of linear layers using orthogonal initialization.
        Policy output layer uses a smaller gain for more stable initial exploration.
        """
        if isinstance(m, nn.Linear):
            nn.init.orthogonal_(m.weight, gain=np.sqrt(2))
            nn.init.constant_(m.bias, 0.0)

        # Special initialization for policy output layer
        # A smaller gain (0.01) helps keep initial policy probabilities close to uniform
        # preventing extreme actions early in training.
        nn.init.orthogonal_(self.policy.weight, gain=0.01)
        nn.init.constant_(self.policy.bias, 0.0)

    def forward(self, x):
        """
        Forward pass through the shared layers and then the policy and value heads.
        """
        shared = self.shared(x)
        return self.policy(shared), self.value(shared)

    def get_action_and_value(self, x, action=None):
        """
        Given state(s) `x`, returns an action, its log probability, entropy, and value estimate.
        If `action` is provided, it calculates log_prob for that specific action.
        """
        logits, value = self(x)
        probs = Categorical(logits=logits)
        if action is None:
            action = probs.sample()  # Sample action from the distribution
        return action, probs.log_prob(action), probs.entropy(), value


class PPOAgent:
    """
    Proximal Policy Optimization (PPO) agent implementation.
    Handles collecting trajectories, computing advantages, and updating the policy and value networks.
    """

    def __init__(self, state_dim, action_dim):
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        print(f"Using device: {self.device}")

        self.agent = ActorCritic(state_dim, action_dim).to(self.device)
        # Adam optimizer for both actor and critic, using a single learning rate
        self.optimizer = optim.Adam(self.agent.parameters(), lr=LR_ACTOR, eps=1e-5)

        # For tracking training statistics over updates
        self.training_stats = {
            'policy_loss': [],
            'value_loss': [],
            'entropy': [],
            'approx_kl': [],
            'clipfrac': []
        }

    def get_action(self, state):
        """
        Selects an action based on the current policy for a given state.
        Returns the action, its log probability, and the predicted value.
        """
        with torch.no_grad():  # No gradient calculation needed for action selection
            state = torch.FloatTensor(state).unsqueeze(0).to(self.device)
            action, logprob, _, value = self.agent.get_action_and_value(state)
            # Ensure 'value' is a scalar Python float using .squeeze().item()
            # This prevents issues with numpy arrays of shape (1,) being appended to lists,
            # which can lead to unexpected 2D numpy arrays and subsequent tensor shape mismatches.
            return action.cpu().numpy()[0], logprob.cpu().numpy()[0], value.squeeze().cpu().numpy().item()

    def update(self, states, actions, logprobs, rewards, dones, values):
        """
        Performs a PPO update using collected trajectories.
        Calculates advantages using GAE, computes losses, and updates network parameters.
        """
        # Convert collected numpy arrays to PyTorch tensors and move to device
        states = torch.FloatTensor(states).to(self.device)
        actions = torch.LongTensor(actions).to(self.device)
        logprobs = torch.FloatTensor(logprobs).to(self.device)
        rewards = torch.FloatTensor(rewards).to(self.device)
        # Convert boolean dones to float for calculations (0.0 or 1.0)
        dones = torch.FloatTensor(dones.astype(np.float32)).to(self.device)
        values = torch.FloatTensor(values).to(self.device)

        # Compute advantages using Generalized Advantage Estimation (GAE)
        advantages = torch.zeros_like(rewards).to(self.device)
        lastgaelam = 0

        # Iterate backwards to compute GAE
        for t in reversed(range(len(rewards))):
            if t == len(rewards) - 1:
                nextnonterminal = 1.0 - dones[t]  # 0 if done, 1 if not done
                nextvalues = 0  # If last step, no next value
            else:
                nextnonterminal = 1.0 - dones[t + 1]
                nextvalues = values[t + 1]

            # TD error calculation
            delta = rewards[t] + GAMMA * nextvalues * nextnonterminal - values[t]
            # GAE formula
            advantages[t] = lastgaelam = delta + GAMMA * LAMBDA * nextnonterminal * lastgaelam

        # Compute returns (target values for the critic)
        returns = advantages + values

        # Normalize advantages for better training stability
        advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-8)

        # Optimizing the policy and value network for K_EPOCHS
        batch_size = states.shape[0]  # Total number of timesteps collected

        clipfracs = []  # To track the fraction of clipped policy updates

        for epoch in range(K_EPOCHS):
            # Create random indices for mini-batches to shuffle data
            indices = torch.randperm(batch_size)

            # Iterate through mini-batches
            for start in range(0, batch_size, BATCH_SIZE):
                end = min(start + BATCH_SIZE, batch_size)
                mb_indices = indices[start:end]  # Indices for the current mini-batch

                # Get mini-batch data
                mb_states = states[mb_indices]
                mb_actions = actions[mb_indices]
                mb_logprobs = logprobs[mb_indices]
                mb_advantages = advantages[mb_indices]
                mb_returns = returns[mb_indices]

                # Forward pass: get new log probabilities, entropy, and value estimates
                _, newlogprob, entropy, newvalue = self.agent.get_action_and_value(
                    mb_states, mb_actions
                )

                # Calculate ratio of new policy probability to old policy probability
                logratio = newlogprob - mb_logprobs
                ratio = logratio.exp()

                with torch.no_grad():
                    # Calculate approximate KL divergence for early stopping and logging
                    # old_approx_kl is a common way to estimate KL for PPO
                    old_approx_kl = (-logratio).mean()
                    approx_kl = ((ratio - 1) - logratio).mean()
                    # Calculate clip fraction: how many updates were actually clipped
                    clipfracs += [((ratio - 1.0).abs() > EPS_CLIP).float().mean().item()]

                # Policy loss (PPO's clipped objective)
                # First term: unclipped advantage * ratio
                pg_loss1 = -mb_advantages * ratio
                pg_loss2 = -mb_advantages * torch.clamp(ratio, 1 - EPS_CLIP, 1 + EPS_CLIP)
                # Take the minimum of the two terms to ensure the clipped objective
                pg_loss = torch.max(pg_loss1, pg_loss2).mean()

                # Value loss (Mean Squared Error between new value estimates and returns)
                newvalue = newvalue.view(-1)  # Flatten value estimates to 1D
                # Ensure mb_returns is also 1D to match newvalue for element-wise subtraction.
                mb_returns = mb_returns.view(-1)

                v_loss_unclipped = (newvalue - mb_returns) ** 2
                v_loss = v_loss_unclipped.mean()

                # Entropy loss (encourages exploration)
                entropy_loss = entropy.mean()

                # Total loss for backpropagation
                # Policy loss - Entropy bonus + Value function loss
                loss = pg_loss - ENT_COEF * entropy_loss + VF_COEF * v_loss

                # Optimize the network
                self.optimizer.zero_grad()  # Clear previous gradients
                loss.backward()  # Compute gradients
                # Clip gradients to prevent exploding gradients
                nn.utils.clip_grad_norm_(self.agent.parameters(), MAX_GRAD_NORM)
                self.optimizer.step()  # Update network weights

            # Early stopping based on KL divergence
            # If KL divergence exceeds target, stop further updates for this batch
            if approx_kl > TARGET_KL:
                break

        # Update training statistics for logging and plotting
        self.training_stats['policy_loss'].append(pg_loss.item())
        self.training_stats['value_loss'].append(v_loss.item())
        self.training_stats['entropy'].append(entropy_loss.item())
        self.training_stats['approx_kl'].append(approx_kl.item())
        self.training_stats['clipfrac'].append(np.mean(clipfracs))  # Average clip fraction over mini-batches

        return {
            'policy_loss': pg_loss.item(),
            'value_loss': v_loss.item(),
            'entropy': entropy_loss.item(),
            'approx_kl': approx_kl.item(),
            'clipfrac': np.mean(clipfracs)
        }

    def save_model(self, filepath):
        """Saves the agent's model and optimizer state, along with training statistics."""
        torch.save({
            'model_state_dict': self.agent.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'training_stats': self.training_stats
        }, filepath)

    def load_model(self, filepath):
        """Loads the agent's model, optimizer state, and training statistics from a file."""
        checkpoint = torch.load(filepath, map_location=self.device)
        self.agent.load_state_dict(checkpoint['model_state_dict'])
        self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        if 'training_stats' in checkpoint:
            self.training_stats = checkpoint['training_stats']


def collect_trajectories(agent, env, timesteps):
    """
    Collects a batch of trajectories from the environment using the current policy.
    Continues interacting with the environment until `timesteps` are collected.
    """
    states = []
    actions = []
    logprobs = []
    rewards = []
    dones = []
    values = []

    obs, _ = env.reset()  # Reset environment at the start of collection
    episode_reward = 0
    episode_length = 0
    step = 0

    while step < timesteps:
        # Get action, logprob, and value from the agent
        action, logprob, value = agent.get_action(obs)

        # Store collected data
        states.append(obs)
        actions.append(action)
        logprobs.append(logprob)
        values.append(value)  # value is now a scalar float

        # Take a step in the environment
        obs, reward, terminated, truncated, _ = env.step(action)
        done = terminated or truncated  # Episode ends if terminated or truncated

        rewards.append(reward)
        dones.append(done)

        episode_reward += reward
        episode_length += 1
        step += 1

        if done:
            # If episode ends, reset environment and reset episode tracking variables
            obs, _ = env.reset()
            episode_reward = 0
            episode_length = 0

    # Convert lists of collected data to numpy arrays
    states_np = np.array(states)
    actions_np = np.array(actions)
    logprobs_np = np.array(logprobs)
    rewards_np = np.array(rewards)
    dones_np = np.array(dones)
    values_np = np.array(values)  # This will now be (TIMESTEPS_PER_BATCH,)

    return (
        states_np,
        actions_np,
        logprobs_np,
        rewards_np,
        dones_np,
        values_np
    )


def evaluate_agent(agent, env_name, num_episodes=5):
    """
    Evaluates the agent's performance over a specified number of episodes.
    Returns the mean and standard deviation of total rewards.
    """
    env = gym.make(env_name)
    total_rewards = []

    for _ in range(num_episodes):
        obs, _ = env.reset()
        episode_reward = 0
        done = False

        while not done:
            action, _, _ = agent.get_action(obs)  # Only need action for evaluation
            obs, reward, terminated, truncated, _ = env.step(action)
            done = terminated or truncated
            episode_reward += reward

        total_rewards.append(episode_reward)
    env.close()  # Close environment after evaluation
    return np.mean(total_rewards), np.std(total_rewards)


def evaluate_and_record(agent, env_name="CartPole-v1", filename="ppo_video.mp4"):
    """
    Records a video of the agent playing in the environment.
    """
    os.makedirs("mp4_files", exist_ok=True)  # Ensure directory exists
    full_path = os.path.join("mp4_files", filename)

    # Create environment with rgb_array render mode for video recording
    env = gym.make(env_name, render_mode="rgb_array")
    obs, _ = env.reset()
    frames = []
    done = False

    while not done:
        frame = env.render()  # Render current frame
        frames.append(frame)

        action, _, _ = agent.get_action(obs)
        obs, _, terminated, truncated, _ = env.step(action)
        done = terminated or truncated
    env.close()  # Close environment after recording

    # Save video using matplotlib.animation
    fig = plt.figure(figsize=(6, 6))
    plt.axis("off")  # Turn off axes for cleaner video
    im = plt.imshow(frames[0])

    def update(i):
        """Update function for animation."""
        im.set_array(frames[i])
        return [im]

    # Create animation and save it
    anim = animation.FuncAnimation(fig, update, frames=len(frames), interval=50)  # 50ms interval = 20 FPS
    anim.save(full_path, writer="ffmpeg")  # Requires ffmpeg to be installed
    plt.close(fig)  # Close the plot to free memory
    print(f"Saved video to {full_path}")


def plot_training_progress(episode_rewards, eval_rewards, training_stats, episode_num=None):
    """
    Plots comprehensive training progress including episode rewards, evaluation rewards,
    and various PPO training statistics (losses, entropy, KL, clip fraction).
    """
    # Create the 'plots' directory if it doesn't exist
    plots_dir = "plots"
    os.makedirs(plots_dir, exist_ok=True)

    fig, axes = plt.subplots(2, 3, figsize=(
    18, 12))  # Create a 2x3 grid of subplots, increased figure size for better readability

    # Plot 1: Episode rewards
    axes[0, 0].plot(episode_rewards, alpha=0.4, label='Episode Rewards', color='skyblue')
    if len(episode_rewards) > 20:  # Plot moving average if enough data
        window = 20
        moving_avg = [np.mean(episode_rewards[max(0, i - window):i + 1]) for i in range(len(episode_rewards))]
        axes[0, 0].plot(moving_avg, label=f'Moving Average ({window} episodes)', color='blue', linewidth=2)
    axes[0, 0].set_xlabel('Episode', fontsize=12)
    axes[0, 0].set_ylabel('Reward', fontsize=12)
    axes[0, 0].set_title('Training Rewards per Episode', fontsize=14)
    axes[0, 0].legend(fontsize=10)
    axes[0, 0].grid(True, linestyle='--', alpha=0.7)
    axes[0, 0].tick_params(axis='both', which='major', labelsize=10)

    # Plot 2: Evaluation rewards
    if eval_rewards:
        eval_episodes = [i * EVAL_EVERY for i in range(len(eval_rewards))]
        axes[0, 1].plot(eval_episodes, eval_rewards, 'o-', label='Evaluation Rewards', color='forestgreen',
                        markersize=5)
        axes[0, 1].set_xlabel('Episode', fontsize=12)
        axes[0, 1].set_ylabel('Average Reward', fontsize=12)
        axes[0, 1].set_title('Evaluation Performance', fontsize=14)
        axes[0, 1].legend(fontsize=10)
        axes[0, 1].grid(True, linestyle='--', alpha=0.7)
        axes[0, 1].tick_params(axis='both', which='major', labelsize=10)

    # Plot 3: Training losses
    if training_stats['policy_loss']:
        axes[0, 2].plot(training_stats['policy_loss'], label='Policy Loss', color='darkorange')
        axes[0, 2].plot(training_stats['value_loss'], label='Value Loss', color='purple')
        axes[0, 2].set_xlabel('Update Step', fontsize=12)
        axes[0, 2].set_ylabel('Loss Value', fontsize=12)
        axes[0, 2].set_title('Policy and Value Losses', fontsize=14)
        axes[0, 2].legend(fontsize=10)
        axes[0, 2].grid(True, linestyle='--', alpha=0.7)
        axes[0, 2].tick_params(axis='both', which='major', labelsize=10)

    # Plot 4: Entropy
    if training_stats['entropy']:
        axes[1, 0].plot(training_stats['entropy'], label='Policy Entropy', color='crimson')
        axes[1, 0].set_xlabel('Update Step', fontsize=12)
        axes[1, 0].set_ylabel('Entropy', fontsize=12)
        axes[1, 0].set_title('Policy Entropy', fontsize=14)
        axes[1, 0].legend(fontsize=10)
        axes[1, 0].grid(True, linestyle='--', alpha=0.7)
        axes[1, 0].tick_params(axis='both', which='major', labelsize=10)

    # Plot 5: KL divergence
    if training_stats['approx_kl']:
        axes[1, 1].plot(training_stats['approx_kl'], label='Approx. KL Divergence', color='teal')
        # Add a horizontal line for the target KL divergence
        axes[1, 1].axhline(y=TARGET_KL, color='red', linestyle='--', label=f'Target KL ({TARGET_KL})', linewidth=1.5)
        axes[1, 1].set_xlabel('Update Step', fontsize=12)
        axes[1, 1].set_ylabel('KL Divergence', fontsize=12)
        axes[1, 1].set_title('Approximate KL Divergence', fontsize=14)
        axes[1, 1].legend(fontsize=10)
        axes[1, 1].grid(True, linestyle='--', alpha=0.7)
        axes[1, 1].tick_params(axis='both', which='major', labelsize=10)

    # Plot 6: Clip fraction
    if training_stats['clipfrac']:
        axes[1, 2].plot(training_stats['clipfrac'], label='Clip Fraction', color='darkblue')
        axes[1, 2].set_xlabel('Update Step', fontsize=12)
        axes[1, 2].set_ylabel('Fraction', fontsize=12)
        axes[1, 2].set_title('PPO Clip Fraction', fontsize=14)
        axes[1, 2].legend(fontsize=10)
        axes[1, 2].grid(True, linestyle='--', alpha=0.7)
        axes[1, 2].tick_params(axis='both', which='major', labelsize=10)

    plt.tight_layout(pad=3.0)  # Adjust subplot parameters for a tight layout with padding

    # Save the plot in the 'plots' directory with an episode number in the filename
    if episode_num is not None:
        full_plot_path = os.path.join(plots_dir, f"ppo_training_progress_episode_{episode_num}.png")
    else:
        full_plot_path = os.path.join(plots_dir, "ppo_training_progress_final.png")

    plt.savefig(full_plot_path, dpi=200, bbox_inches='tight')  # Increased DPI for higher quality
    plt.close(fig)  # Close the plot to free memory
    print(f"Saved training progress plot to {full_plot_path}")


def train():
    """Main training loop for the PPO agent."""
    # Create directories for saving models, videos, and plots
    os.makedirs("model", exist_ok=True)
    os.makedirs("mp4_files", exist_ok=True)
    os.makedirs("plots", exist_ok=True)  # Ensure 'plots' directory exists

    # Initialize environment and agent
    env = gym.make(ENV_NAME)
    state_dim = env.observation_space.shape[0]
    action_dim = env.action_space.n

    print(f"Environment: {ENV_NAME}")
    print(f"State dimension: {state_dim}")
    print(f"Action dimension: {action_dim}")
    print(f"Device: {'cuda' if torch.cuda.is_available() else 'cpu'}")

    agent = PPOAgent(state_dim, action_dim)

    # Training variables
    episode_rewards = []  # To store rewards for each completed episode
    eval_rewards = []  # To store average rewards from evaluation episodes
    best_eval_reward = float('-inf')  # To track the best performing model

    global_step = 0  # Total timesteps collected across all episodes
    episode = 0  # Current episode number

    print("Starting training...")

    # Main training loop: continues until a total number of timesteps is reached
    while global_step < MAX_EPISODES * TIMESTEPS_PER_BATCH:
        # Collect trajectories from the environment using the current policy
        states, actions, logprobs, rewards, dones, values = collect_trajectories(
            agent, env, TIMESTEPS_PER_BATCH
        )

        # Calculate episode rewards for logging and plotting
        # This loop processes the `dones` array to identify episode boundaries
        episode_reward = 0
        for i, done in enumerate(dones):
            episode_reward += rewards[i]
            if done:
                episode_rewards.append(episode_reward)  # Store total reward for the completed episode
                episode += 1  # Increment episode count

                # Print recent average reward every 10 episodes
                if episode % 10 == 0:
                    recent_rewards = episode_rewards[-10:]  # Average of last 10 episodes
                    print(
                        f"Episode {episode}, Recent avg reward: {np.mean(recent_rewards):.2f}, Global step: {global_step}")
                episode_reward = 0  # Reset for the next episode

        # Update policy and value networks using the collected trajectories
        stats = agent.update(states, actions, logprobs, rewards, dones, values)
        global_step += TIMESTEPS_PER_BATCH  # Increment global timestep count

        # Print detailed training statistics periodically
        if episode % 50 == 0 and episode > 0:
            print(f"Training stats - Policy Loss: {stats['policy_loss']:.4f}, "
                  f"Value Loss: {stats['value_loss']:.4f}, "
                  f"Entropy: {stats['entropy']:.4f}, "
                  f"KL: {stats['approx_kl']:.4f}")

        # Evaluation phase: evaluate agent performance and save models/videos periodically
        if episode % EVAL_EVERY == 0 and episode > 0:
            eval_mean, eval_std = evaluate_agent(agent, ENV_NAME, num_episodes=5)
            eval_rewards.append(eval_mean)  # Store average evaluation reward

            print(f"Evaluation at episode {episode}: {eval_mean:.2f} ± {eval_std:.2f}")

            # Save the model at current evaluation point
            model_path = os.path.join("model", f"ppo_model_episode_{episode}.pth")
            agent.save_model(model_path)

            # Save the best performing model so far
            if eval_mean > best_eval_reward:
                best_eval_reward = eval_mean
                best_model_path = os.path.join("model", "ppo_model_best.pth")
                agent.save_model(best_model_path)
                print(f"New best model saved with reward: {eval_mean:.2f}")

            # Record a video of the agent playing
            evaluate_and_record(agent, ENV_NAME, filename=f"eval_ep{episode}.mp4")

            # Save the training progress plot at this evaluation point
            plot_training_progress(episode_rewards, eval_rewards, agent.training_stats, episode_num=episode)

    # Final model save after training completes
    final_model_path = os.path.join("model", "ppo_model_final.pth")
    agent.save_model(final_model_path)
    print(f"Final model saved to {final_model_path}")

    # Final evaluation and video recording
    evaluate_and_record(agent, ENV_NAME, filename="eval_final.mp4")

    # Plot the overall training progress (final plot)
    plot_training_progress(episode_rewards, eval_rewards, agent.training_stats, episode_num="final")

    env.close()  # Close the environment
    print("Training completed!")

    return agent, episode_rewards, eval_rewards


if __name__ == "__main__":
    agent, episode_rewards, eval_rewards = train()
