# evaluate.py

import argparse
import os
import csv
import gymnasium as gym
from stable_baselines3 import PPO, A2C

# Import environments
from envs.web_flow.web_env import WebFlowEnv
from envs.flappy_game.flappy_env import FlappyEnv


def make_env(app, persona):
    """Return the correct environment instance."""
    if app == "web":
        return WebFlowEnv(persona=persona)
    elif app == "flappy":
        return FlappyEnv(persona=persona)
    else:
        raise ValueError("Unknown app: choose 'web' or 'flappy'")


def evaluate(model_path, episodes=30):
    """
    Evaluate a trained model and store episode-level metrics into logs/.
    """
    print("\nLoading model from:", model_path)

    # Detect algorithm type
    algo_name = "ppo" if "ppo" in model_path.lower() else "a2c"
    persona = "explorer" if "explorer" in model_path.lower() else "survivor"
    app = "web" if "web" in model_path.lower() else "flappy"

    # Choose model class
    ModelClass = PPO if algo_name == "ppo" else A2C

    # Load model
    model = ModelClass.load(model_path)

    # Create environment
    env = make_env(app, persona)

    # Prepare logging
    os.makedirs("logs", exist_ok=True)
    csv_path = f"logs/{app}_{algo_name}_{persona}_eval.csv"

    print("Saving evaluation to:", csv_path)

    with open(csv_path, "w", newline="") as f:
        writer = csv.writer(f)

        if app == "web":
            writer.writerow(["episode", "total_reward", "final_page", "error_flag"])
        else:
            writer.writerow(["episode", "total_reward", "pipes_passed", "death_reason"])

        # Run evaluation episodes
        for ep in range(1, episodes + 1):
            obs, _ = env.reset()
            done = False
            truncated = False
            total_reward = 0
            final_info = {}

            while not done and not truncated:
                action, _ = model.predict(obs)
                obs, reward, done, truncated, info = env.step(action)
                total_reward += reward
                final_info = info

            # Web logs
            if app == "web":
                writer.writerow([
                    ep,
                    round(total_reward, 3),
                    env.current_page,
                    env.has_error
                ])

            # Flappy logs
            else:
                writer.writerow([
                    ep,
                    round(total_reward, 3),
                    env.pipes_passed,
                    final_info.get("death_reason", "none")
                ])

            print(f"Episode {ep}/{episodes} complete.")

    print("\nEvaluation complete.")
    print("CSV saved at:", csv_path)
    return csv_path


if __name__ == "__main__":
    parser = argparse.ArgumentParser()

    parser.add_argument("--model_path", type=str, required=True)
    parser.add_argument("--episodes", type=int, default=30)

    args = parser.parse_args()

    evaluate(args.model_path, args.episodes)