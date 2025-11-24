# train.py

import argparse
import yaml
import os
import gymnasium as gym
from stable_baselines3 import PPO, A2C
from stable_baselines3.common.vec_env import DummyVecEnv

# Import environments
from envs.web_flow.web_env import WebFlowEnv
from envs.flappy_game.flappy_env import FlappyEnv


def load_yaml(path):
    with open(path, "r") as f:
        return yaml.safe_load(f)


def make_env(app, persona):
    """Return the correct environment instance."""
    if app == "web":
        return WebFlowEnv(persona=persona)
    elif app == "flappy":
        return FlappyEnv(persona=persona)
    else:
        raise ValueError("Unknown app: choose 'web' or 'flappy'")


def train_model(algo, app, persona, timesteps, seed, config_paths):
    """
    Main training function.
    Loads configs → creates env → trains → saves model.
    """

    # Load configs
    algo_cfg = load_yaml(config_paths["algo"])
    seed_cfg = load_yaml(config_paths["seeds"])

    print("\n--- TRAINING CONFIG ---")
    print("Algorithm:", algo)
    print("App:", app)
    print("Persona:", persona)
    print("Timesteps:", timesteps)
    print("Seed:", seed)

    # Create environment
    env = DummyVecEnv([lambda: make_env(app, persona)])

    # Select algorithm
    if algo == "ppo":
        model = PPO(
            algo_cfg["policy"],
            env,
            seed=seed,
            **{k: v for k, v in algo_cfg.items() if k not in ["algo", "policy"]},
            verbose=1,
        )
    elif algo == "a2c":
        model = A2C(
            algo_cfg["policy"],
            env,
            seed=seed,
            **{k: v for k, v in algo_cfg.items() if k not in ["algo", "policy"]},
            verbose=1,
        )
    else:
        raise ValueError("Unknown algorithm: choose 'ppo' or 'a2c'")

    # Train
    model.learn(total_timesteps=timesteps)

    # Save
    os.makedirs("models", exist_ok=True)
    model_path = f"models/{app}_{algo}_{persona}_seed{seed}.zip"
    model.save(model_path)

    print("\nModel saved to:", model_path)
    return model_path


if __name__ == "__main__":
    parser = argparse.ArgumentParser()

    parser.add_argument("--algo", type=str, required=True, help="ppo or a2c")
    parser.add_argument("--app", type=str, required=True, help="web or flappy")
    parser.add_argument("--persona", type=str, required=True, help="explorer or survivor")
    parser.add_argument("--timesteps", type=int, default=100000)
    parser.add_argument("--seed", type=int, default=7)

    args = parser.parse_args()

    # Config paths
    config_paths = {
        "algo": f"configs/algo_{args.algo}.yaml",
        "seeds": "configs/seeds.yaml",
    }

    train_model(
        algo=args.algo,
        app=args.app,
        persona=args.persona,
        timesteps=args.timesteps,
        seed=args.seed,
        config_paths=config_paths,
    )