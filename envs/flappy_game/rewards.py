# rewards.py

def explorer_reward(pipes_passed, terminated):
    """
    Reward logic for the explorer persona:
    - Small reward each step to encourage movement.
    - Moderate reward for passing pipes.
    - Mild penalty for crashing.
    """
    reward = 0.2              
    reward += pipes_passed * 1.0

    if terminated:
        reward -= 5.0

    return reward


def survivor_reward(pipes_passed, terminated):
    """
    Reward logic for the survivor persona:
    - Strong reward for simply staying alive.
    - Higher reward for passing pipes.
    - Strong penalty for crashing.
    """
    reward = 0.5             
    reward += pipes_passed * 2.0

    if terminated:
        reward -= 10.0

    return reward

# Convenience function if you want dynamic selection.
def compute_reward(persona, pipes_passed, terminated):
    if persona == "survivor":
        return survivor_reward(pipes_passed, terminated)
    return explorer_reward(pipes_passed, terminated)