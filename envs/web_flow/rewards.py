# rewards.py

def explorer_reward(env, action, prev_page, next_page, had_error):
    """
    Reward logic for the explorer persona.
    This function is not used directly (web_env has built-in reward logic),
    but it's included for completeness.
    """
    reward = 0.0

    if next_page != prev_page:
        reward += 3.0

    if had_error:
        reward -= 2.0

    if next_page == "done":
        reward += 10.0

    return reward


def survivor_reward(env, action, prev_page, next_page, had_error):
    """
    Reward logic for the survivor persona.
    """
    reward = 0.0

    if not had_error and next_page != "done":
        reward += 1.0

    if had_error:
        reward -= 8.0

    if next_page == "done":
        reward += 5.0

    return reward