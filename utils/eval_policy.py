import numpy as np


def eval_policy(policy, env, eval_episodes=10, eval_cnt=None):
    eval_env = env

    avg_reward = 0.0
    for _ in range(eval_episodes):
        state, _ = eval_env.reset()
        done = False
        while not done:
            action = policy.select_action(np.array(state))
            next_state, reward, terminated, truncated, _ = eval_env.step(action)
            done = float(terminated or truncated)

            avg_reward += reward
            state = next_state
    avg_reward /= eval_episodes

    print(
        "[{}] Evaluation over {} episodes: {}".format(
            eval_cnt, eval_episodes, avg_reward
        )
    )

    return avg_reward
