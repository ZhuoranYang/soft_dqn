import gym

import soft_dqn
from baselines import logger
from baselines.common import set_global_seeds


def callback(lcl, glb):
    # stop training if reward exceeds 199
    is_solved = lcl['t'] > 100 and sum(lcl['episode_rewards'][-101:-1]) / 100 >= 199
    return is_solved


def main():
    env = gym.make("CartPole-v0")
    model = soft_dqn.models.mlp([64])
    act = soft_dqn.learn(
        env,
        q_func=model,
        lr=1e-3,
        max_timesteps=100000,
        buffer_size=50000,
        exploration_fraction=0.05,
        exploration_final_eps=0.01,
        print_freq=10,
        callback=callback,
        use_soft_max=True,
        temperature=10
    )
    print("Saving model to cartpole_model.pkl")
    act.save("cartpole_model.pkl")


if __name__ == '__main__':
    main()
