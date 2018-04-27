import gym

import soft_dqn
from baselines import logger
from baselines.common import set_global_seeds

def main():
    env = gym.make("MountainCar-v0")
    # Enabling layer_norm here is import for parameter space noise!
    model = soft_dqn.models.mlp([64], layer_norm=True)
    act = soft_dqn.learn(
        env,
        q_func=model,
        lr=1e-3,
        max_timesteps=100000,
        buffer_size=50000,
        exploration_fraction=0.1,
        exploration_final_eps=0.1,
        print_freq=10,
        param_noise=True,
        use_soft_max=True,
        temperature=20
    )
    print("Saving model to mountaincar_model.pkl")
    act.save("mountaincar_model.pkl")


if __name__ == '__main__':
    main()
