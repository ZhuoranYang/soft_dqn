from soft_dqn import models  # noqa
from soft_dqn.build_graph import build_act, build_train  # noqa
from soft_dqn.simple import learn, load  # noqa
from baselines.deepq.replay_buffer import ReplayBuffer, PrioritizedReplayBuffer  # noqa

def wrap_atari_dqn(env):
    from baselines.common.atari_wrappers import wrap_deepmind
    return wrap_deepmind(env, frame_stack=True, scale=True)