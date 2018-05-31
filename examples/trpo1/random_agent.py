import argparse
import logging
import sys
import os
import numpy as np
import gym
from arena.games.cust_control.single_gather_env import SingleGatherEnv
from pg_train import random_cont_direction
from gym.monitoring import VideoRecorder


class RandomAgent(object):
    """The world's simplest agent!"""

    def __init__(self, action_space):
        self.action_space = action_space

    def act(self, observation, reward, done):
        return self.action_space.sample()


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=None)
    parser.add_argument('env_id', nargs='?', default='CartPole-v0', help='Select the environment to run')
    args = parser.parse_args()

    # Call `undo_logger_setup` if you want to undo Gym's logger setup
    # and configure things manually. (The default should be fine most
    # of the time.)
    gym.undo_logger_setup()
    logger = logging.getLogger()
    formatter = logging.Formatter('[%(asctime)s] %(message)s')
    handler = logging.StreamHandler(sys.stderr)
    handler.setFormatter(formatter)
    logger.addHandler(handler)

    # You can set the level to logging.DEBUG or logging.WARN if you
    # want to change the amount of output.
    logger.setLevel(logging.INFO)
    # DIRECTIONS = np.array([(1., 0), (0, 1.), (-1., 0), (0, -1.)])
    ANGLES = [0, np.pi / 4, - np.pi / 4]
    DIRECTIONS = np.array([(np.cos(a), np.sin(a)) for a in ANGLES])


    def random_3direction():
        choice = np.random.randint(0, 3)
        return DIRECTIONS[choice, :]


    cwd = os.getcwd()
    env = SingleGatherEnv(file_path=cwd + "/cust_ant.xml",
                          f_gen_obj=random_cont_direction,
                          use_sparse_reward=True,
                          obj_dist=1.0)
    outdir = './random-agent-results'

    # env = GatherEnv()

    # You provide the directory to write to (can be an existing
    # directory, including one with existing data -- all monitor files
    # will be namespaced). You can also dump to a tempdir if you'd
    # like: tempfile.mkdtemp().

    # env = wrappers.Monitor(env, directory=outdir, force=True)
    # env = ComplexWrapper(env, max_episode_length=1000,
    #                            append_image=False, new_img_size=(64, 64), rgb_to_gray=True,
    #                            visible_state_ids=range(env.observation_space.shape[0]),
    #                            num_frame=3)
    agent = RandomAgent(env.action_space)

    episode_count = 1000
    reward = 0
    done = False

    while True:
        ob = env.reset()
        recorder = VideoRecorder(env, path=outdir + ".mp4")
        while True:
            action = agent.act(ob, reward, done)
            # image = env.render(mode='rgb_array')
            recorder.capture_frame()
            # cv2.imwrite("image.jpg", image)
            ob, reward, done, _ = env.step(action)
            if done:
                recorder.close()
                input("Press Enter to continue...")
                break
                # Note there's no env.render() here. But the environment still can open window and
                # render if asked by env.monitor: it calls env.render('rgb_array') to record video.
                # Video is not recorded every episode, see capped_cubic_video_schedule for details.

    # Close the env and write monitor result info to disk
    # env.close()

    # Upload to the scoreboard. We could also do this from another
    # process if we wanted.
    logger.info(
        "Successfully ran RandomAgent. Now trying to upload results to the scoreboard. If it breaks, you can always just try re-uploading the same results.")
    # gym.upload(outdir)
