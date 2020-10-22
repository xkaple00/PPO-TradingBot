from __future__ import absolute_import
from __future__ import print_function
from __future__ import division
import os
# os.environ["CUDA_VISIBLE_DEVICES"]="1"

import tensorflow as tf
import sys
import numpy as np


from TFTraderEnv import OhlcvEnv
from stable_baselines import DQN, PPO2, A2C, ACKTR
from stable_baselines.common.cmd_util import make_vec_env
from stable_baselines.common.policies import MlpPolicy, MlpLstmPolicy, MlpLnLstmPolicy
from stable_baselines.common.callbacks import BaseCallback
from stable_baselines.results_plotter import load_results, ts2xy
from stable_baselines.bench import Monitor
import shutil

class SaveOnBestTrainingRewardCallback(BaseCallback):
    """
    Callback for saving a model (the check is done every ``check_freq`` steps)
    based on the training reward (in practice, we recommend using ``EvalCallback``).

    :param check_freq: (int)
    :param log_dir: (str) Path to the folder where the model will be saved.
      It must contains the file created by the ``Monitor`` wrapper.
    :param verbose: (int)
    """
    def __init__(self, check_freq: int, log_dir: str, verbose=1):
        super(SaveOnBestTrainingRewardCallback, self).__init__(verbose)
        self.check_freq = check_freq
        self.log_dir = log_dir
        self.save_path = os.path.join(log_dir, 'best_model')
        self.best_mean_reward = -np.inf

    def _init_callback(self) -> None:
        # Create folder if needed
        if self.save_path is not None:
            os.makedirs(self.save_path, exist_ok=True)

    def _on_step(self) -> bool:
        if self.n_calls % self.check_freq == 0:

          # Retrieve training reward
          x, y = ts2xy(load_results(self.log_dir), 'timesteps')
          if len(x) > 0:
              # Mean training reward over the last 100 episodes
              mean_reward = np.mean(y[-100:])
              if self.verbose > 0:
                print("Num timesteps: {}".format(self.num_timesteps))
                print("Best mean reward: {:.2f} - Last mean reward per episode: {:.2f}".format(self.best_mean_reward, mean_reward))

              # New best model, you could save the agent here
              if mean_reward > self.best_mean_reward:
                  self.best_mean_reward = mean_reward
                  # Example for saving best model
                  if self.verbose > 0:
                    print("Saving new best model to {}".format(self.save_path))
                  self.model.save(self.save_path)

        return True


def main():

    # create environment for train and test
    # PATH_TRAIN = "./data/train/"
    # PATH_TEST = "./data/test/"
    # PATH_TRAIN = "../data/XAGUSD/"
    # PATH_TRAIN = "../data/BRENT_linux_timestamp/"
    PATH_TRAIN = "../data/ETH/"


    TIMESTEP = 512 # window size 30
    environment =  OhlcvEnv(window_size=TIMESTEP, path=PATH_TRAIN, train=True)
    # test_environment = OhlcvEnv(window_size=TIMESTEP, path=PATH_TEST, train=False)


    if os.path.isdir('./logs'): 
        shutil.rmtree('./logs')

    if os.path.isdir('./saved_agent'): 
        shutil.rmtree('./saved_agent')

    if not os.path.exists('./info'):
        os.makedirs('./info')

   
    # Create log dir
    log_dir = "./saved_agent"
    os.makedirs(log_dir, exist_ok=True)


    # Logs will be saved in log_dir/monitor.csv
    environment = Monitor(environment, log_dir)

    environment = make_vec_env(lambda: environment, n_envs=8)
    
    callback = SaveOnBestTrainingRewardCallback(check_freq=1000, log_dir="./saved_agent")

    # model = PPO2.load("./best_model.zip", env=environment)

    model = PPO2(MlpLstmPolicy, environment, gamma=0.999, n_steps=512, tensorboard_log='./logs', nminibatches=8, verbose=1) #MlpLstmPolicy #n_cpu_tf_sess=8
    model.learn(total_timesteps=10000000, callback=callback)

    model.save("./saved_agent") 

if __name__ == '__main__':
    main()