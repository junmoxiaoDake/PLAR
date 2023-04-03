import time

from envs import REGISTRY as env_REGISTRY
from functools import partial
from components.episode_buffer import EpisodeBatch
import numpy as np


import seaborn as sns
import pandas as pd
import matplotlib.pyplot as plt

class EpisodeRunner:

    def __init__(self, args, logger):
        self.args = args
        self.logger = logger
        self.batch_size = self.args.batch_size_run
        assert self.batch_size == 1

        self.env = env_REGISTRY[self.args.env](**self.args.env_args)
        self.episode_limit = self.env.episode_limit
        self.t = 0

        self.t_env = 0
        self.count_time = 0
        self.obs_total_sum_time = 0
        self.train_returns = []
        self.test_returns = []
        self.train_stats = {}
        self.test_stats = {}


        self.paint_flag = True
        self.log_train_stats_t = -1000000



    def setup(self, scheme, groups, preprocess, mac):
        self.new_batch = partial(EpisodeBatch, scheme, groups, self.batch_size, self.episode_limit + 1,
                                 preprocess=preprocess, device=self.args.device)
        self.mac = mac


    def get_env_info(self):
        return self.env.get_env_info()


    def save_replay(self):
        self.env.save_replay()

    def close_env(self):
        self.env.close()


    def reset(self):
        self.batch = self.new_batch()
        self.env.reset()
        self.t = 0

    def paint_time_difference(self, array1, array2, array3, array4, mapName):
        data = array1,array2,array3,array4
        label = ['matrix', 'improve_matrix', 'original','comprehensive']
        df = []
        for i in range(len(data)):
            df.append(pd.DataFrame(data[i]).melt(var_name='run_time', value_name='time_difference'))
            df[i]['method'] = label[i]

        df = pd.concat(df)  # 合并
        sns.lineplot(x="run_time", y="time_difference", hue="method", style="method", data=df)
        plt.title(mapName)
        plt.show()


    def paint_times(self, times_array,comprehensive_times_array,mapName):

        data = times_array, comprehensive_times_array
        label = ['matrix_origin', 'comprehensive']
        df = []
        for i in range(len(data)):
            df.append(pd.DataFrame(data[i]).melt(var_name='run_time', value_name='times'))
            df[i]['method'] = label[i]
        df = pd.concat(df)
        sns.lineplot(x="run_time", y="times", hue="method", style="method", data=df)
        plt.title(mapName)
        plt.show()


    def paint_survival_agents(self, array1, array2, mapName):
        data = array1, array2
        label = ['allies', 'enemies']
        df = []
        for i in range(len(data)):
            df.append(pd.DataFrame(data[i]).melt(var_name='run_time', value_name='survive_number'))
            df[i]['Type'] = label[i]

        df = pd.concat(df)
        sns.lineplot(x="run_time", y="survive_number", hue="Type", style="Type", data=df)
        plt.title(mapName)
        plt.show()

    def paint_map(self):

        if self.count_time > 200 and self.paint_flag==True:
            array_data = self.env.get_time_difference_array()
            self.paint_time_difference(array_data[0], array_data[1], array_data[2], array_data[3],array_data[4])

            times_data = self.env.get_times()
            self.paint_times(times_data[0],times_data[1], array_data[4])
            self.paint_flag = False
            # survival_data = self.env.get_survival_agents()
            # self.paint_survival_agents(survival_data[0], survival_data[1], array_data[4])


    def run(self, test_mode=False, choose_matrix=False, choose_state = True):
        self.reset()

        terminated = False
        episode_return = 0
        self.mac.init_hidden(batch_size=self.batch_size)

        obs_sum_time = 0

        mean_field_actions = np.zeros(self.env.n_actions)

        while not terminated:
            if choose_matrix == False:
                obs_start_time = time.time()
                temp_obs = self.env.get_cut_down_obs()
                obs_end_time = time.time()
            else:
                obs_start_time = time.time()
                temp_obs = self.env.get_obs_comprehensive()
                obs_end_time = time.time()

            obs_difference_time = obs_end_time - obs_start_time

            obs_sum_time += obs_difference_time
            if choose_state:
                pre_transition_data = {
                    "state": [self.env.get_state()],
                    "avail_actions": [self.env.get_avail_actions()],
                    "obs": [temp_obs]
                }
            else:
                pre_transition_data = {
                    "avail_actions": [self.env.get_avail_actions()],
                    "obs": [temp_obs]
                }

            # self.count_time += 1
            # self.paint_map()

            if hasattr(self.args, 'choose_meanfield'):
                if self.args.choose_meanfield:
                    pre_transition_data["mean_field_actions"] = mean_field_actions

            self.batch.update(pre_transition_data, ts=self.t)

            actions = self.mac.select_actions(self.batch, t_ep=self.t, t_env=self.t_env, test_mode=test_mode)

            mean_field_actions = np.mean(list(map(lambda x: np.eye(self.env.n_actions)[x], actions[0])), axis=0, keepdims=True)[0]


            reward, terminated, env_info = self.env.step(actions[0])
            episode_return += reward

            post_transition_data = {
                "actions": actions,
                "reward": [(reward,)],
                "terminated": [(terminated != env_info.get("episode_limit", False),)],
            }

            self.batch.update(post_transition_data, ts=self.t)

            self.t += 1

        if choose_matrix == False:
            obs_start_time1 = time.time()
            temp_last_obs = self.env.get_cut_down_obs()
            obs_end_time1 = time.time()
        else:
            obs_start_time1 = time.time()
            temp_last_obs = self.env.get_obs_comprehensive()
            obs_end_time1 = time.time()

        obs_difference_time1 = obs_end_time1 - obs_start_time1

        obs_sum_time += obs_difference_time1
        if choose_state:
            last_data = {
                "state": [self.env.get_state()],
                "avail_actions": [self.env.get_avail_actions()],
                "obs": [temp_last_obs]
            }
        else:
            last_data = {
                "avail_actions": [self.env.get_avail_actions()],
                "obs": [temp_last_obs]
            }

        if hasattr(self.args, 'choose_meanfield'):
            if self.args.choose_meanfield:
                last_data["mean_field_actions"] = mean_field_actions

        self.batch.update(last_data, ts=self.t)

        # Select actions in the last stored state
        actions = self.mac.select_actions(self.batch, t_ep=self.t, t_env=self.t_env, test_mode=test_mode)
        self.batch.update({"actions": actions}, ts=self.t)

        cur_stats = self.test_stats if test_mode else self.train_stats
        cur_returns = self.test_returns if test_mode else self.train_returns
        log_prefix = "test_" if test_mode else ""

        cur_stats.update({k: cur_stats.get(k, 0) + env_info.get(k, 0) for k in set(cur_stats) | set(env_info)})
        cur_stats["n_episodes"] = 1 + cur_stats.get("n_episodes", 0)
        cur_stats["ep_length"] = self.t + cur_stats.get("ep_length", 0)

        if not test_mode:
            self.t_env += self.t
            self.obs_total_sum_time += obs_sum_time
            cur_stats["obs_total_time"] = self.obs_total_sum_time


        cur_returns.append(episode_return)

        if test_mode and (len(self.test_returns) == self.args.test_nepisode):
            self._log(cur_returns, cur_stats, log_prefix)
        elif self.t_env - self.log_train_stats_t >= self.args.runner_log_interval:
            self._log(cur_returns, cur_stats, log_prefix)
            if hasattr(self.mac.action_selector, "epsilon"):
                self.logger.log_stat("epsilon", self.mac.action_selector.epsilon, self.t_env)
            self.log_train_stats_t = self.t_env

        return self.batch


    def _log(self, returns, stats, prefix):
        self.logger.log_stat(prefix + "return_mean", np.mean(returns), self.t_env)
        self.logger.log_stat(prefix + "return_std", np.std(returns), self.t_env)
        returns.clear()

        for k, v in stats.items():
            if k != "n_episodes":
                self.logger.log_stat(prefix + k + "_mean" , v/stats["n_episodes"], self.t_env)
        stats.clear()
