from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

from smac.env.multiagentenv import MultiAgentEnv
from smac.env.starcraft2.maps import get_map_params

import atexit
from warnings import warn
from operator import attrgetter
from copy import deepcopy
import numpy as np
import enum
import math
import time
from absl import logging

from pysc2 import maps
from pysc2 import run_configs
from pysc2.lib import protocol

from s2clientprotocol import common_pb2 as sc_common
from s2clientprotocol import sc2api_pb2 as sc_pb
from s2clientprotocol import raw_pb2 as r_pb
from s2clientprotocol import debug_pb2 as d_pb

from line_profiler import LineProfiler
from scipy import spatial




races = {
    "R": sc_common.Random,
    "P": sc_common.Protoss,
    "T": sc_common.Terran,
    "Z": sc_common.Zerg,
}

difficulties = {
    "1": sc_pb.VeryEasy,
    "2": sc_pb.Easy,
    "3": sc_pb.Medium,
    "4": sc_pb.MediumHard,
    "5": sc_pb.Hard,
    "6": sc_pb.Harder,
    "7": sc_pb.VeryHard,
    "8": sc_pb.CheatVision,
    "9": sc_pb.CheatMoney,
    "A": sc_pb.CheatInsane,
}

actions = {
    "move": 16,  # target: PointOrUnit
    "attack": 23,  # target: PointOrUnit
    "stop": 4,  # target: None
    "heal": 386,  # Unit
}


class Direction(enum.IntEnum):
    NORTH = 0
    SOUTH = 1
    EAST = 2
    WEST = 3


class StarCraft2Env(MultiAgentEnv):
    """The StarCraft II environment for decentralised multi-agent
    micromanagement scenarios.
    """

    def __init__(
        self,
        map_name="8m",
        step_mul=8,
        move_amount=2,
        difficulty="7",
        game_version=None,
        seed=None,
        continuing_episode=False,
        obs_all_health=True,
        obs_own_health=True,
        obs_last_action=False,
        obs_pathing_grid=False,
        obs_terrain_height=False,
        obs_instead_of_state=False,
        obs_timestep_number=False,
        state_last_action=True,
        state_timestep_number=False,
        reward_sparse=False,
        reward_only_positive=True,
        reward_death_value=10,
        reward_win=200,
        reward_defeat=0,
        reward_negative_scale=0.5,
        reward_scale=True,
        reward_scale_rate=20,
        replay_dir="",
        replay_prefix="",
        window_size_x=1920,
        window_size_y=1200,
        heuristic_ai=False,
        heuristic_rest=False,
        debug=False,
    ):
        """
        Create a StarCraftC2Env environment.

        Parameters
        ----------
        map_name : str, optional
            The name of the SC2 map to play (default is "8m"). The full list
            can be found by running bin/map_list.
        step_mul : int, optional
            How many game steps per agent step (default is 8). None
            indicates to use the default map step_mul.
        move_amount : float, optional
            How far away units are ordered to move per step (default is 2).
        difficulty : str, optional
            The difficulty of built-in computer AI bot (default is "7").
        game_version : str, optional
            StarCraft II game version (default is None). None indicates the
            latest version.
        seed : int, optional
            Random seed used during game initialisation. This allows to
        continuing_episode : bool, optional
            Whether to consider episodes continuing or finished after time
            limit is reached (default is False).
        obs_all_health : bool, optional
            Agents receive the health of all units (in the sight range) as part
            of observations (default is True).
        obs_own_health : bool, optional
            Agents receive their own health as a part of observations (default
            is False). This flag is ignored when obs_all_health == True.
        obs_last_action : bool, optional
            Agents receive the last actions of all units (in the sight range)
            as part of observations (default is False).
        obs_pathing_grid : bool, optional
            Whether observations include pathing values surrounding the agent
            (default is False).
        obs_terrain_height : bool, optional
            Whether observations include terrain height values surrounding the
            agent (default is False).
        obs_instead_of_state : bool, optional
            Use combination of all agents' observations as the global state
            (default is False).
        obs_timestep_number : bool, optional
            Whether observations include the current timestep of the episode
            (default is False).
        state_last_action : bool, optional
            Include the last actions of all agents as part of the global state
            (default is True).
        state_timestep_number : bool, optional
            Whether the state include the current timestep of the episode
            (default is False).
        reward_sparse : bool, optional
            Receive 1/-1 reward for winning/loosing an episode (default is
            False). Whe rest of reward parameters are ignored if True.
        reward_only_positive : bool, optional
            Reward is always positive (default is True).
        reward_death_value : float, optional
            The amount of reward received for killing an enemy unit (default
            is 10). This is also the negative penalty for having an allied unit
            killed if reward_only_positive == False.
        reward_win : float, optional
            The reward for winning in an episode (default is 200).
        reward_defeat : float, optional
            The reward for loosing in an episode (default is 0). This value
            should be nonpositive.
        reward_negative_scale : float, optional
            Scaling factor for negative rewards (default is 0.5). This
            parameter is ignored when reward_only_positive == True.
        reward_scale : bool, optional
            Whether or not to scale the reward (default is True).
        reward_scale_rate : float, optional
            Reward scale rate (default is 20). When reward_scale == True, the
            reward received by the agents is divided by (max_reward /
            reward_scale_rate), where max_reward is the maximum possible
            reward per episode without considering the shield regeneration
            of Protoss units.
        replay_dir : str, optional
            The directory to save replays (default is None). If None, the
            replay will be saved in Replays directory where StarCraft II is
            installed.
        replay_prefix : str, optional
            The prefix of the replay to be saved (default is None). If None,
            the name of the map will be used.
        window_size_x : int, optional
            The length of StarCraft II window size (default is 1920).
        window_size_y: int, optional
            The height of StarCraft II window size (default is 1200).
        heuristic_ai: bool, optional
            Whether or not to use a non-learning heuristic AI (default False).
        heuristic_rest: bool, optional
            At any moment, restrict the actions of the heuristic AI to be
            chosen from actions available to RL agents (default is False).
            Ignored if heuristic_ai == False.
        debug: bool, optional
            Log messages about observations, state, actions and rewards for
            debugging purposes (default is False).
        """


        self.paint_flag = True

        self.matrix_array = np.array([])
        self.improve_matrix_array = np.array([])

        self.original_array = np.array([])

        self.comprehensive_array = np.array([])


        self.times_array = np.array([])
        self.comprehensive_times_array = np.array([])


        self.survival_allys = np.array([])
        self.survival_enemies =  np.array([])


        self.map_name = map_name
        map_params = get_map_params(self.map_name)
        self.n_agents = map_params["n_agents"]
        self.n_enemies = map_params["n_enemies"]
        self.episode_limit = map_params["limit"]
        self._move_amount = move_amount
        self._step_mul = step_mul
        self.difficulty = difficulty

        self.obs_own_health = obs_own_health
        self.obs_all_health = obs_all_health
        self.obs_instead_of_state = obs_instead_of_state
        self.obs_last_action = obs_last_action
        self.obs_pathing_grid = obs_pathing_grid
        self.obs_terrain_height = obs_terrain_height
        self.obs_timestep_number = obs_timestep_number
        self.state_last_action = state_last_action
        self.state_timestep_number = state_timestep_number
        if self.obs_all_health:
            self.obs_own_health = True
        self.n_obs_pathing = 8
        self.n_obs_height = 9

        self.reward_sparse = reward_sparse
        self.reward_only_positive = reward_only_positive
        self.reward_negative_scale = reward_negative_scale
        self.reward_death_value = reward_death_value
        self.reward_win = reward_win
        self.reward_defeat = reward_defeat
        self.reward_scale = reward_scale
        self.reward_scale_rate = reward_scale_rate

        self.game_version = game_version
        self.continuing_episode = continuing_episode
        self._seed = seed
        self.heuristic_ai = heuristic_ai
        self.heuristic_rest = heuristic_rest
        self.debug = debug
        self.window_size = (window_size_x, window_size_y)
        self.replay_dir = replay_dir
        self.replay_prefix = replay_prefix

        self.n_actions_no_attack = 6
        self.n_actions_move = 4
        self.n_actions = self.n_actions_no_attack + 5

        self._agent_race = map_params["a_race"]
        self._bot_race = map_params["b_race"]
        self.shield_bits_ally = 1 if self._agent_race == "P" else 0
        self.shield_bits_enemy = 1 if self._bot_race == "P" else 0
        self.unit_type_bits = map_params["unit_type_bits"]
        self.map_type = map_params["map_type"]
        self._unit_types = None

        self.max_reward = (
            self.n_enemies * self.reward_death_value + self.reward_win
        )


        self.ally_state_attr_names = [
            "health",
            "energy/cooldown",
            "rel_x",
            "rel_y",
        ]
        self.enemy_state_attr_names = ["health", "rel_x", "rel_y"]

        if self.shield_bits_ally > 0:
            self.ally_state_attr_names += ["shield"]
        if self.shield_bits_enemy > 0:
            self.enemy_state_attr_names += ["shield"]

        if self.unit_type_bits > 0:
            bit_attr_names = [
                "type_{}".format(bit) for bit in range(self.unit_type_bits)
            ]
            self.ally_state_attr_names += bit_attr_names
            self.enemy_state_attr_names += bit_attr_names

        self.agents = {}
        self.enemies = {}
        self._episode_count = 0
        self._episode_steps = 0
        self._total_steps = 0
        self._obs = None
        self.battles_won = 0
        self.battles_game = 0
        self.timeouts = 0
        self.force_restarts = 0
        self.last_stats = None
        self.death_tracker_ally = np.zeros(self.n_agents)
        self.death_tracker_enemy = np.zeros(self.n_enemies)
        self.previous_ally_units = None
        self.previous_enemy_units = None
        self.last_action = np.zeros((self.n_agents, self.n_actions))
        self._min_unit_type = 0
        self.marine_id = self.marauder_id = self.medivac_id = 0
        self.hydralisk_id = self.zergling_id = self.baneling_id = 0
        self.stalker_id = self.colossus_id = self.zealot_id = 0
        self.max_distance_x = 0
        self.max_distance_y = 0
        self.map_x = 0
        self.map_y = 0
        self.reward = 0
        self.renderer = None
        self.terrain_height = None
        self.pathing_grid = None
        self._run_config = None
        self._sc2_proc = None
        self._controller = None

        atexit.register(lambda: self.close())

    def _launch(self):
        """Launch the StarCraft II game."""
        self._run_config = run_configs.get(version=self.game_version)
        _map = maps.get(self.map_name)

        # Setting up the interface
        interface_options = sc_pb.InterfaceOptions(raw=True, score=False)
        self._sc2_proc = self._run_config.start(
            window_size=self.window_size, want_rgb=False
        )
        self._controller = self._sc2_proc.controller

        # Request to create the game
        create = sc_pb.RequestCreateGame(
            local_map=sc_pb.LocalMap(
                map_path=_map.path,
                map_data=self._run_config.map_data(_map.path),
            ),
            realtime=False,
            random_seed=self._seed,
        )
        create.player_setup.add(type=sc_pb.Participant)
        create.player_setup.add(
            type=sc_pb.Computer,
            race=races[self._bot_race],
            difficulty=difficulties[self.difficulty],
        )
        self._controller.create_game(create)

        join = sc_pb.RequestJoinGame(
            race=races[self._agent_race], options=interface_options
        )
        self._controller.join_game(join)

        game_info = self._controller.game_info()
        map_info = game_info.start_raw
        map_play_area_min = map_info.playable_area.p0
        map_play_area_max = map_info.playable_area.p1
        self.max_distance_x = map_play_area_max.x - map_play_area_min.x
        self.max_distance_y = map_play_area_max.y - map_play_area_min.y
        self.map_x = map_info.map_size.x
        self.map_y = map_info.map_size.y

        if map_info.pathing_grid.bits_per_pixel == 1:
            vals = np.array(list(map_info.pathing_grid.data)).reshape(
                self.map_x, int(self.map_y / 8)
            )
            self.pathing_grid = np.transpose(
                np.array(
                    [
                        [(b >> i) & 1 for b in row for i in range(7, -1, -1)]
                        for row in vals
                    ],
                    dtype=np.bool,
                )
            )
        else:
            self.pathing_grid = np.invert(
                np.flip(
                    np.transpose(
                        np.array(
                            list(map_info.pathing_grid.data), dtype=np.bool
                        ).reshape(self.map_x, self.map_y)
                    ),
                    axis=1,
                )
            )

        self.terrain_height = (
            np.flip(
                np.transpose(
                    np.array(list(map_info.terrain_height.data)).reshape(
                        self.map_x, self.map_y
                    )
                ),
                1,
            )
            / 255
        )

    def reset(self):
        """Reset the environment. Required after each full episode.
        Returns initial observations and states.
        """
        self._episode_steps = 0
        if self._episode_count == 0:
            # Launch StarCraft II
            self._launch()
        else:
            self._restart()

        # Information kept for counting the reward
        self.death_tracker_ally = np.zeros(self.n_agents)
        self.death_tracker_enemy = np.zeros(self.n_enemies)
        self.previous_ally_units = None
        self.previous_enemy_units = None
        self.win_counted = False
        self.defeat_counted = False

        self.last_action = np.zeros((self.n_agents, self.n_actions))

        if self.heuristic_ai:
            self.heuristic_targets = [None] * self.n_agents

        try:
            self._obs = self._controller.observe()
            self.init_units()
        except (protocol.ProtocolError, protocol.ConnectionError):
            self.full_restart()

        if self.debug:
            logging.debug(
                "Started Episode {}".format(self._episode_count).center(
                    60, "*"
                )
            )

        return self.get_cut_down_obs(), self.get_state()

    def _restart(self):
        """Restart the environment by killing all units on the map.
        There is a trigger in the SC2Map file, which restarts the
        episode when there are no units left.
        """
        try:
            self._kill_all_units()
            self._controller.step(2)
        except (protocol.ProtocolError, protocol.ConnectionError):
            self.full_restart()

    def full_restart(self):
        """Full restart. Closes the SC2 process and launches a new one."""
        self._sc2_proc.close()
        self._launch()
        self.force_restarts += 1

    def step(self, actions):
        """A single environment step. Returns reward, terminated, info."""
        actions_int = [int(a) for a in actions]

        self.last_action = np.eye(self.n_actions)[np.array(actions_int)]

        # Collect individual actions
        sc_actions = []
        if self.debug:
            logging.debug("Actions".center(60, "-"))

        for a_id, action in enumerate(actions_int):
            if not self.heuristic_ai:
                sc_action = self.get_agent_action(a_id, action)
            else:
                sc_action, action_num = self.get_agent_action_heuristic(
                    a_id, action
                )
                actions[a_id] = action_num
            if sc_action:
                sc_actions.append(sc_action)

        # Send action request
        req_actions = sc_pb.RequestAction(actions=sc_actions)
        try:
            self._controller.actions(req_actions)
            # Make step in SC2, i.e. apply actions
            self._controller.step(self._step_mul)
            # Observe here so that we know if the episode is over.
            self._obs = self._controller.observe()
        except (protocol.ProtocolError, protocol.ConnectionError):
            self.full_restart()
            return 0, True, {}

        self._total_steps += 1
        self._episode_steps += 1

        # Update units
        game_end_code = self.update_units()

        terminated = False
        reward = self.reward_battle()
        info = {"battle_won": False}

        # count units that are still alive
        dead_allies, dead_enemies = 0, 0
        for _al_id, al_unit in self.agents.items():
            if al_unit.health == 0:
                dead_allies += 1
        for _e_id, e_unit in self.enemies.items():
            if e_unit.health == 0:
                dead_enemies += 1

        info["dead_allies"] = dead_allies
        info["dead_enemies"] = dead_enemies

        if game_end_code is not None:
            # Battle is over
            terminated = True
            self.battles_game += 1
            if game_end_code == 1 and not self.win_counted:
                self.battles_won += 1
                self.win_counted = True
                info["battle_won"] = True
                if not self.reward_sparse:
                    reward += self.reward_win
                else:
                    reward = 1
            elif game_end_code == -1 and not self.defeat_counted:
                self.defeat_counted = True
                if not self.reward_sparse:
                    reward += self.reward_defeat
                else:
                    reward = -1

        elif self._episode_steps >= self.episode_limit:
            # Episode limit reached
            terminated = True
            if self.continuing_episode:
                info["episode_limit"] = True
            self.battles_game += 1
            self.timeouts += 1

        if self.debug:
            logging.debug("Reward = {}".format(reward).center(60, "-"))

        if terminated:
            self._episode_count += 1

        if self.reward_scale:
            reward /= self.max_reward / self.reward_scale_rate

        self.reward = reward

        return reward, terminated, info

    def get_agent_action(self, a_id, action):
        """Construct the action for agent a_id."""
        avail_actions = self.get_avail_agent_actions(a_id)
        assert (
            avail_actions[action] == 1
        ), "Agent {} cannot perform action {}".format(a_id, action)

        unit = self.get_unit_by_id(a_id)
        tag = unit.tag
        x = unit.pos.x
        y = unit.pos.y

        if action == 0:
            # no-op (valid only when dead)
            assert unit.health == 0, "No-op only available for dead agents."
            if self.debug:
                logging.debug("Agent {}: Dead".format(a_id))
            return None
        elif action == 1:
            # stop
            cmd = r_pb.ActionRawUnitCommand(
                ability_id=actions["stop"],
                unit_tags=[tag],
                queue_command=False,
            )
            if self.debug:
                logging.debug("Agent {}: Stop".format(a_id))

        elif action == 2:
            # move north
            cmd = r_pb.ActionRawUnitCommand(
                ability_id=actions["move"],
                target_world_space_pos=sc_common.Point2D(
                    x=x, y=y + self._move_amount
                ),
                unit_tags=[tag],
                queue_command=False,
            )
            if self.debug:
                logging.debug("Agent {}: Move North".format(a_id))

        elif action == 3:
            # move south
            cmd = r_pb.ActionRawUnitCommand(
                ability_id=actions["move"],
                target_world_space_pos=sc_common.Point2D(
                    x=x, y=y - self._move_amount
                ),
                unit_tags=[tag],
                queue_command=False,
            )
            if self.debug:
                logging.debug("Agent {}: Move South".format(a_id))

        elif action == 4:
            # move east
            cmd = r_pb.ActionRawUnitCommand(
                ability_id=actions["move"],
                target_world_space_pos=sc_common.Point2D(
                    x=x + self._move_amount, y=y
                ),
                unit_tags=[tag],
                queue_command=False,
            )
            if self.debug:
                logging.debug("Agent {}: Move East".format(a_id))

        elif action == 5:
            # move west
            cmd = r_pb.ActionRawUnitCommand(
                ability_id=actions["move"],
                target_world_space_pos=sc_common.Point2D(
                    x=x - self._move_amount, y=y
                ),
                unit_tags=[tag],
                queue_command=False,
            )
            if self.debug:
                logging.debug("Agent {}: Move West".format(a_id))
        else:
            # attack/heal units that are in range
            target_nearest_num = action - self.n_actions_no_attack
            target_id = 0
            target_num = self.n_enemies
            target_items = self.enemies.items()
            if self.map_type == "MMM" and unit.unit_type == self.medivac_id:
                # Medivacs cannot heal themselves or other flying units
                target_items = [
                    (t_id, t_unit)
                    for (t_id, t_unit) in self.agents.items()
                    if t_unit.unit_type != self.medivac_id
                ]
                target_num = self.n_agents
            t_dist = np.full(target_num, np.inf)
            for t_id, t_unit in target_items:
                t_x = t_unit.pos.x
                t_y = t_unit.pos.y
                dist = self.distance(x, y, t_x, t_y)
                t_dist[t_id] = dist
            temp_list = deepcopy(t_dist)

            full_flag = False
            for i in range(target_nearest_num+1):
                index = temp_list.argmin()
                t_unit = self.enemies[index]
                if self.map_type == "MMM" and unit.unit_type == self.medivac_id:
                    t_unit = self.get_unit_by_id(index)
                if t_unit.health <= 0:
                    temp_list[index] = float('inf')
                    while True:
                        index = temp_list.argmin()
                        t_unit = self.enemies[index]
                        if self.map_type == "MMM" and unit.unit_type == self.medivac_id:
                            t_unit = self.get_unit_by_id(index)
                        if temp_list[index] == float('inf'):
                            full_flag = True
                            break
                        if t_unit.health <= 0:
                            temp_list[index] = float('inf')
                            continue
                        else:
                            break
                if full_flag:
                    break
                target_id = index
                temp_list[index] = float('inf')

            if self.map_type == "MMM" and unit.unit_type == self.medivac_id:
                target_unit = self.agents[target_id]
                action_name = "heal"
            else:
                target_unit = self.enemies[target_id]
                action_name = "attack"

            action_id = actions[action_name]
            target_tag = target_unit.tag

            cmd = r_pb.ActionRawUnitCommand(
                ability_id=action_id,
                target_unit_tag=target_tag,
                unit_tags=[tag],
                queue_command=False,
            )

            if self.debug:
                logging.debug(
                    "Agent {} {}s unit # {}".format(
                        a_id, action_name, target_id
                    )
                )

        sc_action = sc_pb.Action(action_raw=r_pb.ActionRaw(unit_command=cmd))
        return sc_action

    def get_agent_action_heuristic(self, a_id, action):
        unit = self.get_unit_by_id(a_id)
        tag = unit.tag

        target = self.heuristic_targets[a_id]
        if unit.unit_type == self.medivac_id:
            if (
                target is None
                or self.agents[target].health == 0
                or self.agents[target].health == self.agents[target].health_max
            ):
                min_dist = math.hypot(self.max_distance_x, self.max_distance_y)
                min_id = -1
                for al_id, al_unit in self.agents.items():
                    if al_unit.unit_type == self.medivac_id:
                        continue
                    if (
                        al_unit.health != 0
                        and al_unit.health != al_unit.health_max
                    ):
                        dist = self.distance(
                            unit.pos.x,
                            unit.pos.y,
                            al_unit.pos.x,
                            al_unit.pos.y,
                        )
                        if dist < min_dist:
                            min_dist = dist
                            min_id = al_id
                self.heuristic_targets[a_id] = min_id
                if min_id == -1:
                    self.heuristic_targets[a_id] = None
                    return None, 0
            action_id = actions["heal"]
            target_tag = self.agents[self.heuristic_targets[a_id]].tag
        else:
            if target is None or self.enemies[target].health == 0:
                min_dist = math.hypot(self.max_distance_x, self.max_distance_y)
                min_id = -1
                for e_id, e_unit in self.enemies.items():
                    if (
                        unit.unit_type == self.marauder_id
                        and e_unit.unit_type == self.medivac_id
                    ):
                        continue
                    if e_unit.health > 0:
                        dist = self.distance(
                            unit.pos.x, unit.pos.y, e_unit.pos.x, e_unit.pos.y
                        )
                        if dist < min_dist:
                            min_dist = dist
                            min_id = e_id
                self.heuristic_targets[a_id] = min_id
                if min_id == -1:
                    self.heuristic_targets[a_id] = None
                    return None, 0
            action_id = actions["attack"]
            target_tag = self.enemies[self.heuristic_targets[a_id]].tag

        action_num = self.heuristic_targets[a_id] + self.n_actions_no_attack

        # Check if the action is available
        if (
            self.heuristic_rest
            and self.get_avail_agent_actions(a_id)[action_num] == 0
        ):

            # Move towards the target rather than attacking/healing
            if unit.unit_type == self.medivac_id:
                target_unit = self.agents[self.heuristic_targets[a_id]]
            else:
                target_unit = self.enemies[self.heuristic_targets[a_id]]

            delta_x = target_unit.pos.x - unit.pos.x
            delta_y = target_unit.pos.y - unit.pos.y

            if abs(delta_x) > abs(delta_y):  # east or west
                if delta_x > 0:  # east
                    target_pos = sc_common.Point2D(
                        x=unit.pos.x + self._move_amount, y=unit.pos.y
                    )
                    action_num = 4
                else:  # west
                    target_pos = sc_common.Point2D(
                        x=unit.pos.x - self._move_amount, y=unit.pos.y
                    )
                    action_num = 5
            else:  # north or south
                if delta_y > 0:  # north
                    target_pos = sc_common.Point2D(
                        x=unit.pos.x, y=unit.pos.y + self._move_amount
                    )
                    action_num = 2
                else:  # south
                    target_pos = sc_common.Point2D(
                        x=unit.pos.x, y=unit.pos.y - self._move_amount
                    )
                    action_num = 3

            cmd = r_pb.ActionRawUnitCommand(
                ability_id=actions["move"],
                target_world_space_pos=target_pos,
                unit_tags=[tag],
                queue_command=False,
            )
        else:
            # Attack/heal the target
            cmd = r_pb.ActionRawUnitCommand(
                ability_id=action_id,
                target_unit_tag=target_tag,
                unit_tags=[tag],
                queue_command=False,
            )

        sc_action = sc_pb.Action(action_raw=r_pb.ActionRaw(unit_command=cmd))
        return sc_action, action_num

    def reward_battle(self):
        """Reward function when self.reward_spare==False.
        Returns accumulative hit/shield point damage dealt to the enemy
        + reward_death_value per enemy unit killed, and, in case
        self.reward_only_positive == False, - (damage dealt to ally units
        + reward_death_value per ally unit killed) * self.reward_negative_scale
        """
        if self.reward_sparse:
            return 0

        reward = 0
        delta_deaths = 0
        delta_ally = 0
        delta_enemy = 0

        neg_scale = self.reward_negative_scale

        # update deaths
        for al_id, al_unit in self.agents.items():
            if not self.death_tracker_ally[al_id]:
                # did not die so far
                prev_health = (
                    self.previous_ally_units[al_id].health
                    + self.previous_ally_units[al_id].shield
                )
                if al_unit.health == 0:
                    # just died
                    self.death_tracker_ally[al_id] = 1
                    if not self.reward_only_positive:
                        delta_deaths -= self.reward_death_value * neg_scale
                    delta_ally += prev_health * neg_scale
                else:
                    # still alive
                    delta_ally += neg_scale * (
                        prev_health - al_unit.health - al_unit.shield
                    )

        for e_id, e_unit in self.enemies.items():
            if not self.death_tracker_enemy[e_id]:
                prev_health = (
                    self.previous_enemy_units[e_id].health
                    + self.previous_enemy_units[e_id].shield
                )
                if e_unit.health == 0:
                    self.death_tracker_enemy[e_id] = 1
                    delta_deaths += self.reward_death_value
                    delta_enemy += prev_health
                else:
                    delta_enemy += prev_health - e_unit.health - e_unit.shield

        if self.reward_only_positive:
            reward = abs(delta_enemy + delta_deaths)  # shield regeneration
        else:
            reward = delta_enemy + delta_deaths - delta_ally

        return reward

    def get_total_actions(self):
        """Returns the total number of actions an agent could ever take."""
        return self.n_actions

    @staticmethod
    def distance(x1, y1, x2, y2):
        """Distance between two points."""
        return math.hypot(x2 - x1, y2 - y1)

    def unit_shoot_range(self, agent_id):
        """Returns the shooting range for an agent."""
        return 6

    def unit_sight_range(self, agent_id):
        """Returns the sight range for an agent."""
        return 9

    def unit_max_cooldown(self, unit):
        """Returns the maximal cooldown for a unit."""
        switcher = {
            self.marine_id: 15,
            self.marauder_id: 25,
            self.medivac_id: 200,  # max energy
            self.stalker_id: 35,
            self.zealot_id: 22,
            self.colossus_id: 24,
            self.hydralisk_id: 10,
            self.zergling_id: 11,
            self.baneling_id: 1,
        }
        return switcher.get(unit.unit_type, 15)

    def save_replay(self):
        """Save a replay."""
        prefix = self.replay_prefix or self.map_name
        replay_dir = self.replay_dir or ""
        replay_path = self._run_config.save_replay(
            self._controller.save_replay(),
            replay_dir=replay_dir,
            prefix=prefix,
        )
        logging.info("Replay saved at: %s" % replay_path)

    def unit_max_shield(self, unit):
        """Returns maximal shield for a given unit."""
        if unit.unit_type == 74 or unit.unit_type == self.stalker_id:
            return 80  # Protoss's Stalker
        if unit.unit_type == 73 or unit.unit_type == self.zealot_id:
            return 50  # Protoss's Zaelot
        if unit.unit_type == 4 or unit.unit_type == self.colossus_id:
            return 150  # Protoss's Colossus

    def can_move(self, unit, direction):
        """Whether a unit can move in a given direction."""
        m = self._move_amount / 2

        if direction == Direction.NORTH:
            x, y = int(unit.pos.x), int(unit.pos.y + m)
        elif direction == Direction.SOUTH:
            x, y = int(unit.pos.x), int(unit.pos.y - m)
        elif direction == Direction.EAST:
            x, y = int(unit.pos.x + m), int(unit.pos.y)
        else:
            x, y = int(unit.pos.x - m), int(unit.pos.y)

        if self.check_bounds(x, y) and self.pathing_grid[x, y]:
            return True

        return False

    def get_surrounding_points(self, unit, include_self=False):
        """Returns the surrounding points of the unit in 8 directions."""
        x = int(unit.pos.x)
        y = int(unit.pos.y)

        ma = self._move_amount

        points = [
            (x, y + 2 * ma),
            (x, y - 2 * ma),
            (x + 2 * ma, y),
            (x - 2 * ma, y),
            (x + ma, y + ma),
            (x - ma, y - ma),
            (x + ma, y - ma),
            (x - ma, y + ma),
        ]

        if include_self:
            points.append((x, y))

        return points

    def check_bounds(self, x, y):
        """Whether a point is within the map bounds."""
        return 0 <= x < self.map_x and 0 <= y < self.map_y

    def get_surrounding_pathing(self, unit):
        """Returns pathing values of the grid surrounding the given unit."""
        points = self.get_surrounding_points(unit, include_self=False)
        vals = [
            self.pathing_grid[x, y] if self.check_bounds(x, y) else 1
            for x, y in points
        ]
        return vals

    def get_surrounding_height(self, unit):
        """Returns height values of the grid surrounding the given unit."""
        points = self.get_surrounding_points(unit, include_self=True)
        vals = [
            self.terrain_height[x, y] if self.check_bounds(x, y) else 1
            for x, y in points
        ]
        return vals



    def delete_trace_mat(self, A):
        m = A.shape[0]
        strided = np.lib.stride_tricks.as_strided
        s0, s1 = A.strides
        out = strided(A.ravel()[1:], shape=(m - 1, m), strides=(s0 + s1, s1)).reshape(m, -1)

        return out

    def delete_multi_trace_mat(self, X):
        m = X.shape[0]
        strided = np.lib.stride_tricks.as_strided
        s0, s1 = X.strides
        out = strided(X.ravel()[self.unit_type_bits:], shape=(m - 1, m * self.unit_type_bits),
                      strides=(s0 + s1 * self.unit_type_bits, s1)).reshape(m, -1)

        return out

    def original_get_all_submatrix(self):
        sight_range = 9

        shoot_range = 6

        flag = False


        our_agents_position = np.mat(np.zeros((self.n_agents, 2), dtype=np.float64))
        enemy_agents_position = np.mat(np.zeros((self.n_enemies, 2), dtype=np.float32))
        with_rival_euclidean_distance = np.mat(np.zeros((self.n_agents, self.n_enemies), dtype=np.float32))
        with_ally_euclidean_distance = np.mat(np.zeros((self.n_agents, self.n_agents), dtype=np.float32))

        with_rival_relative_x = np.mat(np.zeros((self.n_agents, self.n_enemies), dtype=np.float32))
        with_rival_relative_y = np.mat(np.zeros((self.n_agents, self.n_enemies), dtype=np.float32))

        with_ally_relative_x = np.mat(np.zeros((self.n_agents, self.n_agents), dtype=np.float32))
        with_ally_relative_y = np.mat(np.zeros((self.n_agents, self.n_agents), dtype=np.float32))
        our_agents_health = np.mat(np.zeros((self.n_agents, 1), dtype=int))
        our_health_value = np.mat(np.zeros((self.n_agents, 1), dtype=np.float32))
        our_shield_value = np.mat(np.zeros((self.n_agents, 1), dtype=np.float32))
        our_type = np.mat(np.zeros((self.n_agents, self.unit_type_bits), dtype=np.float32))
        enemy_type = np.mat(np.zeros((self.n_enemies, self.unit_type_bits), dtype=np.float32))
        enemy_agents_health = np.mat(np.zeros((self.n_enemies, 1), dtype=int))
        enemy_health_value = np.mat(np.zeros((self.n_enemies, 1), dtype=np.float32))

        enemy_shield_value = np.mat(np.zeros((self.n_enemies, 1), dtype=np.float32))

        move_feats_dim = self.get_obs_move_feats_size()
        move_feats_mat = np.mat(np.zeros((self.n_agents, move_feats_dim), dtype=np.float32))

        attack_enemies_mat = np.mat(np.zeros((self.n_agents, self.n_enemies), dtype=np.float32))

        health_enemies_mat = []
        shield_enemies_mat = []

        enemy_type_mat = []

        our_type_mat = []
        medivac_indexs = []

        shield_ally_mat = []


        visible_mat = np.mat(np.zeros((self.n_agents, self.n_agents-1), dtype=np.float32))



        for e_id, e_unit in self.enemies.items():
            enemy_agents_position[e_id] = [e_unit.pos.x, e_unit.pos.y]

            if e_unit.health > 0 :
                enemy_agents_health[e_id] = 1
            enemy_health_value[e_id] = (e_unit.health / e_unit.health_max)
            if self.shield_bits_enemy > 0:
                max_shield = self.unit_max_shield(e_unit)
                enemy_shield_value[e_id] = (e_unit.shield / max_shield)

            if self.unit_type_bits > 0:
                type_id = self.get_unit_type_id(e_unit, False)
                enemy_type[e_id, type_id] = 1


        for a_id, a_unit in self.agents.items():
            our_agents_position[a_id] = [a_unit.pos.x, a_unit.pos.y]
            if self.can_move(a_unit, Direction.NORTH):
                move_feats_mat[a_id, 0] = 1
            if self.can_move(a_unit, Direction.SOUTH):
                move_feats_mat[a_id, 1] = 1
            if self.can_move(a_unit, Direction.EAST):
                move_feats_mat[a_id, 2] = 1
            if self.can_move(a_unit, Direction.WEST):
                move_feats_mat[a_id, 3] = 1

            if self.map_type == "MMM" and a_unit.unit_type == self.medivac_id:
                flag = True
                medivac_indexs.append(a_id)

            if a_unit.health > 0 :
                our_agents_health[a_id] = 1

            our_health_value[a_id] = (a_unit.health / a_unit.health_max)
            if self.shield_bits_ally > 0:
                max_shield = self.unit_max_shield(a_unit)
                our_shield_value[a_id] = (a_unit.shield / max_shield)

            if self.unit_type_bits > 0:
                type_id = self.get_unit_type_id(a_unit, True)
                our_type[a_id, type_id] = 1

        with_rival_euclidean_distance = np.sqrt(np.sum(np.power(our_agents_position, 2), 1) * np.ones((1, self.n_enemies)) + np.ones((self.n_agents, 1)) * np.sum(np.power(enemy_agents_position, 2),
                                                                                           1).T - 2 * our_agents_position * enemy_agents_position.T)


        with_ally_euclidean_distance = np.sqrt(np.sum(np.power(our_agents_position, 2), 1) * np.ones((1, self.n_agents)) +
                np.ones((self.n_agents, 1)) * np.sum(np.power(our_agents_position, 2),1).T - 2 * our_agents_position * our_agents_position.T)
        with_rival_relative_x = enemy_agents_position[:, 0].T - our_agents_position[:, 0] * np.ones((1, self.n_enemies))
        with_rival_relative_y = enemy_agents_position[:, 1].T - our_agents_position[:, 1] * np.ones((1, self.n_enemies))

        with_ally_relative_x = our_agents_position[:, 0].T - our_agents_position[:, 0] * np.ones((1, self.n_agents))
        with_ally_relative_y = our_agents_position[:, 1].T - our_agents_position[:, 1] * np.ones((1, self.n_agents))



        attack_enemies_mat[with_rival_euclidean_distance <= shoot_range] = 1
        if flag:
            temp_mask = with_rival_euclidean_distance[medivac_indexs, :]
            temp_mask[:,:] = 0
            temp_mask[:,0:self.n_agents] = with_ally_euclidean_distance[medivac_indexs, :]

            temp_mask[temp_mask <= shoot_range] = 1
            temp_mask[:,medivac_indexs] = 0
            temp_mask[:, self.n_agents:self.n_enemies] = 0
            temp_mask[temp_mask > shoot_range] = 0
            attack_enemies_mat[medivac_indexs, :] = temp_mask


        health_enemies_mat = np.ones((self.n_agents , 1)) * enemy_health_value.T
        if self.shield_bits_enemy > 0:
            shield_enemies_mat = np.ones((self.n_agents , 1)) * enemy_shield_value.T

        mask_dead_mat = np.ones((self.n_agents , 1)) * enemy_agents_health.T

        with_rival_relative_x = np.multiply(mask_dead_mat, with_rival_relative_x)
        with_rival_relative_y = np.multiply(mask_dead_mat, with_rival_relative_y)
        attack_enemies_mat = np.multiply(mask_dead_mat, attack_enemies_mat)
        health_enemies_mat = np.multiply(mask_dead_mat, health_enemies_mat)
        if self.shield_bits_enemy > 0:
            shield_enemies_mat = np.multiply(mask_dead_mat, shield_enemies_mat)
            shield_enemies_mat[with_rival_euclidean_distance >= sight_range] = 0
        if self.unit_type_bits > 0:

            repeat_mask_dead_mat = np.repeat(mask_dead_mat, self.unit_type_bits, axis=1)
            enemy_type_mat = np.ones((self.n_agents , 1)) * enemy_type.flatten()
            enemy_type_mat = np.multiply(repeat_mask_dead_mat, enemy_type_mat)
            repeat_with_rival_ed = np.repeat(with_rival_euclidean_distance, self.unit_type_bits, axis=1)
            enemy_type_mat[repeat_with_rival_ed >= sight_range] = 0


        health_enemies_mat[with_rival_euclidean_distance >= sight_range] = 0

        attack_enemies_mat[with_rival_euclidean_distance >= sight_range] = 0
        with_rival_relative_x[with_rival_euclidean_distance >= sight_range] = 0
        with_rival_relative_y[with_rival_euclidean_distance >= sight_range] = 0
        with_rival_euclidean_distance[with_rival_euclidean_distance >= sight_range] = 0
        with_rival_euclidean_distance = np.multiply(mask_dead_mat,
                                                    with_rival_euclidean_distance)


        health_ally_mat = (np.ones((self.n_agents, 1)) * our_health_value.T).A
        health_ally_mat = self.delete_trace_mat(health_ally_mat)


        with_ally_euclidean_distance = with_ally_euclidean_distance.A
        with_ally_euclidean_distance = self.delete_trace_mat(with_ally_euclidean_distance)

        with_ally_relative_x = with_ally_relative_x.A
        with_ally_relative_x = self.delete_trace_mat(with_ally_relative_x)

        with_ally_relative_y = with_ally_relative_y.A
        with_ally_relative_y = self.delete_trace_mat(with_ally_relative_y)

        mask_ally_dead_mat = np.ones((self.n_agents, 1)) * our_agents_health.T
        mask_ally_dead_mat = mask_ally_dead_mat.A

        mask_ally_dead_mat = self.delete_trace_mat(mask_ally_dead_mat)


        if self.shield_bits_ally > 0:
            shield_ally_mat = (np.ones((self.n_agents, 1)) * our_shield_value.T).A
            shield_ally_mat = self.delete_trace_mat(shield_ally_mat)
            shield_ally_mat[with_ally_euclidean_distance >= sight_range] = 0
            shield_ally_mat = np.mat(np.multiply(mask_ally_dead_mat, shield_ally_mat))



        if self.unit_type_bits > 0:

            repeat_mask_ally_dead_mat = np.repeat(mask_ally_dead_mat, self.unit_type_bits, axis=1)
            our_type_mat = (np.ones((self.n_agents, 1)) * our_type.flatten()).A
            our_type_mat = self.delete_multi_trace_mat(our_type_mat)
            our_type_mat = np.mat(np.multiply(repeat_mask_ally_dead_mat, our_type_mat))
            repeat_with_ally_ed = np.repeat(with_ally_euclidean_distance, self.unit_type_bits, axis=1)
            our_type_mat[repeat_with_ally_ed >= sight_range] = 0


        visible_mat[with_ally_euclidean_distance < sight_range] = 1


        with_ally_relative_x[with_ally_euclidean_distance >= sight_range] = 0
        with_ally_relative_y[with_ally_euclidean_distance >= sight_range] = 0

        health_ally_mat[with_ally_euclidean_distance >= sight_range] = 0


        with_ally_relative_x = np.multiply(mask_ally_dead_mat, with_ally_relative_x)
        with_ally_relative_y = np.multiply(mask_ally_dead_mat, with_ally_relative_y)
        visible_mat = np.multiply(mask_ally_dead_mat, visible_mat)
        health_ally_mat = np.mat(np.multiply(mask_ally_dead_mat, health_ally_mat))
        with_ally_euclidean_distance = np.multiply(mask_ally_dead_mat, with_ally_euclidean_distance)


        with_ally_euclidean_distance[with_ally_euclidean_distance >= sight_range] = 0



        own_feats_mat = our_health_value
        if self.shield_bits_ally > 0:
            own_feats_mat = np.append(own_feats_mat, our_shield_value, axis = 1)
        if self.unit_type_bits > 0:
            own_feats_mat = np.append(own_feats_mat, our_type, axis = 1)


        return with_rival_euclidean_distance, with_rival_relative_x, with_rival_relative_y, \
        with_ally_euclidean_distance, with_ally_relative_x, with_ally_relative_y, our_agents_health, enemy_agents_health, \
        move_feats_mat, attack_enemies_mat, health_enemies_mat, shield_enemies_mat, enemy_type_mat,\
               visible_mat, health_ally_mat, shield_ally_mat, our_type_mat, own_feats_mat


    def change_get_all_submatrix(self):
        sight_range = 9

        shoot_range = 6

        flag = False


        our_agents_position = np.zeros((self.n_agents, 2), dtype=np.float64)
        enemy_agents_position = np.zeros((self.n_enemies, 2), dtype=np.float32)
        with_rival_euclidean_distance = np.zeros((self.n_agents, self.n_enemies), dtype=np.float32)
        with_ally_euclidean_distance = np.zeros((self.n_agents, self.n_agents), dtype=np.float32)

        with_rival_relative_x = np.zeros((self.n_agents, self.n_enemies), dtype=np.float32)
        with_rival_relative_y = np.zeros((self.n_agents, self.n_enemies), dtype=np.float32)

        with_ally_relative_x = np.zeros((self.n_agents, self.n_agents), dtype=np.float32)
        with_ally_relative_y = np.zeros((self.n_agents, self.n_agents), dtype=np.float32)

        our_agents_health = np.zeros((1, self.n_agents), dtype=int)
        our_health_value = np.zeros((1, self.n_agents), dtype=np.float32)
        our_shield_value = np.zeros((1, self.n_agents), dtype=np.float32)

        our_type = np.zeros((self.n_agents, self.unit_type_bits), dtype=np.float32)

        enemy_type = np.zeros((self.n_enemies, self.unit_type_bits), dtype=np.float32)

        enemy_agents_health = np.zeros((1, self.n_enemies), dtype=int)
        enemy_health_value = np.zeros((1, self.n_enemies), dtype=np.float32)

        enemy_shield_value = np.zeros((1, self.n_enemies), dtype=np.float32)

        move_feats_dim = self.get_obs_move_feats_size()
        move_feats_mat = np.zeros((self.n_agents, move_feats_dim), dtype=np.float32)

        attack_enemies_mat = np.zeros((self.n_agents, self.n_enemies), dtype=np.float32)

        health_enemies_mat = []
        shield_enemies_mat = []

        enemy_type_mat = []

        our_type_mat = []
        medivac_indexs = []

        shield_ally_mat = []


        visible_mat = np.zeros((self.n_agents, self.n_agents-1), dtype=np.float32)

        start_time_for = time.time()

        for e_id, e_unit in self.enemies.items():
            enemy_agents_position[e_id] = [e_unit.pos.x, e_unit.pos.y]

            if e_unit.health > 0 :
                enemy_agents_health[0, e_id] = 1

            enemy_health_value[0, e_id] = (e_unit.health / e_unit.health_max)
            if self.shield_bits_enemy > 0:
                max_shield = self.unit_max_shield(e_unit)
                enemy_shield_value[0, e_id] = (e_unit.shield / max_shield)

            if self.unit_type_bits > 0:
                type_id = self.get_unit_type_id(e_unit, False)
                enemy_type[e_id, type_id] = 1


        for a_id, a_unit in self.agents.items():
            our_agents_position[a_id] = [a_unit.pos.x, a_unit.pos.y]

            if self.can_move(a_unit, Direction.NORTH):
                move_feats_mat[a_id, 0] = 1
                
                
            if self.can_move(a_unit, Direction.SOUTH):
                move_feats_mat[a_id, 1] = 1
            if self.can_move(a_unit, Direction.EAST):
                move_feats_mat[a_id, 2] = 1
            if self.can_move(a_unit, Direction.WEST):
                move_feats_mat[a_id, 3] = 1

            if self.map_type == "MMM" and a_unit.unit_type == self.medivac_id:
                flag = True
                medivac_indexs.append(a_id)

            if a_unit.health > 0 :
                our_agents_health[0, a_id] = 1

            our_health_value[0, a_id] = (a_unit.health / a_unit.health_max)
            if self.shield_bits_ally > 0:
                max_shield = self.unit_max_shield(a_unit)
                our_shield_value[0, a_id] = (a_unit.shield / max_shield)

            if self.unit_type_bits > 0:
                type_id = self.get_unit_type_id(a_unit, True)
                our_type[a_id, type_id] = 1



        end_time_for = time.time()

        with_rival_euclidean_distance = spatial.distance.cdist(our_agents_position, enemy_agents_position, "euclidean")
        with_ally_euclidean_distance = spatial.distance.cdist(our_agents_position, our_agents_position, "euclidean")


        with_rival_relative_x = enemy_agents_position[:, 0].reshape(1, self.n_enemies) - our_agents_position[:, 0].reshape(self.n_agents, 1)
        with_rival_relative_y = enemy_agents_position[:, 1].reshape(1, self.n_enemies) - our_agents_position[:, 1].reshape(self.n_agents, 1)


        with_ally_relative_x = our_agents_position[:, 0].reshape(1, self.n_agents) - our_agents_position[:, 0].reshape(self.n_agents, 1)
        with_ally_relative_y = our_agents_position[:, 1].reshape(1, self.n_agents) - our_agents_position[:, 1].reshape(self.n_agents, 1)


        attack_enemies_mat[with_rival_euclidean_distance <= shoot_range] = 1
        if flag:
            temp_mask = with_rival_euclidean_distance[medivac_indexs, :]
            temp_mask[:,:] = 0
            temp_mask[:,0:self.n_agents] = with_ally_euclidean_distance[medivac_indexs, :]

            temp_mask[temp_mask <= shoot_range] = 1
            temp_mask[:,medivac_indexs] = 0
            temp_mask[:, self.n_agents:self.n_enemies] = 0
            temp_mask[temp_mask > shoot_range] = 0
            attack_enemies_mat[medivac_indexs, :] = temp_mask

        health_enemies_mat = np.tile(enemy_health_value, (self.n_agents, 1))
        if self.shield_bits_enemy > 0:
            shield_enemies_mat = np.tile(enemy_shield_value, (self.n_agents, 1))


        enemy_sight_flag = (with_rival_euclidean_distance >= sight_range)




        with_rival_relative_x = with_rival_relative_x * enemy_agents_health
        with_rival_relative_y = with_rival_relative_y * enemy_agents_health
        attack_enemies_mat = attack_enemies_mat * enemy_agents_health
        health_enemies_mat = health_enemies_mat * enemy_agents_health
        if self.shield_bits_enemy > 0:
            shield_enemies_mat = shield_enemies_mat * enemy_agents_health
            shield_enemies_mat[enemy_sight_flag] = 0
        if self.unit_type_bits > 0:

            mask_dead_mat = np.dot(np.ones((self.n_agents , 1)), enemy_agents_health)
            repeat_mask_dead_mat = np.repeat(mask_dead_mat, self.unit_type_bits, axis=1)
            enemy_type_mat = np.ones((self.n_agents , 1)) * enemy_type.flatten()
            enemy_type_mat = np.multiply(repeat_mask_dead_mat, enemy_type_mat)

            repeat_with_rival_ed = np.repeat(with_rival_euclidean_distance, self.unit_type_bits, axis=1)
            enemy_type_mat[repeat_with_rival_ed >= sight_range] = 0



        health_enemies_mat[enemy_sight_flag] = 0

        attack_enemies_mat[enemy_sight_flag] = 0
        with_rival_relative_x[enemy_sight_flag] = 0
        with_rival_relative_y[enemy_sight_flag] = 0
        with_rival_euclidean_distance[enemy_sight_flag] = 0



        with_rival_euclidean_distance = with_rival_euclidean_distance * enemy_agents_health



        health_ally_mat = np.tile(our_health_value, (self.n_agents, 1))
        health_ally_mat = self.delete_trace_mat(health_ally_mat)


        with_ally_euclidean_distance = self.delete_trace_mat(with_ally_euclidean_distance)

        with_ally_relative_x = self.delete_trace_mat(with_ally_relative_x)

        with_ally_relative_y = self.delete_trace_mat(with_ally_relative_y)

        mask_ally_dead_mat = np.tile(our_agents_health, (self.n_agents, 1))

        mask_ally_dead_mat = self.delete_trace_mat(mask_ally_dead_mat)





        ally_sight_flag = (with_ally_euclidean_distance >= sight_range)



        if self.shield_bits_ally > 0:
            shield_ally_mat = np.tile(our_shield_value, (self.n_agents, 1))
            shield_ally_mat = self.delete_trace_mat(shield_ally_mat)
            shield_ally_mat[ally_sight_flag] = 0
            shield_ally_mat = np.multiply(mask_ally_dead_mat, shield_ally_mat)



        if self.unit_type_bits > 0:

            repeat_mask_ally_dead_mat = np.repeat(mask_ally_dead_mat, self.unit_type_bits, axis=1)
            our_type_mat = np.dot(np.ones((self.n_agents, 1)), our_type.reshape(1, -1))
            our_type_mat = self.delete_multi_trace_mat(our_type_mat)

            our_type_mat = np.multiply(repeat_mask_ally_dead_mat, our_type_mat)

            repeat_with_ally_ed = np.repeat(with_ally_euclidean_distance, self.unit_type_bits, axis=1)
            our_type_mat[repeat_with_ally_ed >= sight_range] = 0



        visible_mat[~ally_sight_flag] = 1



        with_ally_relative_x[ally_sight_flag] = 0
        with_ally_relative_y[ally_sight_flag] = 0

        health_ally_mat[ally_sight_flag] = 0


        with_ally_relative_x = np.multiply(mask_ally_dead_mat, with_ally_relative_x)
        with_ally_relative_y = np.multiply(mask_ally_dead_mat, with_ally_relative_y)
        visible_mat = np.multiply(mask_ally_dead_mat, visible_mat)
        health_ally_mat = np.multiply(mask_ally_dead_mat, health_ally_mat)
        with_ally_euclidean_distance = np.multiply(mask_ally_dead_mat, with_ally_euclidean_distance)


        with_ally_euclidean_distance[ally_sight_flag] = 0


        own_feats_mat = our_health_value.reshape(-1, 1)
        if self.shield_bits_ally > 0:
            own_feats_mat = np.append(own_feats_mat, our_shield_value.reshape(-1, 1), axis = 1)
        if self.unit_type_bits > 0:
            own_feats_mat = np.append(own_feats_mat, our_type, axis = 1)


        return with_rival_euclidean_distance, with_rival_relative_x, with_rival_relative_y, \
        with_ally_euclidean_distance, with_ally_relative_x, with_ally_relative_y, our_agents_health, enemy_agents_health, \
        move_feats_mat, attack_enemies_mat, health_enemies_mat, shield_enemies_mat, enemy_type_mat,\
               visible_mat, health_ally_mat, shield_ally_mat, our_type_mat, own_feats_mat


    def change_get_agents_obs(self):


        start_time_create = time.time()
        sight_range = 9

        move_feats_dim = self.get_obs_move_feats_size()
        enemy_number, n_feature = self.get_obs_enemy_feats_size()
        enemy_feats_dim = enemy_number * n_feature
        n_allies, nf_al= self.get_obs_ally_feats_size()
        ally_feats_dim = n_allies * nf_al
        own_feats_dim = self.get_obs_own_feats_size()

        agent_obs_dim = move_feats_dim + enemy_feats_dim + ally_feats_dim + own_feats_dim


        enemy_feats_mat = np.zeros((self.n_agents, enemy_feats_dim), dtype=np.float32)
        ally_feats_mat = np.zeros((self.n_agents, ally_feats_dim), dtype=np.float32)

        end_time_create = time.time()


        start_time_submatrix = time.time()

        with_rival_euclidean_distance, with_rival_relative_x, with_rival_relative_y, \
        with_ally_euclidean_distance, with_ally_relative_x, with_ally_relative_y, our_agents_health, enemy_agents_health, \
        move_feats_mat, attack_enemies_mat, health_enemies_mat, shield_enemies_mat, enemy_type_mat, \
        visible_mat, health_ally_mat, shield_ally_mat, our_type_mat, own_feats_mat = self.change_get_all_submatrix()


        with_rival_euclidean_distance = with_rival_euclidean_distance / sight_range
        with_rival_relative_x = with_rival_relative_x / sight_range
        with_rival_relative_y = with_rival_relative_y / sight_range

        end_time_submatrix = time.time()

        start_time_index = time.time()
        attack_index = [i * n_feature for i in range(self.n_enemies)]
        ed_index = [i * n_feature + 1 for i in range(self.n_enemies)]
        rx_index = [i * n_feature + 2 for i in range(self.n_enemies)]
        ry_index = [i * n_feature + 3 for i in range(self.n_enemies)]
        health_index = [i * n_feature + 4 for i in range(self.n_enemies)]
        if self.shield_bits_enemy > 0 and self.unit_type_bits > 0:
            shield_index = [i * n_feature + 5 for i in range(self.n_enemies)]
            type_index = [i * n_feature + 6 + j for i in range(self.n_enemies) for j in range(self.unit_type_bits)]
            enemy_feats_mat[:, shield_index] = shield_enemies_mat
            enemy_feats_mat[:, type_index] = enemy_type_mat
        elif self.shield_bits_enemy > 0:
            shield_index = [i * n_feature + 5 for i in range(self.n_enemies)]
            enemy_feats_mat[:, shield_index] = shield_enemies_mat
        elif self.unit_type_bits > 0:
            type_index = [i * n_feature + 5 + j for i in range(self.n_enemies) for j in range(self.unit_type_bits)]
            enemy_feats_mat[:, type_index] = enemy_type_mat
        enemy_feats_mat[:, attack_index] = attack_enemies_mat
        enemy_feats_mat[:, ed_index] = with_rival_euclidean_distance
        enemy_feats_mat[:, rx_index] = with_rival_relative_x
        enemy_feats_mat[:, ry_index] = with_rival_relative_y
        enemy_feats_mat[:, health_index] = health_enemies_mat
        end_time_index = time.time()


        with_ally_euclidean_distance = with_ally_euclidean_distance / sight_range
        with_ally_relative_x = with_ally_relative_x / sight_range
        with_ally_relative_y = with_ally_relative_y / sight_range


        start_time_index_ally = time.time()
        visible_index = [i * nf_al for i in range(self.n_agents-1)]
        ed_index_ally = [i * nf_al + 1 for i in range(self.n_agents-1)]
        rx_index_ally = [i * nf_al + 2 for i in range(self.n_agents-1)]
        ry_index_ally = [i * nf_al + 3 for i in range(self.n_agents-1)]
        health_index_ally = [i * nf_al + 4 for i in range(self.n_agents-1)]
        if self.shield_bits_ally > 0 and self.unit_type_bits > 0:
            shield_index_ally = [i * nf_al + 5 for i in range(self.n_agents-1)]
            type_index = [i * nf_al + 6 + j for i in range(self.n_agents-1) for j in range(self.unit_type_bits)]
            ally_feats_mat[:, shield_index_ally] = shield_ally_mat
            ally_feats_mat[:, type_index] = our_type_mat
        elif self.shield_bits_ally > 0:
            shield_index_ally = [i * nf_al + 5 for i in range(self.n_agents-1)]
            ally_feats_mat[:, shield_index_ally] = shield_ally_mat
        elif self.unit_type_bits > 0:
            type_index = [i * nf_al + 5 + j for i in range(self.n_agents-1) for j in range(self.unit_type_bits)]
            ally_feats_mat[:, type_index] = our_type_mat
        ally_feats_mat[:, visible_index] = visible_mat
        ally_feats_mat[:, ed_index_ally] = with_ally_euclidean_distance
        ally_feats_mat[:, rx_index_ally] = with_ally_relative_x
        ally_feats_mat[:, ry_index_ally] = with_ally_relative_y
        ally_feats_mat[:, health_index_ally] = health_ally_mat
        end_time_index_ally = time.time()

        start_time_end_joint = time.time()
        agent_obs = np.concatenate((move_feats_mat, enemy_feats_mat, ally_feats_mat, own_feats_mat),axis = 1)
        agent_obs = agent_obs * our_agents_health.reshape(self.n_agents , 1)
        end_time_end_joint = time.time()



        return agent_obs

    def original_change_get_agents_obs(self):



        sight_range = 9


        move_feats_dim = self.get_obs_move_feats_size()
        enemy_number, n_feature = self.get_obs_enemy_feats_size()
        enemy_feats_dim = enemy_number * n_feature
        n_allies, nf_al = self.get_obs_ally_feats_size()
        ally_feats_dim = n_allies * nf_al
        own_feats_dim = self.get_obs_own_feats_size()

        agent_obs_dim = move_feats_dim + enemy_feats_dim + ally_feats_dim + own_feats_dim


        enemy_feats_mat = np.mat(np.zeros((self.n_agents, enemy_feats_dim), dtype=np.float32))
        ally_feats_mat = np.mat(np.zeros((self.n_agents, ally_feats_dim), dtype=np.float32))

        with_rival_euclidean_distance, with_rival_relative_x, with_rival_relative_y, \
        with_ally_euclidean_distance, with_ally_relative_x, with_ally_relative_y, our_agents_health, enemy_agents_health, \
        move_feats_mat, attack_enemies_mat, health_enemies_mat, shield_enemies_mat, enemy_type_mat, \
        visible_mat, health_ally_mat, shield_ally_mat, our_type_mat, own_feats_mat = self.original_get_all_submatrix()


        with_rival_euclidean_distance = with_rival_euclidean_distance / sight_range
        with_rival_relative_x = with_rival_relative_x / sight_range
        with_rival_relative_y = with_rival_relative_y / sight_range


        attack_index = [i * n_feature for i in range(self.n_enemies)]
        ed_index = [i * n_feature + 1 for i in range(self.n_enemies)]
        rx_index = [i * n_feature + 2 for i in range(self.n_enemies)]
        ry_index = [i * n_feature + 3 for i in range(self.n_enemies)]
        health_index = [i * n_feature + 4 for i in range(self.n_enemies)]
        if self.shield_bits_enemy > 0 and self.unit_type_bits > 0:
            shield_index = [i * n_feature + 5 for i in range(self.n_enemies)]
            type_index = [i * n_feature + 6 + j for i in range(self.n_enemies) for j in range(self.unit_type_bits)]
            enemy_feats_mat[:, shield_index] = shield_enemies_mat
            enemy_feats_mat[:, type_index] = enemy_type_mat
        elif self.shield_bits_enemy > 0:
            shield_index = [i * n_feature + 5 for i in range(self.n_enemies)]
            enemy_feats_mat[:, shield_index] = shield_enemies_mat
        elif self.unit_type_bits > 0:
            type_index = [i * n_feature + 5 + j for i in range(self.n_enemies) for j in range(self.unit_type_bits)]
            enemy_feats_mat[:, type_index] = enemy_type_mat
        enemy_feats_mat[:, attack_index] = attack_enemies_mat
        enemy_feats_mat[:, ed_index] = with_rival_euclidean_distance
        enemy_feats_mat[:, rx_index] = with_rival_relative_x
        enemy_feats_mat[:, ry_index] = with_rival_relative_y
        enemy_feats_mat[:, health_index] = health_enemies_mat

        with_ally_euclidean_distance = np.mat(with_ally_euclidean_distance / sight_range)
        with_ally_relative_x = np.mat(with_ally_relative_x / sight_range)
        with_ally_relative_y = np.mat(with_ally_relative_y / sight_range)

        visible_index = [i * nf_al for i in range(self.n_agents - 1)]
        ed_index_ally = [i * nf_al + 1 for i in range(self.n_agents - 1)]
        rx_index_ally = [i * nf_al + 2 for i in range(self.n_agents - 1)]
        ry_index_ally = [i * nf_al + 3 for i in range(self.n_agents - 1)]
        health_index_ally = [i * nf_al + 4 for i in range(self.n_agents - 1)]
        if self.shield_bits_ally > 0 and self.unit_type_bits > 0:
            shield_index_ally = [i * nf_al + 5 for i in range(self.n_agents - 1)]
            type_index = [i * nf_al + 6 + j for i in range(self.n_agents - 1) for j in range(self.unit_type_bits)]
            ally_feats_mat[:, shield_index_ally] = shield_ally_mat
            ally_feats_mat[:, type_index] = our_type_mat
        elif self.shield_bits_ally > 0:
            shield_index_ally = [i * nf_al + 5 for i in range(self.n_agents - 1)]
            ally_feats_mat[:, shield_index_ally] = shield_ally_mat
        elif self.unit_type_bits > 0:
            type_index = [i * nf_al + 5 + j for i in range(self.n_agents - 1) for j in range(self.unit_type_bits)]
            ally_feats_mat[:, type_index] = our_type_mat
        ally_feats_mat[:, visible_index] = visible_mat
        ally_feats_mat[:, ed_index_ally] = with_ally_euclidean_distance
        ally_feats_mat[:, rx_index_ally] = with_ally_relative_x
        ally_feats_mat[:, ry_index_ally] = with_ally_relative_y
        ally_feats_mat[:, health_index_ally] = health_ally_mat

        agent_obs = np.concatenate((move_feats_mat, enemy_feats_mat, ally_feats_mat, own_feats_mat), axis=1)

        our_agents_health = our_agents_health * np.ones((1, agent_obs_dim))

        agent_obs = np.multiply(our_agents_health, agent_obs)


        return agent_obs


    def get_obs_agent(self, agent_id):
        """Returns observation for agent_id. The observation is composed of:

        - agent movement features (where it can move to, height information
            and pathing grid)
        - enemy features (available_to_attack, health, relative_x, relative_y,
            shield, unit_type)
        - ally features (visible, distance, relative_x, relative_y, shield,
            unit_type)
        - agent unit features (health, shield, unit_type)

        All of this information is flattened and concatenated into a list,
        in the aforementioned order. To know the sizes of each of the
        features inside the final list of features, take a look at the
        functions ``get_obs_move_feats_size()``,
        ``get_obs_enemy_feats_size()``, ``get_obs_ally_feats_size()`` and
        ``get_obs_own_feats_size()``.

        The size of the observation vector may vary, depending on the
        environment configuration and type of units present in the map.
        For instance, non-Protoss units will not have shields, movement
        features may or may not include terrain height and pathing grid,
        unit_type is not included if there is only one type of unit in the
        map etc.).

        NOTE: Agents should have access only to their local observations
        during decentralised execution.
        """
        unit = self.get_unit_by_id(agent_id)

        move_feats_dim = self.get_obs_move_feats_size()
        enemy_feats_dim = self.get_obs_enemy_feats_size()
        ally_feats_dim = self.get_obs_ally_feats_size()
        own_feats_dim = self.get_obs_own_feats_size()

        move_feats = np.zeros(move_feats_dim, dtype=np.float32)
        enemy_feats = np.zeros(enemy_feats_dim, dtype=np.float32)
        ally_feats = np.zeros(ally_feats_dim, dtype=np.float32)
        own_feats = np.zeros(own_feats_dim, dtype=np.float32)
        if unit.health > 0:
            x = unit.pos.x
            y = unit.pos.y
            sight_range = self.unit_sight_range(agent_id)


            avail_actions = self.get_avail_agent_actions(agent_id)
            for m in range(self.n_actions_move):
                move_feats[m] = avail_actions[m + 2]

            ind = self.n_actions_move

            if self.obs_pathing_grid:
                move_feats[
                    ind : ind + self.n_obs_pathing
                ] = self.get_surrounding_pathing(unit)
                ind += self.n_obs_pathing

            if self.obs_terrain_height:
                move_feats[ind:] = self.get_surrounding_height(unit)

            for e_id, e_unit in self.enemies.items():
                e_x = e_unit.pos.x
                e_y = e_unit.pos.y
                dist = self.distance(x, y, e_x, e_y)

                if (
                    dist < sight_range and e_unit.health > 0
                ):
                    enemy_feats[e_id, 0] = avail_actions[
                        self.n_actions_no_attack + e_id
                    ]
                    enemy_feats[e_id, 1] = dist / sight_range
                    enemy_feats[e_id, 2] = (
                        e_x - x
                    ) / sight_range
                    enemy_feats[e_id, 3] = (
                        e_y - y
                    ) / sight_range

                    ind = 4
                    if self.obs_all_health:
                        enemy_feats[e_id, ind] = (
                            e_unit.health / e_unit.health_max
                        )
                        ind += 1
                        if self.shield_bits_enemy > 0:
                            max_shield = self.unit_max_shield(e_unit)
                            enemy_feats[e_id, ind] = (
                                e_unit.shield / max_shield
                            )
                            ind += 1

                    if self.unit_type_bits > 0:
                        type_id = self.get_unit_type_id(e_unit, False)
                        enemy_feats[e_id, ind + type_id] = 1

            al_ids = [
                al_id for al_id in range(self.n_agents) if al_id != agent_id
            ]

            for i, al_id in enumerate(al_ids):

                al_unit = self.get_unit_by_id(al_id)
                al_x = al_unit.pos.x
                al_y = al_unit.pos.y
                dist = self.distance(x, y, al_x, al_y)

                if (
                    dist < sight_range and al_unit.health > 0
                ):
                    ally_feats[i, 0] = 1
                    ally_feats[i, 1] = dist / sight_range
                    ally_feats[i, 2] = (al_x - x) / sight_range
                    ally_feats[i, 3] = (al_y - y) / sight_range

                    ind = 4
                    if self.obs_all_health:
                        ally_feats[i, ind] = (
                            al_unit.health / al_unit.health_max
                        )
                        ind += 1
                        if self.shield_bits_ally > 0:
                            max_shield = self.unit_max_shield(al_unit)
                            ally_feats[i, ind] = (
                                al_unit.shield / max_shield
                            )
                            ind += 1


                    if self.unit_type_bits > 0:
                        type_id = self.get_unit_type_id(al_unit, True)
                        ally_feats[i, ind + type_id] = 1
                        ind += self.unit_type_bits

                    if self.obs_last_action:
                        ally_feats[i, ind:] = self.last_action[al_id]

            ind = 0
            if self.obs_own_health:
                own_feats[ind] = unit.health / unit.health_max
                ind += 1
                if self.shield_bits_ally > 0:
                    max_shield = self.unit_max_shield(unit)
                    own_feats[ind] = unit.shield / max_shield
                    ind += 1
            if self.unit_type_bits > 0:
                type_id = self.get_unit_type_id(unit, True)
                own_feats[ind + type_id] = 1

        agent_obs = np.concatenate(
            (
                move_feats.flatten(),
                enemy_feats.flatten(),
                ally_feats.flatten(),
                own_feats.flatten(),
            )
        )

        if self.obs_timestep_number:
            agent_obs = np.append(
                agent_obs, self._episode_steps / self.episode_limit
            )

        if self.debug:
            logging.debug("Obs Agent: {}".format(agent_id).center(60, "-"))
            logging.debug(
                "Avail. actions {}".format(
                    self.get_avail_agent_actions(agent_id)
                )
            )
            logging.debug("Move feats {}".format(move_feats))
            logging.debug("Enemy feats {}".format(enemy_feats))
            logging.debug("Ally feats {}".format(ally_feats))
            logging.debug("Own feats {}".format(own_feats))

        return agent_obs







    def get_obs(self):
        """Returns all agent observations in a list.
        NOTE: Agents should have access only to their local observations
        during decentralised execution.
        """


        start_time2 = time.time()
        agents_obs = [self.get_obs_agent(i) for i in range(self.n_agents)]
        end_time2 = time.time()
        difference3 = end_time2 - start_time2
        self.original_array = np.append(self.original_array, difference3)


        start_time = time.time()
        orginal_change_agents_obs = self.original_change_get_agents_obs()
        change_agents_obs_format = [np.squeeze(orginal_change_agents_obs[i, :].A) for i in range(self.n_agents)]
        end_time = time.time()
        difference1 = end_time - start_time

        self.matrix_array = np.append(self.matrix_array, difference1)
        start_time1 = time.time()
        change_agents_obs = self.change_get_agents_obs()
        change_agents_obs_format1 = [np.squeeze(change_agents_obs[i, :]) for i in range(self.n_agents)]
        end_time1 = time.time()
        difference2 = end_time1 - start_time1
        self.improve_matrix_array = np.append(self.improve_matrix_array, difference2)

        start_time3 = time.time()

        survival_num_allys = self.n_agents - np.sum(self.death_tracker_ally)

        if survival_num_allys < 6:
            agents_obs = [self.get_obs_agent(i) for i in range(self.n_agents)]
        else:
            change_agents_obs = self.change_get_agents_obs()
            change_agents_obs_format1 = [np.squeeze(change_agents_obs[i, :]) for i in range(self.n_agents)]
        end_time3 = time.time()
        difference4 = end_time3 - start_time3
        self.comprehensive_array = np.append(self.comprehensive_array, difference4)

        self.times_array = np.append(self.times_array, difference3 / difference2)
        self.comprehensive_times_array = np.append(self.comprehensive_times_array, difference3 / difference4)



        survival_num_enemies = self.n_enemies - np.sum(self.death_tracker_enemy)
        if (difference3 / difference2) < 1:
            self.survival_enemies = np.append(self.survival_enemies, survival_num_enemies)
            self.survival_allys = np.append(self.survival_allys, survival_num_allys)
        else:
            self.survival_enemies = np.append(self.survival_enemies, 0)
            self.survival_allys = np.append(self.survival_allys, 0)


        return agents_obs



    def get_obs_original(self):
        agents_obs = [self.get_obs_agent(i) for i in range(self.n_agents)]
        return agents_obs


    def get_cut_down_obs_agent(self, agent_id):
        '''
        :param agent_id:
        :return:
        '''
        unit = self.get_unit_by_id(agent_id)

        move_feats_dim = self.get_obs_move_feats_size()

        n_enemies, n_enemy_feats = self.get_obs_enemy_feats_size()

        cut_down_n_enemies = int(n_enemies/5)
        enemy_feats_dim = (cut_down_n_enemies,n_enemy_feats)


        n_allies, n_ally_feats = self.get_obs_ally_feats_size()

        cut_down_n_allies = int(n_allies / 5)
        ally_feats_dim = (cut_down_n_allies,n_ally_feats)
        own_feats_dim = self.get_obs_own_feats_size()

        move_feats = np.zeros(move_feats_dim, dtype=np.float32)
        enemy_feats = np.zeros(enemy_feats_dim, dtype=np.float32)
        ally_feats = np.zeros(ally_feats_dim, dtype=np.float32)
        own_feats = np.zeros(own_feats_dim, dtype=np.float32)

        enemy_dist = np.zeros(n_enemies, dtype=np.float)
        e_id_list = []

        ally_dist = np.zeros(self.n_agents, dtype=np.float)
        al_id_list = []


        if unit.health > 0:
            x = unit.pos.x
            y = unit.pos.y
            sight_range = self.unit_sight_range(agent_id)

            avail_actions = self.get_avail_agent_actions(agent_id)
            for m in range(self.n_actions_move):
                move_feats[m] = avail_actions[m + 2]

            ind = self.n_actions_move

            if self.obs_pathing_grid:
                move_feats[
                    ind : ind + self.n_obs_pathing
                ] = self.get_surrounding_pathing(unit)
                ind += self.n_obs_pathing

            if self.obs_terrain_height:
                move_feats[ind:] = self.get_surrounding_height(unit)

            for e_id, e_unit in self.enemies.items():
                e_x = e_unit.pos.x
                e_y = e_unit.pos.y
                dist = self.distance(x, y, e_x, e_y)
                enemy_dist[e_id] = dist
            temp_list = deepcopy(enemy_dist)

            for i in range(cut_down_n_enemies):
                index = temp_list.argmin()
                e_id_list.append(index)
                temp_list[index] = float('inf')

            for i,e_id in enumerate(e_id_list):

                e_unit = self.enemies[e_id]
                dist = enemy_dist[e_id]
                e_x = e_unit.pos.x
                e_y = e_unit.pos.y
                if (
                    dist < sight_range and e_unit.health > 0
                ):

                    enemy_feats[i, 0] = avail_actions[
                        self.n_actions_no_attack + e_id
                    ]
                    enemy_feats[i, 1] = dist / sight_range
                    enemy_feats[i, 2] = (
                        e_x - x
                    ) / sight_range
                    enemy_feats[i, 3] = (
                        e_y - y
                    ) / sight_range

                    ind = 4
                    if self.obs_all_health:
                        enemy_feats[i, ind] = (
                            e_unit.health / e_unit.health_max
                        )
                        ind += 1
                        if self.shield_bits_enemy > 0:
                            max_shield = self.unit_max_shield(e_unit)
                            enemy_feats[i, ind] = (
                                e_unit.shield / max_shield
                            )
                            ind += 1

                    if self.unit_type_bits > 0:
                        type_id = self.get_unit_type_id(e_unit, False)
                        enemy_feats[i, ind + type_id] = 1

            for al_id, al_unit in self.agents.items():
                al_x = al_unit.pos.x
                al_y = al_unit.pos.y
                if al_id == agent_id:
                    ally_dist[al_id] = float('inf')
                    continue
                dist = self.distance(x, y, al_x, al_y)
                ally_dist[al_id] = dist


            temp_list2 = deepcopy(ally_dist)
            for i in range(cut_down_n_allies):
                index = temp_list2.argmin()
                al_id_list.append(index)
                temp_list2[index] = float('inf')

            for i, al_id in enumerate(al_id_list):

                al_unit = self.get_unit_by_id(al_id)
                al_x = al_unit.pos.x
                al_y = al_unit.pos.y
                dist = ally_dist[al_id]

                if (
                    dist < sight_range and al_unit.health > 0
                ):
                    ally_feats[i, 0] = 1
                    ally_feats[i, 1] = dist / sight_range
                    ally_feats[i, 2] = (al_x - x) / sight_range
                    ally_feats[i, 3] = (al_y - y) / sight_range

                    ind = 4
                    if self.obs_all_health:
                        ally_feats[i, ind] = (
                            al_unit.health / al_unit.health_max
                        )  # health
                        ind += 1
                        if self.shield_bits_ally > 0:
                            max_shield = self.unit_max_shield(al_unit)
                            ally_feats[i, ind] = (
                                al_unit.shield / max_shield
                            )  # shield
                            ind += 1


                    if self.unit_type_bits > 0:
                        type_id = self.get_unit_type_id(al_unit, True)
                        ally_feats[i, ind + type_id] = 1
                        ind += self.unit_type_bits

                    if self.obs_last_action:
                        ally_feats[i, ind:] = self.last_action[al_id]


            ind = 0
            if self.obs_own_health:
                own_feats[ind] = unit.health / unit.health_max
                ind += 1
                if self.shield_bits_ally > 0:
                    max_shield = self.unit_max_shield(unit)
                    own_feats[ind] = unit.shield / max_shield
                    ind += 1
            if self.unit_type_bits > 0:
                type_id = self.get_unit_type_id(unit, True)
                own_feats[ind + type_id] = 1

        agent_obs = np.concatenate(
            (
                move_feats.flatten(),
                enemy_feats.flatten(),
                ally_feats.flatten(),
                own_feats.flatten(),
            )
        )

        if self.obs_timestep_number:
            agent_obs = np.append(
                agent_obs, self._episode_steps / self.episode_limit
            )

        if self.debug:
            logging.debug("Obs Agent: {}".format(agent_id).center(60, "-"))
            logging.debug(
                "Avail. actions {}".format(
                    self.get_avail_agent_actions(agent_id)
                )
            )
            logging.debug("Move feats {}".format(move_feats))
            logging.debug("Enemy feats {}".format(enemy_feats))
            logging.debug("Ally feats {}".format(ally_feats))
            logging.debug("Own feats {}".format(own_feats))

        return agent_obs

    def get_cut_down_five_agent(self, agent_id):
        '''
        :param agent_id:
        :return:
        '''
        unit = self.get_unit_by_id(agent_id)

        move_feats_dim = self.get_obs_move_feats_size()

        n_enemies, n_enemy_feats = self.get_obs_enemy_feats_size()

        cut_down_n_enemies = 5
        enemy_feats_dim = (cut_down_n_enemies,n_enemy_feats)


        n_allies, n_ally_feats = self.get_obs_ally_feats_size()

        cut_down_n_allies = 5
        ally_feats_dim = (cut_down_n_allies,n_ally_feats)
        own_feats_dim = self.get_obs_own_feats_size()

        move_feats = np.zeros(move_feats_dim, dtype=np.float32)
        enemy_feats = np.zeros(enemy_feats_dim, dtype=np.float32)
        ally_feats = np.zeros(ally_feats_dim, dtype=np.float32)
        own_feats = np.zeros(own_feats_dim, dtype=np.float32)

        enemy_dist = np.zeros(n_enemies, dtype=np.float)
        e_id_list = []

        ally_dist = np.zeros(self.n_agents, dtype=np.float)
        al_id_list = []


        if unit.health > 0:
            x = unit.pos.x
            y = unit.pos.y
            sight_range = self.unit_sight_range(agent_id)


            avail_actions = self.get_avail_agent_actions(agent_id)
            for m in range(self.n_actions_move):
                move_feats[m] = avail_actions[m + 2]

            ind = self.n_actions_move

            if self.obs_pathing_grid:
                move_feats[
                    ind : ind + self.n_obs_pathing
                ] = self.get_surrounding_pathing(unit)
                ind += self.n_obs_pathing

            if self.obs_terrain_height:
                move_feats[ind:] = self.get_surrounding_height(unit)

            for e_id, e_unit in self.enemies.items():
                e_x = e_unit.pos.x
                e_y = e_unit.pos.y
                dist = self.distance(x, y, e_x, e_y)
                enemy_dist[e_id] = dist
            temp_list = deepcopy(enemy_dist)

            full_flag = False
            for i in range(cut_down_n_enemies):
                index = temp_list.argmin()
                e_unit = self.enemies[index]
                if e_unit.health <= 0:
                    temp_list[index] = float('inf')
                    while True:
                        index = temp_list.argmin()
                        e_unit = self.enemies[index]
                        if temp_list[index] == float('inf'):
                            full_flag = True
                            break
                        if e_unit.health <= 0:
                            temp_list[index] = float('inf')
                            continue
                        else:
                            break
                if full_flag:
                    break
                e_id_list.append(index)
                temp_list[index] = float('inf')

            for i,e_id in enumerate(e_id_list):

                e_unit = self.enemies[e_id]
                dist = enemy_dist[e_id]
                e_x = e_unit.pos.x
                e_y = e_unit.pos.y
                if (
                    dist < sight_range and e_unit.health > 0
                ):  # visible and alive
                    # Sight range > shoot range
                    enemy_feats[i, 0] = avail_actions[
                        self.n_actions_no_attack + i
                    ]  # available
                    enemy_feats[i, 1] = dist / sight_range  # distance
                    enemy_feats[i, 2] = (
                        e_x - x
                    ) / sight_range
                    enemy_feats[i, 3] = (
                        e_y - y
                    ) / sight_range

                    ind = 4
                    if self.obs_all_health:
                        enemy_feats[i, ind] = (
                            e_unit.health / e_unit.health_max
                        )
                        ind += 1
                        if self.shield_bits_enemy > 0:
                            max_shield = self.unit_max_shield(e_unit)
                            enemy_feats[i, ind] = (
                                e_unit.shield / max_shield
                            )
                            ind += 1

                    if self.unit_type_bits > 0:
                        type_id = self.get_unit_type_id(e_unit, False)
                        enemy_feats[i, ind + type_id] = 1

            for al_id, al_unit in self.agents.items():
                al_x = al_unit.pos.x
                al_y = al_unit.pos.y
                if al_id == agent_id:
                    ally_dist[al_id] = float('inf')
                    continue
                dist = self.distance(x, y, al_x, al_y)
                ally_dist[al_id] = dist

            temp_list2 = deepcopy(ally_dist)
            full_flag_ally = False
            for i in range(cut_down_n_allies):
                index = temp_list2.argmin()
                al_unit = self.get_unit_by_id(index)
                if al_unit.health <= 0:
                    temp_list2[index] = float('inf')
                    while True:
                        index = temp_list2.argmin()
                        al_unit = self.get_unit_by_id(index)
                        if temp_list2[index] == float('inf'):
                            full_flag_ally = True
                            break
                        if al_unit.health <= 0:
                            temp_list2[index] = float('inf')
                            continue
                        else:
                            break
                if full_flag_ally:
                    break

                al_id_list.append(index)
                temp_list2[index] = float('inf')

            for i, al_id in enumerate(al_id_list):

                al_unit = self.get_unit_by_id(al_id)
                al_x = al_unit.pos.x
                al_y = al_unit.pos.y
                dist = ally_dist[al_id]

                if (
                    dist < sight_range and al_unit.health > 0
                ):  # visible and alive
                    ally_feats[i, 0] = 1  # visible
                    ally_feats[i, 1] = dist / sight_range  # distance
                    ally_feats[i, 2] = (al_x - x) / sight_range  # relative X
                    ally_feats[i, 3] = (al_y - y) / sight_range  # relative Y

                    ind = 4
                    if self.obs_all_health:
                        ally_feats[i, ind] = (
                            al_unit.health / al_unit.health_max
                        )  # health
                        ind += 1
                        if self.shield_bits_ally > 0:
                            max_shield = self.unit_max_shield(al_unit)
                            ally_feats[i, ind] = (
                                al_unit.shield / max_shield
                            )  # shield
                            ind += 1


                    if self.unit_type_bits > 0:
                        type_id = self.get_unit_type_id(al_unit, True)
                        ally_feats[i, ind + type_id] = 1
                        ind += self.unit_type_bits

                    if self.obs_last_action:
                        ally_feats[i, ind:] = self.last_action[al_id]

            # Own features
            ind = 0
            if self.obs_own_health:
                own_feats[ind] = unit.health / unit.health_max
                ind += 1
                if self.shield_bits_ally > 0:
                    max_shield = self.unit_max_shield(unit)
                    own_feats[ind] = unit.shield / max_shield
                    ind += 1
            if self.unit_type_bits > 0:
                type_id = self.get_unit_type_id(unit, True)
                own_feats[ind + type_id] = 1

        agent_obs = np.concatenate(
            (
                move_feats.flatten(),
                enemy_feats.flatten(),
                ally_feats.flatten(),
                own_feats.flatten(),
            )
        )

        if self.obs_timestep_number:
            agent_obs = np.append(
                agent_obs, self._episode_steps / self.episode_limit
            )

        if self.debug:
            logging.debug("Obs Agent: {}".format(agent_id).center(60, "-"))
            logging.debug(
                "Avail. actions {}".format(
                    self.get_avail_agent_actions(agent_id)
                )
            )
            logging.debug("Move feats {}".format(move_feats))
            logging.debug("Enemy feats {}".format(enemy_feats))
            logging.debug("Ally feats {}".format(ally_feats))
            logging.debug("Own feats {}".format(own_feats))

        return agent_obs

    def get_cut_down_obs(self):

        agents_obs = [self.get_cut_down_five_agent(i) for i in range(self.n_agents)]
        return agents_obs




    def get_obs_comprehensive(self):
        survival_num_allys = self.n_agents - np.sum(self.death_tracker_ally)

        if survival_num_allys < 6:
            agents_obs = [self.get_obs_agent(i) for i in range(self.n_agents)]
        else:
            agents_obs = self.change_get_agents_obs()
            agents_obs = [np.squeeze(agents_obs[i, :]) for i in range(self.n_agents)]

        return agents_obs


    def get_time_difference_array(self):

        return self.matrix_array.reshape(1, -1),self.improve_matrix_array.reshape(1,-1),self.original_array.reshape(1, -1)\
            ,self.comprehensive_array.reshape(1, -1),self.map_name


    def get_times(self):
        return self.times_array.reshape(1, -1),self.comprehensive_times_array.reshape(1, -1)

    def get_survival_agents(self):
        return self.survival_allys.reshape(1, -1),self.survival_enemies.reshape(1, -1)


    def get_state(self):
        """Returns the global state.
        NOTE: This functon should not be used during decentralised execution.
        """
        if self.obs_instead_of_state:
            obs_concat = np.concatenate(self.get_obs_original(), axis=0).astype(
                np.float32
            )
            return obs_concat

        state_dict = self.get_state_dict()

        state = np.append(
            state_dict["allies"].flatten(), state_dict["enemies"].flatten()
        )
        if "last_action" in state_dict:
            state = np.append(state, state_dict["last_action"].flatten())
        if "timestep" in state_dict:
            state = np.append(state, state_dict["timestep"])

        state = state.astype(dtype=np.float32)

        if self.debug:
            logging.debug("STATE".center(60, "-"))
            logging.debug("Ally state {}".format(state_dict["allies"]))
            logging.debug("Enemy state {}".format(state_dict["enemies"]))
            if self.state_last_action:
                logging.debug("Last actions {}".format(self.last_action))

        return state

    def get_ally_num_attributes(self):
        return len(self.ally_state_attr_names)

    def get_enemy_num_attributes(self):
        return len(self.enemy_state_attr_names)

    def get_state_dict(self):
        """Returns the global state as a dictionary.

        - allies: numpy array containing agents and their attributes
        - enemies: numpy array containing enemies and their attributes
        - last_action: numpy array of previous actions for each agent
        - timestep: current no. of steps divided by total no. of steps

        NOTE: This function should not be used during decentralised execution.
        """

        # number of features equals the number of attribute names
        nf_al = self.get_ally_num_attributes()
        nf_en = self.get_enemy_num_attributes()

        ally_state = np.zeros((self.n_agents, nf_al))
        enemy_state = np.zeros((self.n_enemies, nf_en))

        center_x = self.map_x / 2
        center_y = self.map_y / 2

        for al_id, al_unit in self.agents.items():
            if al_unit.health > 0:
                x = al_unit.pos.x
                y = al_unit.pos.y
                max_cd = self.unit_max_cooldown(al_unit)

                ally_state[al_id, 0] = (
                    al_unit.health / al_unit.health_max
                )  # health
                if (
                    self.map_type == "MMM"
                    and al_unit.unit_type == self.medivac_id
                ):
                    ally_state[al_id, 1] = al_unit.energy / max_cd  # energy
                else:
                    ally_state[al_id, 1] = (
                        al_unit.weapon_cooldown / max_cd
                    )  # cooldown
                ally_state[al_id, 2] = (
                    x - center_x
                ) / self.max_distance_x  # relative X
                ally_state[al_id, 3] = (
                    y - center_y
                ) / self.max_distance_y  # relative Y

                if self.shield_bits_ally > 0:
                    max_shield = self.unit_max_shield(al_unit)
                    ally_state[al_id, 4] = (
                        al_unit.shield / max_shield
                    )  # shield

                if self.unit_type_bits > 0:
                    type_id = self.get_unit_type_id(al_unit, True)
                    ally_state[al_id, type_id - self.unit_type_bits] = 1

        for e_id, e_unit in self.enemies.items():
            if e_unit.health > 0:
                x = e_unit.pos.x
                y = e_unit.pos.y

                enemy_state[e_id, 0] = (
                    e_unit.health / e_unit.health_max
                )  # health
                enemy_state[e_id, 1] = (
                    x - center_x
                ) / self.max_distance_x  # relative X
                enemy_state[e_id, 2] = (
                    y - center_y
                ) / self.max_distance_y  # relative Y

                if self.shield_bits_enemy > 0:
                    max_shield = self.unit_max_shield(e_unit)
                    enemy_state[e_id, 3] = e_unit.shield / max_shield  # shield

                if self.unit_type_bits > 0:
                    type_id = self.get_unit_type_id(e_unit, False)
                    enemy_state[e_id, type_id - self.unit_type_bits] = 1

        state = {"allies": ally_state, "enemies": enemy_state}

        if self.state_last_action:
            state["last_action"] = self.last_action
        if self.state_timestep_number:
            state["timestep"] = self._episode_steps / self.episode_limit

        return state

    def get_obs_enemy_feats_size(self):
        """Returns the dimensions of the matrix containing enemy features.
        Size is n_enemies x n_features.
        """
        nf_en = 4 + self.unit_type_bits

        if self.obs_all_health:
            nf_en += 1 + self.shield_bits_enemy

        return self.n_enemies, nf_en

    def get_obs_ally_feats_size(self):
        """Returns the dimensions of the matrix containing ally features.
        Size is n_allies x n_features.
        """
        nf_al = 4 + self.unit_type_bits

        if self.obs_all_health:
            nf_al += 1 + self.shield_bits_ally

        if self.obs_last_action:
            nf_al += self.n_actions

        return self.n_agents - 1, nf_al

    def get_obs_own_feats_size(self):
        """
        Returns the size of the vector containing the agents' own features.
        """
        own_feats = self.unit_type_bits
        if self.obs_own_health:
            own_feats += 1 + self.shield_bits_ally
        if self.obs_timestep_number:
            own_feats += 1

        return own_feats

    def get_obs_move_feats_size(self):
        """Returns the size of the vector containing the agents's movement-
        related features.
        """
        move_feats = self.n_actions_move
        if self.obs_pathing_grid:
            move_feats += self.n_obs_pathing
        if self.obs_terrain_height:
            move_feats += self.n_obs_height

        return move_feats

    def get_obs_size(self):
        """Returns the size of the observation."""
        own_feats = self.get_obs_own_feats_size()
        move_feats = self.get_obs_move_feats_size()

        n_enemies, n_enemy_feats = self.get_obs_enemy_feats_size()
        n_allies, n_ally_feats = self.get_obs_ally_feats_size()

        enemy_feats = 5 * n_enemy_feats
        ally_feats = 5 * n_ally_feats

        return move_feats + enemy_feats + ally_feats + own_feats

    def get_state_size(self):
        """Returns the size of the global state."""
        if self.obs_instead_of_state:
            return self.get_obs_size() * self.n_agents

        nf_al = 4 + self.shield_bits_ally + self.unit_type_bits
        nf_en = 3 + self.shield_bits_enemy + self.unit_type_bits

        enemy_state = self.n_enemies * nf_en
        ally_state = self.n_agents * nf_al

        size = enemy_state + ally_state

        if self.state_last_action:
            size += self.n_agents * self.n_actions
        if self.state_timestep_number:
            size += 1

        return size

    def get_visibility_matrix(self):
        """Returns a boolean numpy array of dimensions
        (n_agents, n_agents + n_enemies) indicating which units
        are visible to each agent.
        """
        arr = np.zeros(
            (self.n_agents, self.n_agents + self.n_enemies),
            dtype=np.bool,
        )

        for agent_id in range(self.n_agents):
            current_agent = self.get_unit_by_id(agent_id)
            if current_agent.health > 0:  # it agent not dead
                x = current_agent.pos.x
                y = current_agent.pos.y
                sight_range = self.unit_sight_range(agent_id)

                # Enemies
                for e_id, e_unit in self.enemies.items():
                    e_x = e_unit.pos.x
                    e_y = e_unit.pos.y
                    dist = self.distance(x, y, e_x, e_y)

                    if dist < sight_range and e_unit.health > 0:
                        # visible and alive
                        arr[agent_id, self.n_agents + e_id] = 1

                # The matrix for allies is filled symmetrically
                al_ids = [
                    al_id for al_id in range(self.n_agents) if al_id > agent_id
                ]
                for _, al_id in enumerate(al_ids):
                    al_unit = self.get_unit_by_id(al_id)
                    al_x = al_unit.pos.x
                    al_y = al_unit.pos.y
                    dist = self.distance(x, y, al_x, al_y)

                    if dist < sight_range and al_unit.health > 0:
                        # visible and alive
                        arr[agent_id, al_id] = arr[al_id, agent_id] = 1

        return arr

    def get_unit_type_id(self, unit, ally):
        """Returns the ID of unit type in the given scenario."""
        if ally:  # use new SC2 unit types
            type_id = unit.unit_type - self._min_unit_type
        else:  # use default SC2 unit types
            if self.map_type == "stalkers_and_zealots":
                # id(Stalker) = 74, id(Zealot) = 73
                type_id = unit.unit_type - 73
            elif self.map_type == "colossi_stalkers_zealots":
                # id(Stalker) = 74, id(Zealot) = 73, id(Colossus) = 4
                if unit.unit_type == 4:
                    type_id = 0
                elif unit.unit_type == 74:
                    type_id = 1
                else:
                    type_id = 2
            elif self.map_type == "bane":
                if unit.unit_type == 9:
                    type_id = 0
                else:
                    type_id = 1
            elif self.map_type == "MMM":
                if unit.unit_type == 51:
                    type_id = 0
                elif unit.unit_type == 48:
                    type_id = 1
                else:
                    type_id = 2

        return type_id

    def get_avail_agent_actions(self, agent_id):
        """Returns the available actions for agent_id."""
        unit = self.get_unit_by_id(agent_id)
        if unit.health > 0:
            avail_actions = [0] * self.n_actions

            # stop should be allowed
            avail_actions[1] = 1

            # see if we can move
            if self.can_move(unit, Direction.NORTH):
                avail_actions[2] = 1
            if self.can_move(unit, Direction.SOUTH):
                avail_actions[3] = 1
            if self.can_move(unit, Direction.EAST):
                avail_actions[4] = 1
            if self.can_move(unit, Direction.WEST):
                avail_actions[5] = 1

            # Can attack only alive units that are alive in the shooting range
            shoot_range = self.unit_shoot_range(agent_id)

            target_items = self.enemies.items()
            if self.map_type == "MMM" and unit.unit_type == self.medivac_id:
                # Medivacs cannot heal themselves or other flying units
                target_items = [
                    (t_id, t_unit)
                    for (t_id, t_unit) in self.agents.items()
                    if t_unit.unit_type != self.medivac_id
                ]

            x = unit.pos.x
            y = unit.pos.y
            target_nearest_num = 5
            target_num = self.n_enemies

            if self.map_type == "MMM" and unit.unit_type == self.medivac_id:
                target_num = self.n_agents

            t_id_list = []
            t_dist = np.full(target_num, np.inf)
            for t_id, t_unit in target_items:
                t_x = t_unit.pos.x
                t_y = t_unit.pos.y
                dist = self.distance(x, y, t_x, t_y)
                t_dist[t_id] = dist
            temp_list = deepcopy(t_dist)

            full_flag = False
            for i in range(target_nearest_num):
                index = temp_list.argmin()
                t_unit = self.enemies[index]
                if self.map_type == "MMM" and unit.unit_type == self.medivac_id:
                    t_unit = self.get_unit_by_id(index)
                if t_unit.health <= 0:
                    temp_list[index] = float('inf')
                    while True:
                        index = temp_list.argmin()
                        t_unit = self.enemies[index]
                        if self.map_type == "MMM" and unit.unit_type == self.medivac_id:
                            t_unit = self.get_unit_by_id(index)
                        if temp_list[index] == float('inf'):
                            full_flag = True
                            break
                        if t_unit.health <= 0:
                            temp_list[index] = float('inf')
                            continue
                        else:
                            break
                if full_flag:
                    break
                t_id_list.append(index)
                temp_list[index] = float('inf')


            for i, t_id in enumerate(t_id_list):
                t_unit = self.enemies[t_id]
                if self.map_type == "MMM" and unit.unit_type == self.medivac_id:
                    t_unit = self.get_unit_by_id(t_id)
                if t_unit.health > 0:
                    dist = t_dist[t_id]
                    if dist <= shoot_range:
                        avail_actions[i + self.n_actions_no_attack] = 1

            return avail_actions

        else:

            return [1] + [0] * (self.n_actions - 1)

    def get_avail_actions(self):
        """Returns the available actions of all agents in a list."""
        avail_actions = []
        for agent_id in range(self.n_agents):
            avail_agent = self.get_avail_agent_actions(agent_id)
            avail_actions.append(avail_agent)
        return avail_actions

    def close(self):
        """Close StarCraft II."""
        if self.renderer is not None:
            self.renderer.close()
            self.renderer = None
        if self._sc2_proc:
            self._sc2_proc.close()

    def seed(self):
        """Returns the random seed used by the environment."""
        return self._seed

    def render(self, mode="human"):
        if self.renderer is None:
            from smac.env.starcraft2.render import StarCraft2Renderer

            self.renderer = StarCraft2Renderer(self, mode)
        assert (
            mode == self.renderer.mode
        ), "mode must be consistent across render calls"
        return self.renderer.render(mode)

    def _kill_all_units(self):
        """Kill all units on the map."""
        units_alive = [
            unit.tag for unit in self.agents.values() if unit.health > 0
        ] + [unit.tag for unit in self.enemies.values() if unit.health > 0]
        debug_command = [
            d_pb.DebugCommand(kill_unit=d_pb.DebugKillUnit(tag=units_alive))
        ]
        self._controller.debug(debug_command)

    def init_units(self):
        """Initialise the units."""
        while True:
            # Sometimes not all units have yet been created by SC2
            self.agents = {}
            self.enemies = {}

            ally_units = [
                unit
                for unit in self._obs.observation.raw_data.units
                if unit.owner == 1
            ]
            ally_units_sorted = sorted(
                ally_units,
                key=attrgetter("unit_type", "pos.x", "pos.y"),
                reverse=False,
            )

            for i in range(len(ally_units_sorted)):
                self.agents[i] = ally_units_sorted[i]
                if self.debug:
                    logging.debug(
                        "Unit {} is {}, x = {}, y = {}".format(
                            len(self.agents),
                            self.agents[i].unit_type,
                            self.agents[i].pos.x,
                            self.agents[i].pos.y,
                        )
                    )

            for unit in self._obs.observation.raw_data.units:
                if unit.owner == 2:
                    self.enemies[len(self.enemies)] = unit
                    if self._episode_count == 0:
                        self.max_reward += unit.health_max + unit.shield_max

            if self._episode_count == 0:
                min_unit_type = min(
                    unit.unit_type for unit in self.agents.values()
                )
                self._init_ally_unit_types(min_unit_type)

            all_agents_created = len(self.agents) == self.n_agents
            all_enemies_created = len(self.enemies) == self.n_enemies

            self._unit_types = [
                unit.unit_type for unit in ally_units_sorted
            ] + [
                unit.unit_type
                for unit in self._obs.observation.raw_data.units
                if unit.owner == 2
            ]

            if all_agents_created and all_enemies_created:  # all good
                return

            try:
                self._controller.step(1)
                self._obs = self._controller.observe()
            except (protocol.ProtocolError, protocol.ConnectionError):
                self.full_restart()
                self.reset()

    def get_unit_types(self):
        if self._unit_types is None:
            warn(
                "unit types have not been initialized yet, please call"
                "env.reset() to populate this and call t1286he method again."
            )

        return self._unit_types

    def update_units(self):
        """Update units after an environment step.
        This function assumes that self._obs is up-to-date.
        """
        n_ally_alive = 0
        n_enemy_alive = 0

        # Store previous state
        self.previous_ally_units = deepcopy(self.agents)
        self.previous_enemy_units = deepcopy(self.enemies)

        for al_id, al_unit in self.agents.items():
            updated = False
            for unit in self._obs.observation.raw_data.units:
                if al_unit.tag == unit.tag:
                    self.agents[al_id] = unit
                    updated = True
                    n_ally_alive += 1
                    break

            if not updated:  # dead
                al_unit.health = 0

        for e_id, e_unit in self.enemies.items():
            updated = False
            for unit in self._obs.observation.raw_data.units:
                if e_unit.tag == unit.tag:
                    self.enemies[e_id] = unit
                    updated = True
                    n_enemy_alive += 1
                    break

            if not updated:  # dead
                e_unit.health = 0

        if (
            n_ally_alive == 0
            and n_enemy_alive > 0
            or self.only_medivac_left(ally=True)
        ):
            return -1  # lost
        if (
            n_ally_alive > 0
            and n_enemy_alive == 0
            or self.only_medivac_left(ally=False)
        ):
            return 1  # won
        if n_ally_alive == 0 and n_enemy_alive == 0:
            return 0

        return None

    def _init_ally_unit_types(self, min_unit_type):
        """Initialise ally unit types. Should be called once from the
        init_units function.
        """
        self._min_unit_type = min_unit_type
        if self.map_type == "marines":
            self.marine_id = min_unit_type
        elif self.map_type == "stalkers_and_zealots":
            self.stalker_id = min_unit_type
            self.zealot_id = min_unit_type + 1
        elif self.map_type == "colossi_stalkers_zealots":
            self.colossus_id = min_unit_type
            self.stalker_id = min_unit_type + 1
            self.zealot_id = min_unit_type + 2
        elif self.map_type == "MMM":
            self.marauder_id = min_unit_type
            self.marine_id = min_unit_type + 1
            self.medivac_id = min_unit_type + 2
        elif self.map_type == "zealots":
            self.zealot_id = min_unit_type
        elif self.map_type == "hydralisks":
            self.hydralisk_id = min_unit_type
        elif self.map_type == "stalkers":
            self.stalker_id = min_unit_type
        elif self.map_type == "colossus":
            self.colossus_id = min_unit_type
        elif self.map_type == "bane":
            self.baneling_id = min_unit_type
            self.zergling_id = min_unit_type + 1

    def only_medivac_left(self, ally):
        """Check if only Medivac units are left."""
        if self.map_type != "MMM":
            return False

        if ally:
            units_alive = [
                a
                for a in self.agents.values()
                if (a.health > 0 and a.unit_type != self.medivac_id)
            ]
            if len(units_alive) == 0:
                return True
            return False
        else:
            units_alive = [
                a
                for a in self.enemies.values()
                if (a.health > 0 and a.unit_type != self.medivac_id)
            ]
            if len(units_alive) == 1 and units_alive[0].unit_type == 54:
                return True
            return False

    def get_unit_by_id(self, a_id):
        """Get unit by ID."""
        return self.agents[a_id]

    def get_stats(self):
        stats = {
            "battles_won": self.battles_won,
            "battles_game": self.battles_game,
            "battles_draw": self.timeouts,
            "win_rate": self.battles_won / self.battles_game,
            "timeouts": self.timeouts,
            "restarts": self.force_restarts,
        }
        return stats

    def get_env_info(self):
        env_info = super().get_env_info()
        env_info["agent_features"] = self.ally_state_attr_names
        env_info["enemy_features"] = self.enemy_state_attr_names
        return env_info
