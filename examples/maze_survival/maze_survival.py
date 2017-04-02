import numpy as np
import gym
from gym import error, spaces, utils
from gym.utils import seeding
from six import StringIO
import sys

MAP = np.asarray([
    "+---------+",  # 0
    "| : : : : |",
    "| : : : : |",
    "| : : : : |",
    "| : : : : |",
    "| : : : : |",
    "+---------+",  # 6
    # 01234567890
], dtype="c")


class MazeSurvivalEnv(gym.Env):
    metadata = {'render.modes': ['human', 'ansi']}

    def __init__(self, max_num_keys=12, seed=1):
        self.max_num_keys = max_num_keys
        self.action_space = spaces.Discrete(5)

        # self.observation_space = spaces.Box(
        #     low=np.array([0,-2,-2,
        #                   0,0,
        #                   -2,-2,
        #                   0,0,0,0]),
        #     high=np.array([self.max_num_keys,2,2,
        #                    self.max_num_keys,1,
        #                    3,3,
        #                    1,1,1,1]))
        self.observation_space = spaces.Discrete(5 * 5 * max_num_keys * max_num_keys * 2)

        self.room_dict = {0: (None, 2, 1, None),
                          1: (3, None, None, 0),
                          2: (0, None, None, 4)
                          }
        self.action_meanings = ("north", "south", "east", "west", "interact")
        self.pos_to_door = {0: {2: 0, -2: 1},
                            2: {0: 2},
                            -2: {0: 3}}
        self.door_to_pos = {0: (0, 2),
                            1: (0, -2),
                            2: (2, 0),
                            3: (-2, 0)}
        self.act_to_v = {0: (0, 1),
                         1: (0, -1),
                         2: (1, 0),
                         3: (-1, 0)}
        self.rand_seed = seed
        self._seed(seed)
        self._reset()

    def _seed(self, seed=None):
        self.np_random, seed = seeding.np_random(seed)
        return [seed]

    @property
    def key_loc(self):
        kx = 7
        ky = 7
        if self.num_keys == self.room_id:
            kx = self.key_locations[self.room_id][0]
            ky = self.key_locations[self.room_id][1]
        return kx, ky

    @property
    def obs(self):
        has_doors = tuple((0 if d is None else 1 for d in self.connected_rooms(self.room_id)))
        return (self.room_id, self.x, self.y,
                self.num_keys, self.door_open) + self.key_loc + has_doors

    def get_encoded_state(self):
        s = self.num_keys
        s = s * 2 + self.door_open
        s = s * (self.max_num_keys + 1) + self.room_id
        s = s * 6 + self.x + 3
        s = s * 6 + self.y + 3
        return s

    def state_as_string(self):
        has_doors = tuple((0 if d is None else 1 for d in self.connected_rooms(self.room_id)))
        states = (self.room_id, self.x, self.y,
                  self.num_keys, self.door_open) + self.key_loc + has_doors
        state_str = ''.join((chr(s + 128) for s in states))
        return state_str

    def _reset(self):
        self._seed(self.rand_seed)
        self.room_id = 0
        self.num_keys = 0
        self.door_open = 0
        self.x = np.random.random_integers(-2, 2)
        self.y = np.random.random_integers(-2, 2)
        self.key_locations = self.np_random.random_integers(-2, 2, size=(self.max_num_keys + 1, 2))

        return self.obs

    def connected_rooms(self, current_id):
        try:
            rooms = self.room_dict[current_id]
        except KeyError:
            if (current_id) % 4 == 3:
                rooms = (None, current_id - 2, current_id + 2, None)
            elif (current_id) % 4 == 1:
                rooms = (current_id + 2, None, None, current_id - 2)
            elif (current_id) % 4 == 0:
                rooms = (None, current_id + 2, current_id - 2, None)
            elif (current_id) % 4 == 2:
                rooms = (current_id - 2, None, None, current_id + 2)
            self.room_dict[current_id] = rooms
        return rooms

    # door_rewards = [1000,2000,3000,4000,5000,6000,7000,8000,9000,10000,11000,12000,13000]
    door_rewards = [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 28000]

    def _step(self, a):
        r = -1
        done = False
        if a == 4:
            if self.num_keys == self.room_id and \
                            self.x == self.key_locations[self.room_id][0] and \
                            self.y == self.key_locations[self.room_id][1]:
                self.num_keys += 1
                self.door_open = 0
                # if self.num_keys == 1:
                #     r = 100
            else:
                try:
                    door_dict = self.pos_to_door[self.x]
                    door = door_dict[self.y]
                    adj_room = self.connected_rooms(self.room_id)[door]
                except KeyError:
                    door = None
                    adj_room = None

                if adj_room is not None and \
                                self.num_keys >= adj_room:
                    self.room_id = adj_room
                    self.x = -self.x
                    self.y = -self.y
                    if self.door_open == 0 and self.num_keys == adj_room:
                        self.door_open = 1
                        r = self.door_rewards[adj_room]
                        if self.room_id == self.max_num_keys:
                            done = True
        else:
            diff = self.act_to_v[a]
            target_x = self.x + +diff[0]
            target_y = self.y + diff[1]
            if abs(target_x) > 2 or abs(target_y) > 2:
                r = -10000
                done = True
            else:
                self.x = target_x
                self.y = target_y
        return self.obs, r, done, {}

    def _render(self, mode='human', close=False):
        if close:
            return
        outfile = StringIO() if mode == 'ansi' else sys.stdout
        out = [[c.decode('utf-8') for c in line] for line in MAP.copy().tolist()]
        out[-self.y + 3][self.x * 2 + 5] = 'A'
        if self.num_keys == self.room_id:
            kx, ky = self.key_loc
            out[-ky + 3][kx * 2 + 5] = 'K' if out[-ky + 3][kx * 2 + 5] != 'A' else 'R'
        adj_rooms = self.connected_rooms(self.room_id)
        for door_id in range(4):
            if adj_rooms[door_id] is not None:
                door_x, door_y = self.door_to_pos[door_id]
                door_x = int(door_x * 2 * 5 / 4 + 5)
                door_y = int(-door_y * 3 / 2 + 3)
                out[door_y][door_x] = "@"
        outfile.write("room:\t{}\tkey:\t{}\n".format(self.room_id, self.num_keys))
        outfile.write("\n".join(["".join(row) for row in out]) + "\n")
        if mode != 'human':
            return outfile
