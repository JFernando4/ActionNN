import numpy as np

from Experiment_Engine.util import check_dict_else_default, check_attribute


class Puddle:

    def __init__(self, headX, headY, tailX, tailY, radius, length):
        self.headX = headX
        self.headY = headY
        self.tailX = tailX
        self.tailY = tailY
        self.radius = radius
        self.length = length

    def getDistance(self, xCoor, yCoor):
        u = (xCoor - self.tailX) * (self.headX - self.tailX) + (yCoor - self.tailY) * (self.headY - self.tailY) / self.length;

        dist = 0.0

        if u < 0.0 or u > 1.0:
            if u < 0.0:
                dist = np.sqrt(np.power((self.tailX - xCoor), 2) + np.power((self.tailY - yCoor), 2))
            else:
                dist = np.sqrt(np.power((self.headX - xCoor), 2) + np.power((self.headY - yCoor), 2))
        else:
            x = self.tailX + u * (self.headX - self.tailX)
            y = self.tailY + u * (self.headY - self.tailY)

            dist = np.sqrt(np.power((x - xCoor), 2) + np.power((y - yCoor), 2))

        if dist < self.radius:
            return self.radius - dist
        else:
            return 0


class PuddleWorld:
    """
    Environment Specifications:
    Number of Actions = 4 (up = 0, down = 1, right = 2, left = 3)
    Observation Dimension = 2 (x-coordinate, y-coordinate)
    Observation Dtype = np.float64
    Reward = -1 at every step + -400 * (distance to the nearest edge of each puddle)
        - This means that the farther into the puddle, the larger the penalty.

    Summary Name: steps_per_episode, reward_per_step
    """

    def __init__(self, config, summary=None):
        """ Parameters:
        Name:                       Type            Default:        Description(omitted when self-explanatory):
        max_episode_length          int             200000          The max number of steps executed in an episoe
                                                                    before forcing a time out
        norm_state                  bool            True            Normalize the state to [-1,1]
        store_summary               bool            False           Whether to store the summary of the environment
        number_of_steps             int             200000          Total number of environment steps
        """
        check_attribute(config, 'current_step', 0)
        self.config = config

        # environment parameters
        self.max_episode_length = check_attribute(config, 'max_episode_length', 200000)
        self.norm_state = check_attribute(config, 'norm_state', True)

        # summary parameters
        self.store_summary = check_attribute(config, 'store_summary', default_value=False)
        self.summary = summary
        self.number_of_steps = check_attribute(config, 'number_of_steps', 200000)

        if self.store_summary:
            assert isinstance(self.summary, dict)
            self.reward_per_step = np.zeros(self.number_of_steps, dtype=np.float64)
            check_dict_else_default(self.summary, "steps_per_episode", [])
            check_dict_else_default(self.summary, "reward_per_step", self.reward_per_step)

        self.num_action = 4
        self.num_state = 2

        """ Inner state of the environment """
        self.episode_step_count = 0
        self.state = np.float64(np.random.uniform(low=0.0, high=0.1, size=(2,)))
        self.puddle1 = Puddle(0.45, 0.75, 0.10, 0.75, 0.1, 0.35)
        self.puddle2 = Puddle(0.45, 0.80, 0.45, 0.40, 0.1, 0.4)

        self.pworld_min_x = 0.0
        self.pworld_max_x = 1.0
        self.pworld_min_y = 0.0
        self.pworld_max_y = 1.0

        self.goalDimension = 0.05
        self.defDisplacement = 0.05

        self.goalXCoor = self.pworld_max_x - self.goalDimension
        self.goalYCoor = self.pworld_max_y - self.goalDimension

    def reset(self):
        if self.store_summary:
            self.summary["steps_per_episode"].append(self.episode_step_count)
        self.episode_step_count = 0
        self.state = np.float64(np.random.uniform(low=0.0, high=0.1, size=(2,)))
        return self.get_current_state()

    def get_current_state(self):
        '''
        normalize to [-1, 1]
        '''
        if self.norm_state:
            s = self.state
            s0 = (s[0] - 0.5) * 2.0
            s1 = (s[1] - 0.5) * 2.0
            return np.array([s0, s1], dtype=np.float64)
        else:
            s = self.state
            return np.array([s[0], s[1]], dtype=np.float64)

    def _terminal(self):
        s = self.state
        return bool((s[0] >= self.goalXCoor) and (s[1] >= self.goalYCoor))

    def _reward(self, x, y, terminal):
        if terminal:
            return -1
        reward = -1
        reward += (-400 * self.puddle1.getDistance(x, y))
        reward += (-400 * self.puddle2.getDistance(x, y))
        reward = reward
        return reward

    def step(self, a):
        self.config.current_step += 1
        self.episode_step_count += 1
        s = self.state

        xpos = s[0]
        ypos = s[1]

        if a == 0:      # up
            ypos += self.defDisplacement
        elif a == 1:    # down
            ypos -= self.defDisplacement
        elif a == 2:    # right
            xpos += self.defDisplacement
        else:           # left
            xpos -= self.defDisplacement

        if xpos > self.pworld_max_x:
            xpos = self.pworld_max_x
        elif xpos < self.pworld_min_x:
            xpos = self.pworld_min_x

        if ypos > self.pworld_max_y:
            ypos = self.pworld_max_y
        elif ypos < self.pworld_min_y:
            ypos = self.pworld_min_y

        s[0] = xpos
        s[1] = ypos
        self.state = np.float64(s)

        timeout = bool(self.episode_step_count >= self.max_episode_length or
                       self.config.current_step >= self.number_of_steps)
        terminal = self._terminal()
        reward = self._reward(xpos, ypos, terminal)
        if self.store_summary:
            self.reward_per_step[self.config.current_step - 1] += reward

        return self.get_current_state(), reward, terminal, timeout
