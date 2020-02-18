import numpy as np

from Experiment_Engine.util import check_attribute, check_dict_else_default


class Acrobot:
    """
    Environment Specifications:
    Number of Actions = 3
    Observation Dimension = 4 (angle1, angle2, vel1, vel2)
    Observation Dtype = np.float64
    Reward = -1 at every step

    Summary Name: steps_per_episode
    """

    def __init__(self, config, summary=None):
        """ Parameters:
        Name:                       Type            Default:        Description(omitted when self-explanatory):
        max_actions                 int             1000            The max number of actions executed before forcing
                                                                    a time out
        norm_state                  bool            True            Normalize the state to [-1,1]
        store_summary               bool            False           Whether to store the summary of the environment
        """
        self.norm_state = check_attribute(config, 'norm_state', True)
        self.max_actions = check_attribute(config, 'max_actions', 1000)
        self.store_summary = check_attribute(config, 'store_summary', False)
        self.summary = summary
        if self.store_summary:
            assert isinstance(self.summary, dict)
            check_dict_else_default(self.summary, "steps_per_episode", [])

        self.num_actions = 3
        self.state_dims = 4

        " Inner state of the environment "
        self.step_count = 0
        self.current_state = np.float64(np.random.uniform(low=-0.5, high=0.5, size=(4,)))
        self.MAX_VEL_1 = 4 * np.pi
        self.MAX_VEL_2 = 9 * np.pi
        self.MAX_THETA_1 = np.pi
        self.MAX_THETA_2 = np.pi
        self.m1 = 1.0
        self.m2 = 1.0
        self.l1 = 1.0
        self.l2 = 1.0
        self.lc1 = 0.5
        self.lc2 = 0.5
        self.I1 = 1.0
        self.I2 = 1.0
        self.g = 9.8
        self.dt = 0.05
        self.acrobotGoalPosition = 1.0

    def reset(self):
        if self.store_summary:
            self.summary["steps_per_episode"].append(self.step_count)
        self.step_count = 0
        self.current_state = np.float64(np.random.uniform(low=-0.5, high=0.5, size=(4,)))
        return self.get_current_state()

    def get_current_state(self):
        '''
        normalize to [-1, 1]
        '''
        if self.norm_state:
            s = self.current_state
            s0 = s[0] / (1 * np.pi)
            s1 = s[1] / (1 * np.pi)
            s2 = s[2] / (4 * np.pi)
            s3 = s[3] / (9 * np.pi)
            return np.array([s0, s1, s2, s3], dtype=np.float64)
        else:
            s = self.current_state
            return np.array([s[0], s[1], s[2], s[3]], dtype=np.float64)

    def _terminal(self):
        s = self.current_state
        firstJointEndHeight = self.l1 * np.cos(s[0])
        secondJointEndHeight = self.l2 * np.sin(np.pi / 2 - s[1] - s[2])
        feet_height = -(firstJointEndHeight + secondJointEndHeight)
        return bool(feet_height > self.acrobotGoalPosition)

    def step(self, a):
        self.step_count += 1
        s = self.current_state

        torque = a - 1.0
        count = 0
        theta1 = s[0]
        theta2 = s[1]
        theta1Dot = s[2]
        theta2Dot = s[3]

        while count < 4:
            d1 = self.m1 * np.power(self.lc1, 2) + \
                 self.m2 * (np.power(self.l1, 2) + np.power(self.lc2, 2) + 2 * self.l1 * self.lc2 * np.cos(theta2)) \
                 + self.I1 + self.I2
            d2 = self.m2 * (np.power(self.lc2, 2) + self.l1 * self.lc2 * np.cos(theta2)) + self.I2
            phi_2 = self.m2 * self.lc2 * self.g * np.cos(theta1 + theta2 - np.pi / 2.0)
            phi_1 = -(self.m2 * self.l1 * self.lc2 * np.power(theta2Dot, 2) * np.sin(theta2)
                      - 2 * self.m2 * self.l1 * self.lc2 * theta1Dot * theta2Dot * np.sin(theta2)) \
                    + (self.m1 * self.lc1 + self.m2 * self.l1) * self.g * np.cos(theta1 - np.pi / 2.0) + phi_2
            theta2_ddot = (torque + (d2 / d1) * phi_1
                           - self.m2 * self.l1 * self.lc2 * np.power(theta1Dot, 2) * np.sin(theta2)
                           - phi_2) / (self.m2 * np.power(self.lc2, 2) + self.I2 - np.power(d2, 2) / d1)
            theta1_ddot = -(d2 * theta2_ddot + phi_1) / d1
            theta1Dot += theta1_ddot * self.dt
            theta2Dot += theta2_ddot * self.dt
            theta1 += theta1Dot * self.dt
            theta2 += theta2Dot * self.dt
            count += 1

        if np.fabs(theta1Dot)>self.MAX_VEL_1:
            theta1Dot = np.sign(theta1Dot) * self.MAX_VEL_1
        if np.fabs(theta2Dot)>self.MAX_VEL_2:
            theta2Dot = np.sign(theta2Dot) * self.MAX_VEL_2
        if np.fabs(theta1) > self.MAX_THETA_1:
            theta1 = np.sign(theta1) * self.MAX_THETA_1
            theta1Dot = 0
        if np.fabs(theta2) > self.MAX_THETA_2:
            theta2 = np.sign(theta2) * self.MAX_THETA_2
            theta2Dot = 0

        s[0] = theta1
        s[1] = theta2
        s[2] = theta1Dot
        s[3] = theta2Dot
        self.current_state = s

        timeout = bool(self.step_count >= self.max_actions)
        terminal = self._terminal()
        reward = -1
        return self.get_current_state(), reward, terminal, timeout
