import numpy as np
from Experiment_Engine.util import check_dict_else_default, check_attribute, Config


class MountainCar:
    """
    Environment Specifications:
    Number of Actions = 3
    Observation Dimension = 2 (position, velocity)
    Observation Dtype = np.float64
    Reward = -1 at every step

    Summary Name: steps_per_episode, reward_per_step

    Note: In this case, the maximum number of steps parameter indicates the maximum over the whole training period, not
          just per episode as in mountain car.
    """

    def __init__(self, config, summary=None):
        assert isinstance(config, Config)
        """ Parameters:
        Name:                       Type            Default:        Description(omitted when self-explanatory):
        # environment parameters
        max_episode_length          int             500000          The max number of actions executed before forcing
                                                                    a time out
        norm_state                  bool            True            Normalize the state to [-1,1]
        # summary parameters
        store_summary               bool            False           Whether to store the summary of the environment
        number_of_steps             int             500000          Total number of environment steps
        """
        check_attribute(config, 'current_step', 0)
        self.config = config

        # environment related variables
        self.max_episode_length = check_attribute(config, 'max_episode_length', default_value=500000)
        self.norm_state = check_attribute(config, 'norm_state', default_value=True)

        # summary related variables
        self.store_summary = check_attribute(config, 'store_summary', default_value=False)
        self.number_of_steps = check_attribute(config, 'number_of_steps', default_value=500000)
        self.summary = summary
        if self.store_summary:
            assert isinstance(self.summary, dict)
            self.reward_per_step = np.zeros(self.number_of_steps, dtype=np.float64)
            check_dict_else_default(self.summary, "steps_per_episode", [])
            check_dict_else_default(self.summary, "reward_per_step", self.reward_per_step)

        # internal state of the environment
        self.episode_step_count = 0
        position = -0.6 + np.random.random() * 0.2
        velocity = 0.0
        self.current_state = np.array((position, velocity), dtype=np.float64)
        self.actions = np.array([0, 1, 2], dtype=int)  # 0 = backward, 1 = coast, 2 = forward
        self.high = np.array([0.5, 0.07], dtype=np.float64)
        self.low = np.array([-1.2, -0.07], dtype=np.float64)
        self.action_dictionary = {0: -1,    # accelerate backwards
                                   1: 0,    # coast
                                   2: 1}    # accelerate forwards

    def reset(self):
        # random() returns a random float in the half open interval [0,1)
        if self.store_summary:
            self.summary["steps_per_episode"].append(self.episode_step_count)
        self.episode_step_count = 0
        position = -0.6 + np.random.random() * 0.2
        velocity = 0.0
        self.current_state = np.array((position, velocity), dtype=np.float64)
        if self.norm_state:
            return self.normalize(self.current_state)
        else:
            return self.current_state

    " Update environment "
    def step(self, A):
        self.episode_step_count += 1
        self.config.current_step += 1

        if A not in self.actions:
            raise ValueError("The action should be one of the following integers: {0, 1, 2}.")
        action = self.action_dictionary[A]
        terminate = False
        timeout = bool(self.episode_step_count >= self.max_episode_length or self.config.current_step >= self.number_of_steps)

        current_position = self.current_state[0]
        current_velocity = self.current_state[1]

        velocity = current_velocity + (0.001 * action) - (0.0025 * np.cos(3 * current_position))
        position = current_position + velocity

        if velocity > 0.07:
            velocity = 0.07
        elif velocity < -0.07:
            velocity = -0.07

        if position < -1.2:
            position = -1.2
            velocity = 0.0
        elif position > 0.5:
            position = 0.5
            terminate = True

        reward = -1 if not terminate else 0
        if self.store_summary:
            self.reward_per_step[self.config.current_step - 1] = reward

        self.current_state = np.array((position, velocity), dtype=np.float64)

        if self.norm_state:
            return self.normalize(self.current_state), reward, terminate, timeout
        else:
            return self.current_state, reward, terminate, timeout

    @staticmethod
    def normalize(state):
        """ normalize to [-1, 1] """
        temp_state = np.zeros(shape=2, dtype=np.float64)
        temp_state[0] = (state[0] + 0.35) / 0.85

        temp_state[1] = (state[1]) / 0.07
        return temp_state

    def get_current_state(self):
        if self.norm_state:
            return self.normalize(self.current_state)
        else:
            return self.current_state


if __name__ == "__main__":
    verbose = False
    random_policy_test = True
    pumping_action_test = True
    actions = 3

    config = Config()
    config.norm_state = True
    config.store_summary = True
    config.max_episode_length = 1000000
    steps = 10000

    if random_policy_test:
        print("==== Results with Random Policy ====")

        config.current_step = 0
        summary = {}

        env = MountainCar(config, summary=summary)

        cumulative_reward = 0
        terminations = 0
        successful_episode_steps = []

        for i in range(steps):
            A = np.random.randint(actions)
            old_state = env.get_current_state()
            next_S, R, terminate, timeout = env.step(A)
            if verbose:
                print("Old state:", np.round(old_state, 3), "-->",
                      "Action:", A, "-->",
                      "New state:", np.round(next_S, 3))
            cumulative_reward += R
            if terminate or timeout:
                if verbose:
                    print("\n## Reset ##\n")
                if terminate:
                    terminations += 1
                    successful_episode_steps.append(env.episode_step_count)
                env.reset()

        print("Number of steps per episode:", summary['steps_per_episode'])
        print("Number of episodes that reached the end:", terminations)
        average_length = np.average(successful_episode_steps) if len(successful_episode_steps) > 0 else np.inf
        print("The average number of steps per episode was:", average_length)
        print("Cumulative reward:", cumulative_reward)
        print("\n\n")

    if pumping_action_test:
        print("==== Results with Pumping Action Policy ====")

        config.current_step = 0
        summary = {}

        env = MountainCar(config, summary=summary)

        for i in range(steps):
            current_state = env.get_current_state()
            A = 1 + np.sign(current_state[1])
            old_state = env.get_current_state()
            next_S, R, terminate, timeout = env.step(A)
            if verbose:
                print("Old state:", np.round(old_state, 3), "-->",
                      "Action:", A, "-->",
                      "New state:", np.round(next_S, 3))
            if terminate or timeout:
                if verbose:
                    print("\n## Reset ##\n")
                env.reset()

        steps_per_episode = summary['steps_per_episode']
        reward_per_step = summary['reward_per_step']

        def compute_return_per_episode(rps, spe):
            rpe = np.zeros(len(spe), dtype=np.float64)
            previous_index = 0
            for i in range(len(spe)):
                rpe[i] = np.sum(rps[previous_index:(previous_index + spe[i])])
                previous_index = previous_index + spe[i]
            return rpe

        return_per_episode = compute_return_per_episode(reward_per_step, steps_per_episode)
        print("Number of steps per episode:", summary['steps_per_episode'])
        print("Number of successful episodes:", len(summary['steps_per_episode']))
        print("Return per episode:", return_per_episode)
        for i in range(len(return_per_episode)):
            assert return_per_episode[i] == -1 * (steps_per_episode[i] - 1)
        print("The average return per episode is:", np.mean(return_per_episode))
