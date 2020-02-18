import numpy as np

from Experiment_Engine.util import check_attribute, check_dict_else_default, Config


class Agent:
    """
    Summary Name: return_per_episode
    """

    def __init__(self, environment, function_approximator, config=None, summary=None):
        self.config = config or Config()
        assert isinstance(config, Config)
        """ 
        Parameters in config:
        Name:                   Type:           Default:            Description: (Omitted when self-explanatory)
        store_summary           bool            False               store the summary of the agent (return per episode)
        """
        self.store_summary = check_attribute(self.config, 'store_summary', False)
        if self.store_summary:
            assert isinstance(summary, dict)
            self.summary = summary
            check_dict_else_default(self.summary, 'return_per_episode', [])

        " Other Parameters "
        # Function Approximator: used to approximate the Q-Values
        self.fa = function_approximator
        # Environment that the agent is interacting with
        self.env = environment
        # Summaries
        self.cumulative_reward = 0

    def train(self, num_episodes):
        if num_episodes == 0: return

        for episode in range(num_episodes):
            # Current State, Action, and Q_values
            S = self.env.get_current_state()
            A = self.fa.choose_action(S)

            T = np.inf
            t = 0

            while t != T:
                # Step in the environment
                new_S, R, terminate, timeout = self.env.step(A)
                # Record Keeping
                self.cumulative_reward += R

                if terminate:
                    T = t + 1
                    new_A = 0
                else:
                    if timeout:
                        T = t + 1
                    new_A = self.fa.choose_action(S)
                self.fa.update(S, A, R, new_S, new_A, terminate)
                S = new_S
                A = new_A

                t += 1
            # End of episode
            self.env.reset()
            if self.store_summary:
                self.summary["return_per_episode"].append(self.cumulative_reward)
                self.cumulative_reward = 0
