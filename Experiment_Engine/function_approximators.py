import numpy as np
import torch

from Experiment_Engine.networks import TwoLayerFullyConnected, ActionNeuralNetwork, weight_init, \
    GatedActionNeuralNetwork, NormActionNeuralNetwork
from Experiment_Engine.util import *


class NeuralNetworkFunctionApproximation:
    """
    Parent class for all the neural networks
    summary: loss_per_step
    """
    def __init__(self, config, summary=None):
        """
        Config --- class that contains all the parameters in used in an experiment.
        Parameters in config:
        Name:                   Type:           Default:            Description: (Omitted when self-explanatory)
        num_actions             int             3                   Number of actions available to the agent
        gamma                   float           1.0                 discount factor
        epsilon                 float           0.1                 exploration parameter
        state_dims              int             2                   number of dimensions of the environment's states
        lr                      float           0.001               learning rate

        # DQN parameters
        tnet_update_freq        int             10                  the update frequency of the target network

        # Parameters for storing summaries
        store_summary           bool            False               store the summary of the agent
        number_of_steps         int             500000              Total number of environment steps
        """
        assert isinstance(config, Config)
        check_attribute(config, 'current_step', 0)
        self.config = config

        self.num_actions = check_attribute(config, 'num_actions', 3)
        self.gamma = check_attribute(config, 'gamma', 1.0)
        self.epsilon = check_attribute(config, 'epsilon', 0.1)
        self.state_dims = check_attribute(config, 'state_dims', 2)
        self.lr = check_attribute(config, 'lr', 0.001)
        # DQN parameters
        self.batch_size = 32
        self.tnet_update_freq = check_attribute(config, 'tnet_update_freq', 10)
        self.replay_buffer = ReplayBuffer(config)
        # summary parameters
        self.store_summary = check_attribute(config, 'store_summary', False)
        self.number_of_steps = check_attribute(config, 'number_of_steps', 500000)

        if self.store_summary:
            assert isinstance(summary, dict)
            self.summary = summary
            self.loss_per_step = np.zeros(self.number_of_steps, dtype=np.float64)
            check_dict_else_default(self.summary, 'loss_per_step', self.loss_per_step)

        # policy network
        self.net = TwoLayerFullyConnected(self.config)
        self.net.apply(weight_init)
        # target network
        self.target_net = TwoLayerFullyConnected(self.config)
        self.target_net.apply(weight_init)

        self.optimizer = torch.optim.Adam(self.net.parameters(), lr=self.lr)

    def compute_return(self, reward, state, termination):
        # Computes the Qlearning return
        with torch.no_grad():
            av_function = torch.max(self.target_net.forward(state), dim=1)[0]
            next_step_bool = torch.from_numpy((1 - np.int64(termination))).float()
            qlearning_return = torch.from_numpy(reward).float() + next_step_bool * self.gamma * av_function
        return qlearning_return

    def choose_action(self, state):
        p = np.random.rand()
        if p > self.epsilon:
            with torch.no_grad():
                # it is extremely unlikely (prob = 0) for there to be two actions with exactly the same action value
                optim_action = self.net.forward(state).argmax().numpy()
            return np.int64(optim_action)
        else:
            return np.random.randint(self.num_actions)

    def save_summary(self, current_loss):
        if not self.store_summary:
            return
        self.summary['loss_per_step'][self.config.current_step - 1] = current_loss

    def update_target_network(self):
        if (self.config.current_step % self.tnet_update_freq) == 0:
            self.target_net.load_state_dict(self.net.state_dict())


class ReplayBuffer:

    def __init__(self, config):
        """
        Parameters in config:
        Name:                   Type:           Default:            Description: (Omitted when self-explanatory)
        state_dims              int             2                   number of dimensions of the environment's state
        buffer_size             int             100                 size of the buffer
        """
        self.state_dims = check_attribute(config, 'state_dims', 2)
        self.buffer_size = check_attribute(config, 'buffer_size', 100)

        """ inner state """
        self.start = 0
        self.length = 0

        self.state = np.empty((self.buffer_size, self.state_dims), dtype=np.float64)
        self.action = np.empty(self.buffer_size, dtype=int)
        self.reward = np.empty(self.buffer_size, dtype=np.float64)
        self.next_state = np.empty((self.buffer_size, self.state_dims), dtype=np.float64)
        self.next_action = np.empty(self.buffer_size, dtype=int)
        self.termination = np.empty(self.buffer_size, dtype=bool)

    def __getitem__(self, idx):
        if isinstance(idx, int):
            if idx < 0 or idx >= self.length:
                raise KeyError()
        elif isinstance(idx, np.ndarray):
            if (idx < 0).any() or (idx >= self.length).any():
                raise KeyError()
        shifted_idx = self.start + idx
        s = self.state.take(shifted_idx, axis=0, mode='wrap')
        a = self.action.take(shifted_idx, axis=0, mode='wrap')
        r = self.reward.take(shifted_idx, axis=0, mode='wrap')
        next_s = self.next_state.take(shifted_idx, axis=0, mode='wrap')
        next_a = self.next_action.take(shifted_idx, axis=0, mode='wrap')
        terminate = self.termination.take(shifted_idx, axis=0, mode='wrap')
        return s, a, r, next_s, next_a, terminate

    def store_transition(self, transition):
        if self.length < self.buffer_size:
            self.length += 1
        elif self.length == self.buffer_size:
            self.start = (self.start + 1) % self.buffer_size
        else:
            raise RuntimeError()

        storing_idx = (self.start + self.length - 1) % self.buffer_size
        state, action, reward, next_state, next_action, termination = transition
        self.state[storing_idx] = state
        self.action[storing_idx] = action
        self.reward[storing_idx] = reward
        self.next_state[storing_idx] = next_state
        self.next_action[storing_idx] = next_action
        self.termination[storing_idx] = termination

    def sample(self, sample_size):
        if sample_size > self.length or sample_size > self.buffer_size:
            raise ValueError("The sample size is to large.")
        sampled_idx = np.random.randint(0, self.length, sample_size)                    # Sample any indices
        # sampled_idx = np.random.choice(self.length, size=sample_size, replace=False)  # Sample unique indices
        return self.__getitem__(sampled_idx)


class VanillaDQN(NeuralNetworkFunctionApproximation):

    def __init__(self, config, summary=None):
        super(VanillaDQN, self).__init__(config, summary)

    def update(self, state, action, reward, next_state, next_action, termination):
        self.replay_buffer.store_transition(transition=(state, action, reward, next_state, next_action, termination))

        if self.replay_buffer.length < self.batch_size:
            self.save_summary(0)
            return

        state, action, reward, next_state, next_action, termination = self.replay_buffer.sample(self.batch_size)
        qlearning_return = self.compute_return(reward, next_state, termination)
        self.optimizer.zero_grad()
        prediction = torch.squeeze(self.net(state).gather(1, torch.from_numpy(action).view(-1,1)))
        loss = (qlearning_return - prediction).pow(2).mean()
        loss.backward()
        self.optimizer.step()

        self.save_summary(loss.detach().numpy())
        self.update_target_network()


class ActionDQN(NeuralNetworkFunctionApproximation):

    def __init__(self, config, summary=None):
        super(ActionDQN, self).__init__(config, summary)
        gated = check_attribute(config, 'gated', False)
        # policy network
        if not gated:
            self.net = ActionNeuralNetwork(config)
            self.target_net = ActionNeuralNetwork(config)
        else:
            self.net = GatedActionNeuralNetwork(config)
            self.target_net = GatedActionNeuralNetwork(config)
        self.net.apply(weight_init)
        self.target_net.apply(weight_init)
        self.target_net.eval()
        self.optimizer = torch.optim.Adam(self.net.parameters(), lr=self.lr)

    def compute_return(self, reward, state, termination):
        # Computes the Qlearning return
        with torch.no_grad():
            av_function = torch.max(self.target_net.forward(state, a=None, full=True), dim=1)[0]  # [0] = values, [1] = indices
            next_step_bool = torch.from_numpy((1 - np.int64(termination))).float()
            qlearning_return = torch.from_numpy(reward).float() + next_step_bool * self.gamma * av_function
        return qlearning_return

    def choose_action(self, state):
        p = np.random.rand()
        if p > self.epsilon:
            self.net.eval()
            with torch.no_grad():
                # it is extremely unlikely (prob = 0) for there to be two actions with exactly the same action value
                state = state.reshape([1, self.state_dims])
                optim_action = self.net.forward(state, a=None, full=True).flatten().argmax().numpy()
            self.net.train()
            return np.int64(optim_action)
        else:
            return np.random.randint(self.num_actions)

    def update(self, state, action, reward, next_state, next_action, termination):
        self.replay_buffer.store_transition(transition=(state, action, reward, next_state, next_action, termination))

        if self.replay_buffer.length < self.batch_size:
            self.save_summary(0)
            return

        state, action, reward, next_state, next_action, termination = self.replay_buffer.sample(self.batch_size)
        qlearning_return = self.compute_return(reward, next_state, termination)
        self.optimizer.zero_grad()
        prediction = torch.squeeze(self.net(state, action))
        loss = (qlearning_return - prediction).pow(2).mean()
        loss.backward()
        self.optimizer.step()
        # if self.config.current_step % 1000 == 0:
        #     print("Step: {0},\t Loss: {1}".format(self.config.current_step, loss))
        self.save_summary(loss.detach().numpy())
        self.update_target_network()
        self.target_net.eval()


class NormActionDQN(NeuralNetworkFunctionApproximation):

    def __init__(self, config, summary=None):
        super(NormActionDQN, self).__init__(config, summary)

        # policy network
        self.net = NormActionNeuralNetwork(config)
        self.target_net = NormActionNeuralNetwork(config)

        self.net.apply(weight_init)
        self.target_net.apply(weight_init)
        self.target_net.eval()
        self.optimizer = torch.optim.Adam(self.net.parameters(), lr=self.lr)

    def compute_return(self, reward, state, termination):
        # Computes the Qlearning return
        with torch.no_grad():
            av_function = torch.max(self.target_net.forward(state, a=None, full=True), dim=1)[0]  # [0] = values, [1] = indices
            next_step_bool = torch.from_numpy((1 - np.int64(termination))).float()
            qlearning_return = torch.from_numpy(reward).float() + next_step_bool * self.gamma * av_function
        return qlearning_return

    def choose_action(self, state):
        p = np.random.rand()
        if p > self.epsilon:
            self.net.eval()
            with torch.no_grad():
                # it is extremely unlikely (prob = 0) for there to be two actions with exactly the same action value
                state = state.reshape([1, self.state_dims])
                optim_action = self.net.forward(state, a=None, full=True).flatten().argmax().numpy()
            self.net.train()
            return np.int64(optim_action)
        else:
            return np.random.randint(self.num_actions)

    def update(self, state, action, reward, next_state, next_action, termination):
        self.replay_buffer.store_transition(transition=(state, action, reward, next_state, next_action, termination))

        if self.replay_buffer.length < self.batch_size:
            self.save_summary(0)
            return

        state, action, reward, next_state, next_action, termination = self.replay_buffer.sample(self.batch_size)
        qlearning_return = self.compute_return(reward, next_state, termination)
        self.optimizer.zero_grad()
        prediction = torch.squeeze(self.net(state, action))
        loss = (qlearning_return - prediction).pow(2).mean()
        loss.backward()
        self.optimizer.step()
        # if self.config.current_step % 1000 == 0:
        #     print("Step: {0},\t Loss: {1}".format(self.config.current_step, loss))
        self.save_summary(loss.detach().numpy())
        self.update_target_network()
        self.target_net.eval()
