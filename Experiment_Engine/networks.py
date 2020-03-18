import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

from Experiment_Engine.util import Config, check_attribute


class TwoLayerFullyConnected(nn.Module):

    def __init__(self, config):
        super(TwoLayerFullyConnected, self).__init__()
        self.config = config
        # format: check_attribute( config_class, attribute_name, default_value, data_type)  # description (optional)
        input_dims = check_attribute(self.config, 'input_dims', 1, data_type=int)
        h1_dims = check_attribute(self.config, 'h1_dims', 1, data_type=int)         # neurons in hidden layer 1
        h2_dims = check_attribute(self.config, 'h2_dims', 1, data_type=int)         # neurons in hidden layer 2
        output_dims = check_attribute(self.config, 'output_dims', 1, data_type=int)

        self.fc1 = nn.Linear(input_dims, h1_dims, bias=True)
        self.fc2 = nn.Linear(h1_dims, h2_dims, bias=True)
        self.fc3 = nn.Linear(h2_dims, output_dims, bias=False)

    def forward(self, x, a=None, return_activations=False):
        x = to_variable(x)
        z1 = self.fc1(x)            # Layer 1: z1 = W1^T x + b1
        x1 = F.relu(z1)             # Layer 1: x1 = gate1(z1)
        z2 = self.fc2(x1)           # Layer 2: z2 = W2^T x1 + b2
        x2 = F.relu(z2)             # Layer 2: x2 = gate2(z2)
        x3 = self.fc3(x2)           # Output Layer: x3 = W3^T x2
        if not return_activations:
            return x3
        else:
            return x1, x2, x3


class ActionNeuralNetwork(nn.Module):

    def __init__(self, config):
        super(ActionNeuralNetwork, self).__init__()
        self.config = config
        # format: check_attribute( config_class, attribute_name, default_value, data_type)  # description (optional)
        input_dims = check_attribute(self.config, 'input_dims', 1, data_type=int)
        h1_dims = check_attribute(self.config, 'h1_dims', 1, data_type=int)  # neurons in hidden layer 1
        h2_dims = check_attribute(self.config, 'h2_dims', 1, data_type=int)  # neurons in hidden layer 2
        self.num_actions = check_attribute(self.config, 'num_actions', 1, data_type=int)
        ppa = check_attribute(self.config, 'ppa', 0.1, data_type=float)     # proportion of neurons per action

        self.fc1 = nn.Linear(input_dims, h1_dims, bias=True)
        self.fc2 = nn.Linear(h1_dims, h2_dims, bias=True)
        self.fc3 = nn.Linear(h2_dims, 1, bias=False)
        self.npa = np.int64(np.floor(h2_dims * ppa))
        assert self.npa * self.num_actions <= h2_dims, "Too many neurons per action!"

        self.masks = torch.zeros((self.num_actions, h2_dims))
        shared_neurons = np.int64(h2_dims - self.npa * self.num_actions)
        exclusive_neurons = 0
        for i in range(self.num_actions):
            self.masks[i][0:shared_neurons] += 1
            self.masks[i][(shared_neurons + exclusive_neurons):(shared_neurons + exclusive_neurons + self.npa)] += 1
            exclusive_neurons += self.npa

    def forward(self, x, a, return_activations=False, full=False):
        x = to_variable(x)
        z1 = self.fc1(x)            # Layer 1: z1 = W1^T x + b1
        x1 = F.relu(z1)             # Layer 1: x1 = gate1(z1)
        z2 = self.fc2(x1)           # Layer 2: z2 = W2^T x1 + b2
        x2 = F.relu(z2)             # Layer 2: x2 = gate2(z2)
        if not full:
            x2 = x2 * self.masks[a.flatten()]
            x3 = self.fc3(x2)           # Output Layer: x3 = W3^T x2
            if not return_activations:
                return x3
            else:
                return x1, x2, x3
        else:
            action_values = None
            for i in range(self.num_actions):
                m = self.masks[i]
                temp_x2 = x2 * m
                x3 = self.fc3(temp_x2)
                if action_values is None:
                    action_values = x3
                else:
                    action_values = torch.cat((action_values, x3), dim=1)
            return action_values
            pass

    def full_forward(self, x):
        action_values = torch.zeros(self.num_actions)
        for a in range(0, self.num_actions):
            action_values[a] += self.forward(x, torch.from_numpy(np.array(a)))[0, 0]
        return action_values


class GatedActionNeuralNetwork(nn.Module):

    def __init__(self, config):
        super(GatedActionNeuralNetwork, self).__init__()
        self.config = config
        # format: check_attribute( config_class, attribute_name, default_value, data_type)  # description (optional)
        input_dims = check_attribute(self.config, 'input_dims', 1, data_type=int)
        h1_dims = check_attribute(self.config, 'h1_dims', 1, data_type=int)  # neurons in hidden layer 1
        h2_dims = check_attribute(self.config, 'h2_dims', 1, data_type=int)  # neurons in hidden layer 2
        self.num_actions = check_attribute(self.config, 'num_actions', 1, data_type=int)
        self.gate_function = check_attribute(self.config, 'gate_function', 'tanh', data_type=str)   # mask gate function

        self.fc1 = nn.Linear(input_dims, h1_dims, bias=True)
        self.fc2 = nn.Linear(h1_dims, h2_dims, bias=True)
        self.fc3 = nn.Linear(h2_dims, 1, bias=False)
        if self.gate_function == 'sigmoid':
            self.gf = torch.sigmoid
        elif self.gate_function == 'tanh':
            self.gf = torch.tanh
        elif self.gate_function == 'noisy_relu':
            self.gf = lambda x: torch.relu(x + torch.empty(x.shape).normal_(mean=0, std=1))
        else:
            raise ValueError("Choose one of the following gate functions: sigmoid, tanh")

        self.action_gates = nn.Parameter(torch.randn((self.num_actions, h2_dims)), requires_grad=True)
        self.action_gates_bias = nn.Parameter(torch.randn(h2_dims), requires_grad=True)

        self.action_indices = torch.arange(start=0, end=self.num_actions, dtype=torch.int64)

    def forward(self, x, a, return_activations=False, full=False):
        x = to_variable(x)
        z1 = self.fc1(x)            # Layer 1: z1 = W1^T x + b1
        x1 = F.relu(z1)             # Layer 1: x1 = gate1(z1)
        z2 = self.fc2(x1)           # Layer 2: z2 = W2^T x1 + b2
        x2 = F.relu(z2)             # Layer 2: x2 = gate2(z2)
        if not full:
            assert a is not None
            g = self.gf(self.action_gates[a.flatten()] + self.action_gates_bias)
            x2 = x2 * g
            x3 = self.fc3(x2)  # Output Layer: x3 = W3^T x2
            if not return_activations:
                return x3
            else:
                return x1, x2, x3, g
        else:
            action_values = None
            for i in range(self.num_actions):
                if self.gate_function == 'noisy_relu':
                    g = torch.relu(self.action_gates[i] + self.action_gates_bias)
                else:
                    g = self.gf(self.action_gates[i] + self.action_gates_bias)
                temp_x2 = x2 * g
                x3 = self.fc3(temp_x2)
                if action_values is None:
                    action_values = x3
                else:
                    action_values = torch.cat((action_values, x3), dim=1)
            return action_values


class BatchNormActionNeuralNetwork(nn.Module):

    def __init__(self, config):
        super(BatchNormActionNeuralNetwork, self).__init__()
        self.config = config
        # format: check_attribute( config_class, attribute_name, default_value, data_type)  # description (optional)
        input_dims = check_attribute(self.config, 'input_dims', 1, data_type=int)
        h1_dims = check_attribute(self.config, 'h1_dims', 1, data_type=int)  # neurons in hidden layer 1
        h2_dims = check_attribute(self.config, 'h2_dims', 1, data_type=int)  # neurons in hidden layer 2
        self.num_actions = check_attribute(self.config, 'num_actions', 1, data_type=int)

        self.fc1 = nn.Linear(input_dims, h1_dims, bias=True)
        self.fc2 = nn.Linear(h1_dims, h2_dims, bias=True)
        self.bn2 = nn.BatchNorm1d(h2_dims, affine=False)
        self.fc3 = nn.Linear(h2_dims, 1, bias=False)

        self.action_scales = nn.Parameter(torch.randn((self.num_actions, h2_dims)), requires_grad=True)
        self.action_shifts = nn.Parameter(torch.randn(self.num_actions, h2_dims), requires_grad=True)

    def forward(self, x, a, return_activations=False, full=False):
        x = to_variable(x)
        z1 = self.fc1(x)            # Layer 1: z1 = W1^T x + b1
        x1 = F.relu(z1)             # Layer 1: x1 = gate1(z1)
        z2 = self.fc2(x1)           # Layer 2: z2 = W2^T x1 + b2
        if not full:
            assert a is not None
            centered_z2 = self.bn2(z2)
            affine_z2 = centered_z2 * self.action_scales[a] + self.action_shifts[a]
            x2 = F.relu(affine_z2)
            x3 = self.fc3(x2)  # Output Layer: x3 = W3^T x2
            if not return_activations:
                return x3
            else:
                return x1, x2, x3, g
        else:
            action_values = None
            for i in range(self.num_actions):
                centered_z2 = self.bn2(z2)
                affine_z2 = centered_z2 * self.action_scales[i] + self.action_shifts[i]
                x2 = F.relu(affine_z2)
                x3 = self.fc3(x2)
                if action_values is None:
                    action_values = x3
                else:
                    action_values = torch.cat((action_values, x3), dim=1)
            return action_values


def to_variable(x):
    if isinstance(x, torch.autograd.Variable):
        return x
    elif isinstance(x, np.ndarray):
        x = np.float64(x)
    x = torch.from_numpy(x).float()
    return torch.autograd.Variable(x)


def weight_init(m):
    # Initializes the weights of the linear layers according to a N(0, 2/nl), where nl is the size of the layers.
    # Biases are initialized with a value of zero.
    if isinstance(m, nn.Linear):
        size = m.weight.size()
        std_dev = np.sqrt(2 / np.prod(size))
        m.weight.data.normal_(0, std_dev)
        if m.bias is not None:
            m.bias.data.uniform_(0, 0)


if __name__ == "__main__":
    import numpy as np
    import argparse

    ###################
    " Argument Parser "
    ###################
    parser = argparse.ArgumentParser()
    parser.add_argument("-minibatch_size", action='store', default=np.int8(32))
    parser.add_argument("-lr", action='store', default=np.float64(0.001))
    parser.add_argument("-threshold", action='store', default=1e-4, type=float)
    parser.add_argument('-regularization', action='store', default='none', type=str, choices=['none', 'l1', 'l2'])
    parser.add_argument('-reg_factor', action='store', default=0.0001, type=float)
    parser.add_argument('-test_copy_params', action='store_true', default=False)
    parser.add_argument('-init_test', action='store_true', default=False)
    parser.add_argument('-simple_training_test', action='store_true', default=False)
    parser.add_argument('-copy_parameters_test', action='store_true', default=False)
    parser.add_argument('-networks_comparison_test', action='store_true', default=False)
    args = parser.parse_args()

    config = Config()

    ############################################
    " Example: initializing the neural network "
    ############################################
    if args.init_test:
        config.input_dims = 2
        config.h1_dims = 2
        config.h2_dims = 2
        config.output_dims = 1
        print("Creating Two Layer Fully Connected Network...")
        network = TwoLayerFullyConnected(config)
        network.apply(weight_init)

        print("Printing Network...")
        print("\t", network, "\n")
        print("Printing Network Parameters:")
        for name, parameter in network.named_parameters():
            print("The parameter name is:", name)
            print("The parameter values are:", parameter)
            # Printing an example of an output
        print("Printing output for ten (1,2) inputs...")
        output = network.forward(torch.from_numpy(np.random.uniform(size=(10,2), low=0, high=1)).float())
        print("\t", output, "\n")

    ########################################
    " Example: training the neural network "
    ########################################
    if args.simple_training_test or args.copy_parameters_test:
        config.input_dims = 2
        config.h1_dims = 32
        config.h2_dims = 256
        config.output_dims = 1
        network = TwoLayerFullyConnected(config)
        network.apply(weight_init)
        print("Learning the function f(x,y) = x^2 + y^2...")
        print("Parameters:")
            # mini-batch size
        minibatch = args.minibatch_size
        print("\tmini-batch size:", minibatch)
            # learning rate
        learning_rate = args.lr
        print("\tlearning rate:", learning_rate)
            # optimizer
        optimizer = torch.optim.Adam(network.parameters(), lr=learning_rate)
            # threshold
        threshold = 1e-6
        print("\tstopping criteria: loss <", threshold, "\n")

        current_loss = np.inf
        training_steps = 0
        while current_loss > threshold:

            inputs = np.random.uniform(size=(minibatch, 2), low=0, high=1)
            x = inputs[:,0]
            y = inputs[:,1]
            fxy = torch.from_numpy(x**2 + y**2).float()

            optimizer.zero_grad()
            prediction = network(torch.from_numpy(inputs).float(), a=None).view(1,-1)
            loss = torch.mean((prediction - fxy)**2)

            if args.regularization != 'none':
                if args.regularization == 'l1': reg_function = torch.abs
                elif args.regularization == 'l2': reg_function = lambda z: torch.pow(z, 2)
                reg_factor = args.reg_factor
                reg_loss = 0
                for name, param in network.named_parameters():
                    reg_loss += torch.sum(reg_function(param))
                loss += reg_factor * reg_loss
            loss.backward()
            optimizer.step()

            training_steps += 1
            current_loss = loss.detach().numpy()
            if training_steps % 1000 == 0:
                print("\tIteration number:", training_steps)
                print("\tloss:", current_loss)
        print("The network finished training after", training_steps, "training steps.\n")

        print("Testing the network's predictions...")
        test_size = 10
        test_inputs = np.random.uniform(size=(test_size,2), low=0, high=1)
        print("The inputs are:")
        for i in range(test_size):
            print("\t", test_inputs[i])
        print("The predictions are:")
        test_labels = test_inputs[:, 0] ** 2 + test_inputs[:, 1] ** 2
        test_predictions = network(torch.from_numpy(test_inputs).float(), a=0).detach().numpy().flatten()
        for i in range(test_size):
            print("\tTrue Value:", np.round(test_labels[i], 3),
                  "\tPrediction:", np.round(test_predictions[i], 3))

        #########################################################
        " Test: copying parameters from one network onto another"
        #########################################################
        if args.copy_parameters_test:
            print("\n\nInitializing another network with the same architecture...")
            network2 = TwoLayerFullyConnected(config)
            print(network2)

            print("Testing the new network's predictions...")
            test_inputs = np.random.uniform(size=(test_size, 2), low=0, high=1)
            test_labels = test_inputs[:, 0] ** 2 + test_inputs[:, 1] ** 2
            test_predictions = network2(torch.from_numpy(test_inputs).float()).detach().numpy().flatten()
            for i in range(test_size):
                print("\tTrue Value:", np.round(test_labels[i], 3),
                      "\tPrediction:", np.round(test_predictions[i], 3))

            print("Copying the parameters from the trained network onto the new network...")
            network2.load_state_dict(network.state_dict())

            print("Testing the new network's predictions again...")
            test_labels = test_inputs[:, 0] ** 2 + test_inputs[:, 1] ** 2
            test_predictions = network2(torch.from_numpy(test_inputs).float()).detach().numpy().flatten()
            test_predictions_old = network(torch.from_numpy(test_inputs).float()).detach().numpy().flatten()
            for i in range(test_size):
                print("\tTrue Value:", np.round(test_labels[i], 3),
                      "\tPrediction:", np.round(test_predictions[i], 3),
                      "\tOld Network Prediction:", np.round(test_predictions_old[i], 3))

    ####################################################
    """ Example: Action NN vs DQN-like NN Performance"""
    ####################################################
    if args.networks_comparison_test:
        minibatch = args.minibatch_size
        learning_rate = args.lr
        threshold = 1e-5

        print("Learning the function f((x,y), a) = x^2 + y^2 + a, for a in {0, 1}")
        print("Parameters:")
        print("\tmini-batch size:", minibatch)
        print("\tlearning rate:", learning_rate)
        print("\tstopping criteria: loss <", threshold, "\n")

        def get_sample(n):
            inputs = np.random.uniform(low=0, high=1, size=(n, 2))
            actions = np.random.randint(low=0, high=2, size=(n, 1))
            x = inputs[:, 0]
            y = inputs[:, 1]
            fxy = torch.from_numpy(x ** 2 + y ** 2 + actions.flatten()).float()
            return torch.from_numpy(inputs).float(), torch.from_numpy(actions), fxy

        dqn_config = Config()
        dqn_config.input_dims = 2
        dqn_config.h1_dims = 32
        dqn_config.h2_dims = 256
        dqn_config.output_dims = 2
        dqn_network = TwoLayerFullyConnected(dqn_config)
        dqn_network.apply(weight_init)
        dqn_optimizer = torch.optim.Adam(dqn_network.parameters(), lr=learning_rate)

        actnn_config = Config()
        actnn_config.input_dims = 2
        actnn_config.h1_dims = 32
        actnn_config.h2_dims = 256
        actnn_config.num_actions = 2
        actnn_config.ppa = 0.1
        actnn_network = ActionNeuralNetwork(actnn_config)
        actnn_network.apply(weight_init)
        actnn_optimizer = torch.optim.Adam(actnn_network.parameters(), lr=learning_rate)

        networks = [('DQN-Like Network', dqn_network, dqn_optimizer),
                    ('Action Network', actnn_network, actnn_optimizer)]

        for network in networks:
            name, net, opt = network
            print("\nTraining " + name + "...")
            current_loss = np.inf
            training_steps = 0
            while current_loss > threshold:

                inputs, actions, fxy = get_sample(minibatch)

                opt.zero_grad()
                if name == 'DQN-Like Network':
                    prediction = net(inputs, a=None).gather(1, actions)
                else:
                    prediction = net(inputs, a=actions)
                loss = torch.mean((prediction.squeeze() - fxy) ** 2)

                loss.backward()
                opt.step()

                training_steps += 1
                current_loss = loss.detach().numpy()
                if training_steps % 1000 == 0:
                    print("\tIteration number:", training_steps)
                    print("\tloss:", current_loss)
            print("\tThe network finished training after", training_steps, "training steps.\n")
            print("\tTesting...")
            test_inputs, test_actions, test_fxy = get_sample(10)
            if name == 'DQN-Like Network':
                test_predictions = net(test_inputs, a=None).detach().numpy()
            else:
                test_predictions = net(test_inputs, a=test_actions).detach().numpy()
            for i in range(10):
                print("\t\tInput:", np.round(test_inputs[i].numpy(), 3),
                      "\tActions:", test_actions[i].numpy(),
                      "\tTrue Values:", np.round(test_fxy[i].numpy(), 3),
                      "\tPredicted Values:", np.round(test_predictions[i], 3))
