
BEST_PARAMETERS_DICTIONARY = {
    'mountain_car': {
        # found by using a sweep with sample size of 30
        # ties were broken by selecting the parameter combination with the highest lower confidence bound
        'DQN': {
            # Buffer Size
            100: {'Freq': 400, 'LearningRate': 0.01},
            1000: {'Freq': 10, 'LearningRate': 0.004},
            5000: {'Freq': 10, 'LearningRate': 0.004},
            20000: {'Freq': 10, 'LearningRate': 0.004},
            80000: {'Freq': 10, 'LearningRate': 0.001},
            'ParameterNames': ['BufferSize', 'Freq', 'LearningRate']
        },
    }
}


class Config:
    """
    Used to store the arguments of all the functions in the package Experiment_Engine. If a function requires arguments,
    it's definition will be: func(config), where config has all the parameters needed for the function.
    """
    def __init__(self):
        pass

    def print(self):
        for k, v in self.__dict__.items():
            print("Parameter name: {0}, \tParameter Value: {1}".format(k, v))


def check_attribute(object_type, attr_name, default_value, choices=None, data_type=None):
    if not hasattr(object_type, attr_name):
        print("Creating attribute", attr_name)
        setattr(object_type, attr_name, default_value)
    if choices:
        if getattr(object_type, attr_name) not in choices:
            raise ValueError("The possible values for this attribute are: " + str(choices))
    if data_type:
        if not isinstance(default_value, data_type):
            raise ValueError("Wrong data type. The expected data type is: " + str(data_type))
    return getattr(object_type, attr_name)


def check_dict_else_default(dict_type, key_name, default_value):
    assert isinstance(dict_type, dict)
    if key_name not in dict_type.keys():
        dict_type[key_name] = default_value
    return dict_type[key_name]


def check_multiple_attributes(object_type, *attributes):
    """
    checks if an object has the desired arguments
    attributes should be a list of 3-tuples with ("name", default_value, [choices], data_type))
    """
    values = []
    for attr in attributes:
        assert len(attr) == 4
        values.append(
            check_attribute(object_type, attr_name=attr[0], default_value=attr[1], choices=attr[2],
                            data_type=attr[3])
        )
    return values
