from scipy.stats import t, chi2   # Faster than importing scipy.stats
import numpy as np
import os
import pickle

PARAMETER_NAME_DICT = {'LearningRate': 'learning_rate', 'BufferSize': 'buffer', 'Freq': 'freq', 'Beta': 'beta',
                       'RegFactor': 'reg_factor', 'DropoutProbability': 'dropout_prob'}


####################################
# Statistical comparison functions #
####################################
# Confidence interval for sample average
def compute_tdist_confidence_interval(sample_mean, sample_std, alpha_proportion, sample_size):
    if sample_size <= 1:
        return None, None, None

    dof = sample_size - 1
    t_dist = t(df=dof)  # from scipy.stats
    tdist_factor = t_dist.ppf(1 - alpha_proportion / 2)

    sqrt_inverse_sample_size = np.sqrt(1 / sample_size)
    me = sample_std * tdist_factor * sqrt_inverse_sample_size
    upper_bound = sample_mean + me
    lower_bound = sample_mean - me

    return upper_bound, lower_bound, me


# Confidence interval for sample variance
def compute_chi2dist_confidence_interval(sample_variance, proportion, sample_size):
    dof = sample_size - 1
    chi2_dist = chi2(dof)

    upper_proportion = 1 - (proportion / 2)
    lower_proportion = proportion / 2

    upper_chi2_value = chi2_dist.ppf(upper_proportion)
    lower_chi2_value = chi2_dist.ppf(lower_proportion)

    # the lower and upper chi2_values get inverted because we're bounding sigma^2 in the equation:
    # chi2.ppf(proportion/2) < (n-1) s^2 / sigma^2 < chi2.ppf(1 - proportion/2)
    upper_bound_ci = (dof * sample_variance) / lower_chi2_value
    lower_bound_ci = (dof * sample_variance) / upper_chi2_value

    return upper_bound_ci, lower_bound_ci


# Welch's Test for difference in sample averages
def compute_welchs_test(mean1, std1, sample_size1, mean2, std2, sample_size2):
    mean_difference = mean1 - mean2
    sum_of_stds = ((std1 ** 2) / sample_size1) + ((std2 ** 2) / sample_size2)
    tdenominator = np.sqrt(sum_of_stds)
    if tdenominator == 0:
        tvalue = 0; p_value = 0
        if mean1 > mean2:
            tvalue = np.inf
            p_value = 1
        elif mean2 > mean1:
            tvalue = -np.inf
            p_value = 0
        elif mean2 == mean1:
            tvalue = 0
            p_value = 0.5
    else:
        tvalue = mean_difference / tdenominator

        dof_numerator = sum_of_stds ** 2
        dof_sample1 = sample_size1 - 1
        dof_sample2 = sample_size2 - 1
        dof_denominator = ((std1 ** 4) / (dof_sample1 * (sample_size2 ** 2))) + \
                          ((std2 ** 4) / (dof_sample2 * (sample_size2 ** 2)))
        dof = dof_numerator / dof_denominator

        t_dist = t(df=dof)  # from scipy.stats
        p_value = t_dist.pdf(tvalue)
    return tvalue, p_value


#################################
# Functions for loading results #
#################################
class ParameterCombinationSummary:
    """
    Loads the results from all the runs of one particular parameter combination
    """
    def __init__(self, param_comb_path, param_comb_name, parameter_names, performance_measure_name='reward_per_step',
                 load_summary=True, summary_size=500, summary_function=np.sum, save_summary=True):
        """
        :param param_comb_path: path to the directory containing all the runs. The directory has the form:
                                ['agent_1', 'agent_2', ..., 'agent_{sample_size}']
        :param param_comb_name: name of the directory, e.g. LearningRate0.01_BufferSize10000_Freq10
        :param parameter_names: name of the parameters. Each method has a specified set of parameters that is included
                                in the directory name (i.e., in param_comb_name).
        :param performance_measure_name: name of the summary used as a performance measure
        :param load_summary: whether to compute the summary from scratch
        :param save_summary
        :param summary_size: size of the array
        :param summary_function: indicates how to aggregate the data of each run (np.avg or np.sum)
        """
        assert isinstance(param_comb_name, str)
        self.perf_meas = performance_measure_name
        self.summary_size = summary_size  # number of episodes in mc or total steps in catcher
        self.performance_function = summary_function  # either np.avg or np.sum
        # extracting the parameter values
        self.param_comb_name = param_comb_name
        self.param_comb_path = param_comb_path
        self.parameter_values = extract_method_parameter_values(parameter_names, self.param_comb_name)
        # extracting information about each run
        self.sample_size = 0
        self.runs = []
        for agent_name in os.listdir(self.param_comb_path):
            run = self.extract_agent_info(agent_name)
            if run is not None:
                self.runs.append(run)
                self.sample_size += 1

        # checking if a summary is already available:
        self.summary_loaded = self.load_summary(load_summary)

        # summary_loaded = False
        # computing summary of all the runs
        if self.sample_size > 0 and not self.summary_loaded:
            self.performances = np.zeros(self.sample_size, dtype=np.float64)
            for i in range(self.sample_size):
                with open(self.runs[i]['summary_path'], mode='rb') as summary_file:
                    run_summary = np.array(pickle.load(summary_file)[self.perf_meas], dtype=np.float64)
                run_perf = self.performance_function(run_summary)
                self.performances[i] = run_perf
                self.runs[i]['performance_measure'] = run_perf
            self.mean_perf = np.average(self.performances)
            self.stddev_perf = np.std(self.performances, ddof=1)
            _, _, self.me = compute_tdist_confidence_interval(self.mean_perf, self.stddev_perf, 0.05, self.sample_size)

            self.mean_psp = np.zeros(self.summary_size, dtype=np.float64)       # psp = per step performance.
            self.stddev_psp = np.zeros(self.summary_size, dtype=np.float64)     # steps = episodes or training steps
            if save_summary:
                self.save_summary()

    def save_summary(self):
        param_comb_summary = {'sample_size': self.sample_size, 'performances': self.performances,
                              'mean_perf': self.mean_perf, 'stddev_perf': self.stddev_perf,
                              'me': self.me, 'mean_psp': self.mean_psp, 'stddev_psp': self.stddev_psp,
                              'runs': self.runs}
        with open(os.path.join(self.param_comb_path, 'param_comb_summary.p'), mode='wb') as summary_file:
            pickle.dump(param_comb_summary, summary_file)

    def load_summary(self, ls=True):
        if not ls:
            return False
        if 'param_comb_summary.p' not in os.listdir(self.param_comb_path):
            return False
        param_comb_summary_path = os.path.join(self.param_comb_path, 'param_comb_summary.p')
        with open(param_comb_summary_path, mode='rb') as param_comb_summary_file:
            param_comb_summary = pickle.load(param_comb_summary_file)
        if self.sample_size == param_comb_summary['sample_size']:
            self.performances = param_comb_summary['performances']
            self.mean_perf = param_comb_summary['mean_perf']
            self.stddev_perf = param_comb_summary['stddev_perf']
            self.runs = param_comb_summary['runs']
            self.me = param_comb_summary['me']
            self.mean_psp = param_comb_summary['mean_psp']
            self.stddev_psp = param_comb_summary['stddev_psp']
            return True
        else:
            return False

    def extract_agent_info(self, agent_name):
        agent_dir_path = os.path.join(self.param_comb_path, agent_name)
        agent_summary_path = os.path.join(agent_dir_path, 'summary.p')
        agent_weights_path = os.path.join(agent_dir_path, 'final_network_weights.pt')
        if not os.path.exists(agent_summary_path) or not os.path.exists(agent_weights_path):
            if 'agent' in agent_name:
                print('No summary or weight file found for', agent_name, 'and method ' + self.param_comb_name + '.')
            return None
        run = {'summary_path': agent_summary_path, 'weights_path': agent_weights_path}
        return run

    def compute_per_step_statistics(self):
        if self.summary_loaded: return
        for i in range(self.sample_size):
            with open(self.runs[i]['summary_path'], mode='rb') as summary_file:
                run_summary = np.array(pickle.load(summary_file)[self.perf_meas], dtype=np.float64)
            self.mean_psp += run_summary / self.sample_size
        for i in range(self.sample_size):
            with open(self.runs[i]['summary_path'], mode='rb') as summary_file:
                run_summary = np.array(pickle.load(summary_file)[self.perf_meas], dtype=np.float64)
            self.stddev_psp += np.sqrt((run_summary - self.mean_psp)**2 / (self.sample_size - 1))
        self.stddev_psp = np.sqrt(self.stddev_psp)

    def sort_agents(self):
        # sort agents according to the mean performance measure across all the episodes
        sorted_runs = []
        performances = []
        for i in range(self.sample_size):
            assert 'performance_measure' in self.runs[i].keys()
            temp_performance = self.runs[i]['performance_measure']
            idx = 1
            while idx <= len(sorted_runs):
                if temp_performance > performances[-idx]:
                    insert_idx = len(sorted_runs) - idx + 1
                    performances.insert(insert_idx, temp_performance)
                    sorted_runs.insert(insert_idx, self.runs[i])
                    break
                else:
                    idx += 1
            if idx > len(sorted_runs):
                performances.insert(0, temp_performance)
                sorted_runs.insert(0, self.runs[i])
        self.runs = sorted_runs

    def print_summary(self, round_dec=2, sep=''):
        if self.sample_size <= 0:
            return
        print("Parameter combination name:", self.param_comb_name)
        print("The performance measure is:", self.perf_meas)
        print("\tSample size:", self.sample_size)
        print("\tSample average:", np.round(self.mean_perf, round_dec))
        print("\tSample standard deviation:", np.round(self.stddev_perf, round_dec))
        print("\tMargin of error:", np.round(self.me, round_dec))
        uci = np.round(self.mean_perf + self.me, round_dec)
        lci = np.round(self.mean_perf - self.me, round_dec)
        print("\tLower and upper 95% confidence interval bounds:", "(" + str(lci) + ", " + str(uci) + ")")
        if sep != "":
            print(sep)

    def write_summary(self, path, round_dec=2, extra_summary_lines=('',)):
        assert isinstance(extra_summary_lines, tuple)
        if self.sample_size <= 0:
            return
        with open(path, mode='w') as summary_file:
            summary_file.write(
                '#-------------------------------------- Method Summary --------------------------------------#\n')
            summary_file.write("Parameter combination name: " + str(self.param_comb_name) + '\n')
            summary_file.write("\tSample size: " + str(self.sample_size) + '\n')
            summary_file.write("\tSample average: " + str(np.round(self.mean_perf, round_dec)) + '\n')
            summary_file.write("\tSample standard deviation: " + str(np.round(self.stddev_perf, round_dec)) + '\n')
            summary_file.write("\tMargin of error: " + str(np.round(self.me, round_dec)) + '\n')
            uci = np.round(self.mean_perf + self.me, round_dec)
            lci = np.round(self.mean_perf - self.me, round_dec)
            summary_file.write("\tLower and Upper 95% C.I. bounds: (" + str(lci) + ", " + str(uci) + ")" + '\n')
            for extra_line in extra_summary_lines:
                if extra_line != '':
                    summary_file.write('\t' + extra_line + '\n')
            summary_file.write(
                '#--------------------------------------------------------------------------------------------#')


class MethodResults:
    """
    Stores a list of ParameterCombinationResults and finds the best parameter combination according to the 95% C.I.
    """
    def __init__(self, method_name):
        """
        :param method_name: e.g. experience_replay, sigmoid_weighted_units
        """
        self.method_name = method_name
        self.top_param_comb = []
        self.all_param_comb = []

    def append(self, param_comb_results):
        # Appends parameter combination results to the all_param_comb list.
        # Additionally, it compares the new item to the parameter combinations results in top_param_comb and updates
        # the list according to the 95% CI's.
        assert isinstance(param_comb_results, ParameterCombinationSummary)
        if param_comb_results.sample_size <= 0: return
        self.all_param_comb.append(param_comb_results)
        append = True
        popped = 0
        for i in range(len(self.top_param_comb)):
            idx = i-popped
            if len(self.top_param_comb) != 0:
                assert isinstance(self.top_param_comb[idx], ParameterCombinationSummary)
                top_mean = self.top_param_comb[idx].mean_perf
                top_me = self.top_param_comb[idx].me
                top_uci, top_lci = (top_mean + top_me, top_mean - top_me)
                candidate_mean = param_comb_results.mean_perf
                candidate_me = param_comb_results.me
                candidate_uci, candidate_lci = (candidate_mean + candidate_me, candidate_mean - candidate_me)
                if candidate_uci < top_lci:
                    append = False
                    break   # performance is statistically significantly lower
                if candidate_lci > top_uci:
                    self.top_param_comb.pop(idx)  # performance is statistically significantly higher
                    popped += 1
                if (top_lci <= candidate_uci <= top_uci) or (top_lci <= candidate_lci <= top_uci):
                    pass    # performance is not statistically significant
        if append:
            self.top_param_comb.append(param_comb_results)

    def refine_top_results(self, cutoff=0.05):
        # Goes through the top results and compares them using a Welch's test. If any particular one dominates over
        # another, then the dominated one is removed from the top results.
        popped = 0
        for i in range(len(self.top_param_comb)):
            idx = i - popped
            current_result = self.top_param_comb[idx]
            for j in range(i + 1, len(self.top_param_comb)):
                comparison_idx = j - popped
                comparison_result = self.top_param_comb[comparison_idx]
                mean_diff, tval, pvalue = compare_sample_average(method1_summary=current_result,
                                                                 method2_summary=comparison_result,
                                                                 roundto=10, verbose=False)
                if pvalue <= cutoff:
                    if mean_diff < 0:
                        self.top_param_comb.pop(idx)
                    else:
                        self.top_param_comb.pop(comparison_idx)
                    popped += 1

    def print_top_results(self, print_bash_variables=False):
        for idx, results in enumerate(self.top_param_comb):
            assert isinstance(results, ParameterCombinationSummary)
            print('#### Result number (in no particular order):', idx+1, '####')
            print('#----------------------------------------------------------------------#')
            results.print_summary(2)
            if print_bash_variables:
                print('Bash variables format:')
                for name, value in results.parameter_values.items():
                    print('\t' + PARAMETER_NAME_DICT[name] + '=' + str(value))
            print('#----------------------------------------------------------------------#\n')

    def print_best_param_comb(self):
        print('#########################################################################################')
        print('The parameter combination from the top results with the highest lower confidence bound is: ')
        best_param_comb_idx = None
        best_param_comb_lbci = -np.inf
        for idx, results in enumerate(self.top_param_comb):
            if best_param_comb_idx is None:
                best_param_comb_idx = idx
                best_param_comb_lbci = results.mean_perf - results.me
            else:
                results_lbci = results.mean_perf - results.me
                if best_param_comb_lbci < results_lbci:
                    best_param_comb_idx = idx
                    best_param_comb_lbci = results.mean_perf - results.me
        for k, v in self.top_param_comb[best_param_comb_idx].parameter_values.items():
            print('\t' + k + ':', v)
        print('#########################################################################################')

    def print_all_results(self):
        for results in self.all_param_comb:
            assert isinstance(results, ParameterCombinationSummary)
            results.print_summary(sep='\n')


def compare_sample_average(method1_summary, method2_summary, roundto=3, verbose=True):
    assert isinstance(method1_summary, ParameterCombinationSummary)
    assert isinstance(method2_summary, ParameterCombinationSummary)

    method1_mean = method1_summary.mean_perf
    method1_sample_stddev = method1_summary.stddev_perf
    method1_sample_size = method1_summary.sample_size

    method2_mean = method2_summary.mean_perf
    method2_sample_stddev = method2_summary.stddev_perf
    method2_sample_size = method2_summary.sample_size

    tval, pval = compute_welchs_test(mean1=method1_mean, std1=method1_sample_stddev, sample_size1=method1_sample_size,
                                     mean2=method2_mean, std2=method2_sample_stddev, sample_size2=method2_sample_size)

    mean_diff = method1_mean - method2_mean
    if verbose:
        print("The difference in means is:", np.round(mean_diff, roundto))
        print("The t-value of the Welch's Test is:", np.round(tval, roundto))
        print("The p-value is:", np.round(pval, roundto))

    return mean_diff, tval, pval


def parse_method_parameters(parameter_names, parameter_values):
    # creates the parameter combination name for given parameter names and parameter values
    # example: parameter_names=['LearningRate', 'BufferSize', 'Freq'], parameter_values=[0.001, 20000, 10]
    #           returns: "LearningRate0.001_BufferSize20000_Freq10"
    assert len(parameter_names) == len(parameter_values)
    parameter_combination_name = ''
    for i in range(len(parameter_values) - 1):
        parameter_combination_name += parameter_names[i] + str(parameter_values[i]) + "_"
    parameter_combination_name += parameter_names[-1] + str(parameter_values[-1])
    return parameter_combination_name


def extract_method_parameter_values(parameter_names, param_comb_name):
    # extract the parameter values from the parameter_comb_name. This would be the opposite operation of
    # parse_method_parameters.
    # example: parameter_names=['LearningRate', 'Buffer_size', 'Freq'],
    #                    param_comb_name="LearningRate0.001_BufferSize20000_Freq10"
    #           returns: {'LearningRate': 0.001, 'BufferSize': 20000, 'Freq': 10}
    split_method_name = param_comb_name.split('_')
    method_parameters = {}
    for parameter_name, name_value_string in zip(parameter_names, split_method_name):
        parameter_value = name_value_string.split(parameter_name)[1]
        method_parameters[parameter_name] = parameter_value
    return method_parameters


def get_method_results_directory(environment_name, method_name):
    method_results_directory = os.path.join(os.getcwd(), 'Results', environment_name, method_name)
    if not os.path.isdir(method_results_directory):
        raise ValueError("There are no result for that combination of environment and method.")
    return method_results_directory

