# Copyright (c) Microsoft Corporation.
# Licensed under the MIT license.

'''
cuhk_prototype_tuner_v2_advisor.py
'''

import sys
import math
import logging
import json_tricks
from schema import Schema, Optional

from nni import ClassArgsValidator
from nni.protocol import CommandType, send
from nni.msg_dispatcher_base import MsgDispatcherBase
from nni.utils import OptimizeMode, MetricType
from nni.common import multi_phase_enabled
from .moo import MultiObjectiveOptimizerManager

logger = logging.getLogger('CUHKPrototypeTunerV2_Advisor')

_next_parameter_id = 0
_KEY = 'TRIAL_BUDGET'
_epsilon = 1e-6


def create_parameter_id():
    """Create an id

    Returns
    -------
    int
        parameter id
    """
    global _next_parameter_id
    _next_parameter_id += 1
    return _next_parameter_id - 1


def create_bracket_parameter_id(brackets_id, brackets_curr_decay, increased_id=-1):
    """Create a full id for a specific bracket's hyperparameter configuration

    Parameters
    ----------
    brackets_id: int
        brackets id
    brackets_curr_decay: int
        brackets curr decay
    increased_id: int
        increased id
    Returns
    -------
    int
        params id
    """
    if increased_id == -1:
        increased_id = str(create_parameter_id())
    params_id = '_'.join([str(brackets_id),
                          str(brackets_curr_decay),
                          increased_id])
    return params_id


class Bracket:
    """
    A bracket in CUHKPrototypeTunerV2, all the information of a bracket is managed by
    an instance of this class.

    Parameters
    ----------
    s: int
        The current Successive Halving iteration index.
    s_max: int
        total number of Successive Halving iterations
    eta: float
        In each iteration, a complete run of sequential halving is executed. In it,
		after evaluating each configuration on the same subset size, only a fraction of
		1/eta of them 'advances' to the next round.
	max_budget : float
		The largest budget to consider. Needs to be larger than min_budget!
		The budgets will be geometrically distributed
        :math:`a^2 + b^2 = c^2 \\sim \\eta^k` for :math:`k\\in [0, 1, ... , num\\_subsets - 1]`.
    optimize_mode: str
        optimize mode, 'maximize' or 'minimize'
    """
    def __init__(self, s, s_max, eta, max_budget, max_concurrency):
        self.s = s
        self.s_max = s_max
        self.eta = eta
        self.max_budget = max_budget

        self.n = math.ceil((s_max + 1) * eta**s / (s + 1) - _epsilon)
        self.r = max_budget / eta**s
        self.i = 0
        self.hyper_configs = []         # [ {id: params}, {}, ... ]
        self.configs_perf = []          # [ {id: [seq, acc]}, {}, ... ]
        self.num_configs_to_run = []    # [ n, n, n, ... ]
        self.num_finished_configs = []  # [ n, n, n, ... ]
        self.no_more_trial = False
        self.max_concurrency = max_concurrency
        self.is_last_round = False

    def update_max_concurrency(self, new_max_concurrency):
        self.max_concurrency = new_max_concurrency

    def is_completed(self):
        """check whether this bracket has sent out all the hyperparameter configurations"""
        return self.no_more_trial

    def get_n_r(self, upper_bound=None):
        """return the values of n and r for the next round"""
        next_n = math.floor(self.n / self.eta**self.i + _epsilon)

        if self.max_concurrency > next_n: # when extra resources are available
            next_n = self.max_concurrency
            self.is_last_round = True # set current round to be the last round
            return next_n, math.floor(self.r * self.eta**self.s +_epsilon)

        # check resource-budget alignment
        # e.g. max_concurrency is 2 and next_n is 3, then next_n will be added to 4 for better alignment
        if self.max_concurrency - next_n % self.max_concurrency > 0:
            next_n = next_n + self.max_concurrency - next_n % self.max_concurrency
        if upper_bound != None and next_n > upper_bound:
            next_n = upper_bound
        return next_n, math.floor(self.r * self.eta**self.i +_epsilon)

    def increase_i(self):
        """i means the ith round. Increase i by 1"""
        self.i += 1

    def set_config_perf(self, i, parameter_id, seq, value):
        """update trial's latest result with its sequence number, e.g., epoch number or batch number

        Parameters
        ----------
        i: int
            the ith round
        parameter_id: int
            the id of the trial/parameter
        seq: int
            sequence number, e.g., epoch number or batch number
        value: int
            latest result(sorting key) with sequence number seq

        Returns
        -------
        None
        """
        if parameter_id in self.configs_perf[i]:
            if self.configs_perf[i][parameter_id][0] < seq:
                self.configs_perf[i][parameter_id] = [seq, value]
        else:
            self.configs_perf[i][parameter_id] = [seq, value]

    def inform_trial_end(self, i, moo_manager, completed_hyper_configs):
        """If the trial is finished and the corresponding round (i.e., i) has all its trials finished,
        it will choose the top k trials for the next round (i.e., i+1)

        Parameters
        ----------
        i: int
            the ith round

        Returns
        -------
        new trial or None:
            If we have generated new trials after this trial end, we will return a new trial parameters.
            Otherwise, we will return None.
        """
        global _KEY
        self.num_finished_configs[i] += 1
        logger.debug('bracket id: %d, round: %d %d, finished: %d, all: %d',
                     self.s, self.i, i, self.num_finished_configs[i], self.num_configs_to_run[i])
        
        if self.num_finished_configs[i] >= self.num_configs_to_run[i] and self.no_more_trial is False:
            current_r = math.floor(self.r * self.eta**(self.i-1) +_epsilon)
            # choose candidate configs from finished configs to run in the next round
            assert self.i == i + 1
            # finish this bracket
            if self.i > self.s or self.is_last_round:
                current_r = math.floor(self.r * self.eta**self.s +_epsilon)
                self.no_more_trial = True
                for params_id in self.configs_perf[i]:
                    params = self.hyper_configs[i][params_id]
                    for config in completed_hyper_configs[current_r][::-1]:
                        if config["param"] == params:
                            moo_manager.receive_trial_result(parameters=params, value=config["value"], budget=current_r)
                return None

            this_round_perf = self.configs_perf[i] # [ {id: [seq, acc]}, {}, ... ]
            sorted_perf = sorted(
                this_round_perf.items(), key=lambda kv: kv[1][1])
            logger.debug(
                'bracket %s next round %s, sorted hyper configs: %s', self.s, self.i, sorted_perf)
            next_n, next_r = self.get_n_r(upper_bound=len(sorted_perf))

            logger.debug('bracket %s next round %s, next_n=%d, next_r=%d',
                         self.s, self.i, next_n, next_r)
            hyper_configs = dict()
            
            for k in range(next_n):
                params_id = sorted_perf[k][0]
                params = self.hyper_configs[i][params_id]
                for config in completed_hyper_configs[current_r][::-1]:
                    if config["param"] == params:
                        moo_manager.receive_trial_result(parameters=params, value=config["value"], budget=current_r)
                        break

                params[_KEY] = next_r  # modify r
                # generate new id
                increased_id = params_id.split('_')[-1]
                new_id = create_bracket_parameter_id(
                    self.s, self.i, increased_id)
                hyper_configs[new_id] = params
            self._record_hyper_configs(hyper_configs)
            return [[key, value] for key, value in hyper_configs.items()]
        return None

    def get_hyperparameter_configurations(self, num, r, manager):
        """generate num hyperparameter configurations from search space using Bayesian optimization

        Parameters
        ----------
        num: int
            the number of hyperparameter configurations
        r: int
            trial budget
        manager: MultiObjectiveOptimizerManager
            manage multiple optimizers with different budget

        Returns
        -------
        list
            a list of hyperparameter configurations. Format: [[key1, value1], [key2, value2], ...]
        """
        global _KEY
        assert self.i == 0
        hyperparameter_configs = dict()
        for _ in range(num):
            params_id = create_bracket_parameter_id(self.s, self.i)
            params = manager.generate_parameters(r)
            params[_KEY] = r
            hyperparameter_configs[params_id] = params
        self._record_hyper_configs(hyperparameter_configs)
        return [[key, value] for key, value in hyperparameter_configs.items()]

    def _record_hyper_configs(self, hyper_configs):
        """after generating one round of hyperconfigs, this function records the generated hyperconfigs,
        creates a dict to record the performance when those hyperconifgs are running, set the number of finished configs
        in this round to be 0, and increase the round number.

        Parameters
        ----------
        hyper_configs: list
            the generated hyperconfigs
        """
        self.hyper_configs.append(hyper_configs)
        self.configs_perf.append(dict())
        self.num_finished_configs.append(0)
        self.num_configs_to_run.append(len(hyper_configs))
        self.increase_i()

class CUHKPrototypeTunerV2ClassArgsValidator(ClassArgsValidator):
    def validate_class_args(self, **kwargs):
        Schema({
            Optional('eta'): self.range('eta', int, 2, 9999),
            Optional('min_budget'): self.range('min_budget', int, 0, 9999),
            Optional('max_budget'): self.range('max_budget', int, 0, 9999),
            Optional('random_seed'): self.range('random_seed', int, 0, 9999)
        }).validate(kwargs)

class CUHKPrototypeTunerV2(MsgDispatcherBase):
    """
    CUHKPrototypeTunerV2 performs robust and efficient hyperparameter optimization
    at scale by combining the speed of Hyperband searches with the
    guidance and guarantees of convergence of Bayesian Optimization.
    Instead of sampling new configurations at random, CUHKPrototypeTunerV2 uses
    kernel density estimators to select promising candidates.

    Parameters
    ----------
    min_budget: float
        The smallest budget to consider. Needs to be positive!
    max_budget: float
        The largest budget to consider. Needs to be larger than min_budget!
        The budgets will be geometrically distributed
        :math:`a^2 + b^2 = c^2 \\sim \\eta^k` for :math:`k\\in [0, 1, ... , num\\_subsets - 1]`.
    eta: int
        In each iteration, a complete run of sequential halving is executed. In it,
        after evaluating each configuration on the same subset size, only a fraction of
        1/eta of them 'advances' to the next round.
        Must be greater or equal to 2.
    random_seed: int
    """

    def __init__(self,
                 min_budget=1,
                 max_budget=3,
                 eta=3,
                 random_seed=0):
        super(CUHKPrototypeTunerV2, self).__init__()
        self.min_budget = min_budget
        self.max_budget = max_budget
        self.eta = eta
        self.random_seed = random_seed

        # all the configs waiting for run
        self.generated_hyper_configs = []
        # all the completed configs
        self.completed_hyper_configs = dict()

        self.s_max = math.floor(
            math.log(self.max_budget / self.min_budget, self.eta) + _epsilon)
        # current bracket(s) number
        self.curr_s = self.s_max
        # In this case, tuner increases self.credit to issue a trial config sometime later.
        self.credit = 0
        self.brackets = dict()
        self.search_space = None
        # [key, value] = [parameter_id, parameter]
        self.parameters = dict()

        # MultiObjectiveOptimizerManager
        self.moo_manager = None

        # record the latest parameter_id of the trial job trial_job_id.
        # if there is no running parameter_id, self.job_id_para_id_map[trial_job_id] == None
        # new trial job is added to this dict and finished trial job is removed from it.
        self.job_id_para_id_map = dict()
        # record the unsatisfied parameter request from trial jobs
        self.unsatisfied_jobs = []
        self.max_concurrency = 1

    def handle_initialize(self, data):
        """Initialize Tuner, including creating Bayesian optimization-based parametric models
        and search space formations

        Parameters
        ----------
        data: search space
            search space of this experiment

        Raises
        ------
        ValueError
            Error: Search space is None
        """
        logger.info('start to handle_initialize')
        self.search_space = data
        self.moo_manager = MultiObjectiveOptimizerManager(random_seed=self.random_seed)
        assert isinstance(self.search_space, dict)
        try:
            assert all(self.search_space[var]["_type"] == "choice" for var in self.search_space)
        except KeyError as ke:
            raise KeyError(f'Error: Search space missed a key:{ke}')
        except AssertionError as ae:
            raise AssertionError(f'Some types in search space are not "choice".')
        self.moo_manager.update_search_space(self.search_space)

        # generate first brackets
        self.generate_new_bracket()
        send(CommandType.Initialized, '')

    def generate_new_bracket(self):
        """generate a new bracket"""
        logger.debug(
            'start to create a new SuccessiveHalving iteration, self.curr_s=%d', self.curr_s)
        if self.curr_s < 0:
            logger.info("s < 0, Finish this round of Hyperband in CUHKPrototypeTunerV2. Generate new round")
            self.curr_s = self.s_max
        self.brackets[self.curr_s] = Bracket(
            s=self.curr_s, s_max=self.s_max, eta=self.eta, max_budget=self.max_budget, max_concurrency=self.max_concurrency
        )
        next_n, next_r = self.brackets[self.curr_s].get_n_r()
        logger.debug(
            'new SuccessiveHalving iteration, next_n=%d, next_r=%d', next_n, next_r)

        generated_hyper_configs = self.brackets[self.curr_s].get_hyperparameter_configurations(
            next_n, next_r, self.moo_manager)
        self.generated_hyper_configs = generated_hyper_configs

    def handle_request_trial_jobs(self, data):
        """receive the number of request and generate trials

        Parameters
        ----------
        data: int
            number of trial jobs that nni manager ask to generate
        """
        if self.max_concurrency < data:
            self.max_concurrency = data
            self.brackets[self.curr_s].update_max_concurrency(self.max_concurrency)
        # Receive new request
        self.credit += data
        for _ in range(self.credit):
            self._request_one_trial_job()

    def _get_one_trial_job(self):
        """get one trial job, i.e., one hyperparameter configuration.

        If this function is called, Command will be sent by CUHKPrototypeTunerV2:
        a. If there is a parameter need to run, will return "NewTrialJob" with a dict:
        {
            'parameter_id': id of new hyperparameter
            'parameter_source': 'algorithm'
            'parameters': value of new hyperparameter
        }
        b. If CUHKPrototypeTunerV2 don't have parameter waiting, will return "NoMoreTrialJobs" with
        {
            'parameter_id': '-1_0_0',
            'parameter_source': 'algorithm',
            'parameters': ''
        }
        """
        if not self.generated_hyper_configs:
            ret = {
                'parameter_id': '-1_0_0',
                'parameter_source': 'algorithm',
                'parameters': ''
            }
            send(CommandType.NoMoreTrialJobs, json_tricks.dumps(ret))
            return None
        assert self.generated_hyper_configs
        params = self.generated_hyper_configs.pop(0)
        ret = {
            'parameter_id': params[0],
            'parameter_source': 'algorithm',
            'parameters': params[1]
        }
        self.parameters[params[0]] = params[1]
        return ret

    def _request_one_trial_job(self):
        """get one trial job, i.e., one hyperparameter configuration.

        If this function is called, Command will be sent by CUHKPrototypeTunerV2:
        a. If there is a parameter need to run, will return "NewTrialJob" with a dict:
        {
            'parameter_id': id of new hyperparameter
            'parameter_source': 'algorithm'
            'parameters': value of new hyperparameter
        }
        b. If CUHKPrototypeTunerV2 don't have parameter waiting, will return "NoMoreTrialJobs" with
        {
            'parameter_id': '-1_0_0',
            'parameter_source': 'algorithm',
            'parameters': ''
        }
        """
        ret = self._get_one_trial_job()
        if ret is not None:
            send(CommandType.NewTrialJob, json_tricks.dumps(ret))
            self.credit -= 1

    def handle_trial_end(self, data):
        """receive the information of trial end and generate next configuration.

        Parameters
        ----------
        data: dict()
            it has three keys: trial_job_id, event, hyper_params
            trial_job_id: the id generated by training service
            event: the job's state
            hyper_params: the hyperparameters (a string) generated and returned by tuner
        """
        logger.debug('Tuner handle trial end, result is %s', data)
        hyper_params = json_tricks.loads(data['hyper_params'])
        self._handle_trial_end(hyper_params['parameter_id'])
        if data['trial_job_id'] in self.job_id_para_id_map:
            del self.job_id_para_id_map[data['trial_job_id']]

    def _send_new_trial(self):
        while self.unsatisfied_jobs:
            ret = self._get_one_trial_job()
            if ret is None:
                break
            one_unsatisfied = self.unsatisfied_jobs.pop(0)
            ret['trial_job_id'] = one_unsatisfied['trial_job_id']
            ret['parameter_index'] = one_unsatisfied['parameter_index']
            # update parameter_id in self.job_id_para_id_map
            self.job_id_para_id_map[ret['trial_job_id']] = ret['parameter_id']
            send(CommandType.SendTrialJobParameter, json_tricks.dumps(ret))
        for _ in range(self.credit):
            self._request_one_trial_job()

    def _handle_trial_end(self, parameter_id):
        s, i, _ = parameter_id.split('_')
        hyper_configs = self.brackets[int(s)].inform_trial_end(int(i), self.moo_manager, self.completed_hyper_configs)

        if hyper_configs is not None:
            logger.debug(
                'bracket %s next round %s, hyper_configs: %s', s, i, hyper_configs)
            self.generated_hyper_configs = self.generated_hyper_configs + hyper_configs
        # Finish this bracket and generate a new bracket
        else:
            if self.brackets[int(s)].no_more_trial:
                self.curr_s -= 1
                self.generate_new_bracket()
        self._send_new_trial()

    def handle_report_metric_data(self, data):
        """receive the metric data and update Bayesian optimization with final result

        Parameters
        ----------
        data:
            it is an object which has keys 'parameter_id', 'value', 'trial_job_id', 'type', 'sequence'.

        Raises
        ------
        ValueError
            Data type not supported
        """
        logger.debug('handle report metric data = %s', data)
        if 'value' in data:
            data['value'] = json_tricks.loads(data['value'])
        if data['type'] == MetricType.REQUEST_PARAMETER:
            assert multi_phase_enabled()
            assert data['trial_job_id'] is not None
            assert data['parameter_index'] is not None
            assert data['trial_job_id'] in self.job_id_para_id_map
            self._handle_trial_end(self.job_id_para_id_map[data['trial_job_id']])
            ret = self._get_one_trial_job()
            if ret is None:
                self.unsatisfied_jobs.append({'trial_job_id': data['trial_job_id'], 'parameter_index': data['parameter_index']})
            else:
                ret['trial_job_id'] = data['trial_job_id']
                ret['parameter_index'] = data['parameter_index']
                # update parameter_id in self.job_id_para_id_map
                self.job_id_para_id_map[data['trial_job_id']] = ret['parameter_id']
                send(CommandType.SendTrialJobParameter, json_tricks.dumps(ret))
        else:
            assert 'value' in data
            value = self.extract_scalar_value(data['value'], extract_key='default', opposite_key='maximize')
            assert 'parameter_id' in data
            s, i, _ = data['parameter_id'].split('_')
            logger.debug('bracket id = %s, metrics value = %s, type = %s', s, value, data['type'])
            s = int(s)

            # add <trial_job_id, parameter_id> to self.job_id_para_id_map here,
            # because when the first parameter_id is created, trial_job_id is not known yet.
            if data['trial_job_id'] in self.job_id_para_id_map:
                assert self.job_id_para_id_map[data['trial_job_id']] == data['parameter_id']
            else:
                self.job_id_para_id_map[data['trial_job_id']] = data['parameter_id']

            assert 'type' in data
            if data['type'] == MetricType.FINAL:
                # and PERIODICAL metric are independent, thus, not comparable.
                assert 'sequence' in data
                self.brackets[s].set_config_perf(
                    int(i), data['parameter_id'], sys.maxsize, value)
                
                _parameters = self.parameters[data['parameter_id']]
                budget = _parameters.pop(_KEY)

                if budget not in self.completed_hyper_configs:
                    self.completed_hyper_configs[budget] = list()
                self.completed_hyper_configs[budget].append({"param":_parameters, "value":data['value']})

            elif data['type'] == MetricType.PERIODICAL:
                self.brackets[s].set_config_perf(
                    int(i), data['parameter_id'], data['sequence'], value)
            else:
                raise ValueError(
                    'Data type not supported: {}'.format(data['type']))

    def handle_add_customized_trial(self, data):
        pass

    def handle_import_data(self, data):
        """Import additional data for tuning

        Parameters
        ----------
        data:
            a list of dictionaries, each of which has at least two keys, 'parameter' and 'value'

        Raises
        ------
        AssertionError
            data doesn't have required key 'parameter' and 'value'
        """
        for entry in data:
            entry['value'] = json_tricks.loads(entry['value'])
        _completed_num = 0
        for trial_info in data:
            logger.info("Importing data, current processing progress %s / %s", _completed_num, len(data))
            _completed_num += 1
            assert "parameter" in trial_info
            _params = trial_info["parameter"]
            assert "value" in trial_info
            _value = trial_info['value']
            if not _value:
                logger.info("Useless trial data, value is %s, skip this trial data.", _value)
                continue
            budget_exist_flag = False
            barely_params = dict()
            for keys in _params:
                if keys == _KEY:
                    _budget = _params[keys]
                    budget_exist_flag = True
                else:
                    barely_params[keys] = _params[keys]
            if not budget_exist_flag:
                _budget = self.max_budget
                logger.info("Set \"TRIAL_BUDGET\" value to %s (max budget)", self.max_budget)
            self.moo_manager.receive_trial_result(parameters=barely_params, value=trial_info['value'], budget=_budget)
        logger.info("Successfully import tuning data to CUHKPrototypeTunerV2.")
    
    def extract_scalar_value(self, raw_data, extract_key='default', opposite_key='maximize'):
        assert isinstance(raw_data, dict)
        assert extract_key in raw_data
        if opposite_key in raw_data and \
            (isinstance(raw_data[opposite_key], str) and raw_data[opposite_key] == 'default' or \
            isinstance(raw_data[opposite_key], list) and 'default' in raw_data[opposite_key]):
            return -raw_data[extract_key]
        else:
            return raw_data[extract_key]
