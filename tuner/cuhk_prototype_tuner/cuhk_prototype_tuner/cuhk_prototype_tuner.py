"""
cuhk_prototype_tuner.py
"""

import logging

import random
import numpy as np
from schema import Optional, Schema
from nni.utils import ClassArgsValidator
from nni.tuner import Tuner

from argparse import Namespace
# from dragonfly import load_config
from .dragonfly.utils.option_handler import get_option_specs, load_options
from .dragonfly.gp.cartesian_product_gp import CPGPFitter
from .dragonfly.opt import multiobjective_gpb_acquisitions
from .dragonfly.opt.multiobjective_gp_bandit import multiobjective_gp_bandit_args
from .dragonfly.exd.exd_core import exd_core_args
from .dragonfly.exd.cp_domain_utils import sample_from_cp_domain, load_config
from .dragonfly.opt.gp_bandit import gp_bandit_args

import ConfigSpace as CS
import ConfigSpace.hyperparameters as CSH

logger = logging.getLogger('CUHK_AutoML')
EVAL_ERROR_CODE = 'eval_error_250320181729'

class CUHKPrototypeClassArgsValidator(ClassArgsValidator):
    def validate_class_args(self, **kwargs):
        Schema({
            Optional('random_seed'): self.range('random_seed', int, 0, 99999999),
            Optional('num_init_evals'): self.range('num_init_evals', int, 1, 100),
            Optional('build_new_model_every'): self.range('build_new_model_every', int, 1, 100)
        }).validate(kwargs)


class CUHKPrototypeTuner(Tuner):
    """
    CUHKPrototypeTuner is a tuner enable tuning with multiple objectives.
    """

    def __init__(self, random_seed=None, num_init_evals=2, build_new_model_every=5):
        """
        Parameters
        ----------
        random_seed : int
          fix random seed if it is set
        """
        if random_seed is not None:
            random.seed(random_seed)
            np.random.seed(random_seed)
        self.num_init_evals = num_init_evals
        self.build_new_model_every = build_new_model_every
        self.total_data = dict()
        self.search_space = dict()
        self.domain = None
        self.trial_idx = 0
        self.num_objectives = 0
        self.curr_acq = None
        self.gps = None
        self.options = Namespace()
        self.history = Namespace(
            query_points=[],
            query_vals=[]
        )
        self.get_acq_opt_max_evals = None
        self.acq_opt_method = 'ga'
        self.eval_points_in_progress = []
        self.config_space=CS.ConfigurationSpace(seed=random_seed)

    def update_search_space(self, search_space):
        """
        Generate a MOO-BO model based on dragonfly
        Update search space definition in tuner by search_space in parameters.
        Will called when first setup experiemnt or update search space in WebUI.
        Parameters
        ----------
        search_space : dict

        search_space example:
        OrderedDict([('learning_rate', OrderedDict([('_type', 'choice'), ('_value', [0.5, 0.25, 0.1, 0.05, 0.025, 0.01, 0.005, 0.0025, 0.001, 0.0005, 0.00025, 0.0001, 5e-05, 2.5e-05, 1e-05])])), ('optimizer', OrderedDict([('_type', 'choice'), ('_value', ['adam', 'sgd', 'rmsp'])])), ('batch_size', OrderedDict([('_type', 'choice'), ('_value', [8, 16, 32, 64, 128])])), ('epoch1', OrderedDict([('_type', 'choice'), ('_value', [3, 5, 7, 9, 11, 13, 15, 17, 19, 21, 23, 25, 27])])), ('epoch', OrderedDict([('_type', 'choice'), ('_value', [1])])), ('filter_num', OrderedDict([('_type', 'choice'), ('_value', [8, 16, 24, 32, 40, 48, 56, 64])])), ('kernel_size', OrderedDict([('_type', 'choice'), ('_value', [1, 2, 3, 4, 5])])), ('weight_decay', OrderedDict([('_type', 'choice'), ('_value', [0.08, 0.04, 0.01, 0.008, 0.004, 0.001, 0.0008, 0.0004, 0.0001, 8e-05, 4e-05, 1e-05])])), ('dense_size', OrderedDict([('_type', 'choice'), ('_value', [64, 128, 256, 512, 1024])]))])
        """
        config = self.sp2config(search_space)
        self.domain = config.domain
        options = self._get_user_options()
        opt_options = load_options(self.get_all_cp_moo_gp_bandit_args())
        options = Namespace() if options is None else options
        for attr in vars(options):
            setattr(opt_options, attr, getattr(options, attr))

        opt_options.acq, opt_options.mode, opt_options.mf_strategy = None, None, None 
        self.options = opt_options
        self.search_space = search_space

        for var in self.search_space:
            _type = str(self.search_space[var]["_type"])
            if _type == 'choice':
                self.config_space.add_hyperparameter(CSH.CategoricalHyperparameter(
                    var, choices=self.search_space[var]["_value"]))
            elif _type == 'randint':
                self.config_space.add_hyperparameter(CSH.UniformIntegerHyperparameter(
                    var, lower=self.search_space[var]["_value"][0], upper=self.search_space[var]["_value"][1] - 1))
            elif _type == 'uniform':
                self.config_space.add_hyperparameter(CSH.UniformFloatHyperparameter(
                    var, lower=self.search_space[var]["_value"][0], upper=self.search_space[var]["_value"][1]))
            elif _type == 'quniform':
                self.config_space.add_hyperparameter(CSH.UniformFloatHyperparameter(
                    var, lower=self.search_space[var]["_value"][0], upper=self.search_space[var]["_value"][1],
                    q=self.search_space[var]["_value"][2]))
            elif _type == 'loguniform':
                self.config_space.add_hyperparameter(CSH.UniformFloatHyperparameter(
                    var, lower=self.search_space[var]["_value"][0], upper=self.search_space[var]["_value"][1],
                    log=True))
            elif _type == 'qloguniform':
                self.config_space.add_hyperparameter(CSH.UniformFloatHyperparameter(
                    var, lower=self.search_space[var]["_value"][0], upper=self.search_space[var]["_value"][1],
                    q=self.search_space[var]["_value"][2], log=True))
            elif _type == 'normal':
                self.config_space.add_hyperparameter(CSH.NormalFloatHyperparameter(
                    var, mu=self.search_space[var]["_value"][1], sigma=self.search_space[var]["_value"][2]))
            elif _type == 'qnormal':
                self.config_space.add_hyperparameter(CSH.NormalFloatHyperparameter(
                    var, mu=self.search_space[var]["_value"][1], sigma=self.search_space[var]["_value"][2],
                    q=self.search_space[var]["_value"][3]))
            elif _type == 'lognormal':
                self.config_space.add_hyperparameter(CSH.NormalFloatHyperparameter(
                    var, mu=self.search_space[var]["_value"][1], sigma=self.search_space[var]["_value"][2],
                    log=True))
            elif _type == 'qlognormal':
                self.config_space.add_hyperparameter(CSH.NormalFloatHyperparameter(
                    var, mu=self.search_space[var]["_value"][1], sigma=self.search_space[var]["_value"][2],
                    q=search_space[var]["_value"][3], log=True))
            else:
                raise ValueError(
                    'unrecognized type in search_space, type is {}'.format(_type))

    def generate_parameters(self, parameter_id, **kwargs):
        """
        Return a set of trial (hyper-)parameters, as a serializable object.
        Parameters
        ----------
        parameter_id : int
        Returns
        -------
        params : dict
        """
        if len(self.search_space) == 0:
            return []
        if self.trial_idx < self.num_init_evals:           
            # if self.trial_idx == 0 or not hasattr(self, 'init_qinfo'):
            #     ret_dom_pts = sample_from_cp_domain(
            #         cp_domain=self.domain, 
            #         num_samples=self.num_init_evals,
            #         domain_samplers=None,
            #         euclidean_sample_type='latin_hc',
            #         integral_sample_type='latin_hc',
            #         nn_sample_type='latin_hc')
            #     self.init_qinfo = ret_dom_pts
            # return self.qinfo2dict(self.init_qinfo[self.trial_idx]) if self.trial_idx < len(self.init_qinfo) else {}
            return self.config_space.sample_configuration().get_dictionary()
        else:
            # opt/multiobjective_gp_bandit.py: 190
            # _main_loop_pre() or _set_next_gp()
            if not hasattr(self, 'gp_processors') or self.gp_processors is None:
                self._build_new_gps(self.num_objectives)
            self.gps = []
            for gp_processor in self.gp_processors:
                ret = gp_processor.gp_fitter.get_next_gp()
                gp_processor.fit_type = ret[0]
                gp_processor.hp_tune_method = ret[1]
                self.gps.append(ret[2])
            for i, gp_processor in enumerate(self.gp_processors):
                if gp_processor.fit_type in ['sample_hps_with_probs', \
                                            'post_sample_hps_with_probs']:
                    reg_data = self._get_moo_gp_reg_data(i)
                    self.gps[i].set_data(reg_data[0], reg_data[1], build_posterior=True)
            
            # self._asynchronous_run_experiment_routine()
            # opt/multiobjective_gp_bandit.py:405
            if not hasattr(self, 'acq_probs'):
                self.acqs_to_use = ['ucb', 'ts'] # opt/multiobjective_gp_bandit.py:96
                self.acqs_to_use_counter = {key: 0 for key in self.acqs_to_use}
                self.acq_uniform_sampling_prob = 0.05
                self.acq_sampling_weights = {key: 1.0 for key in self.acqs_to_use}
                self.acq_probs = self._get_adaptive_ensemble_acq_probs()
                self.acq_probs = self.acq_probs / self.acq_probs.sum()
                assert len(self.acq_probs) == len(self.acqs_to_use)
            if len(self.eval_points_in_progress) > 0:
                self.eval_points_in_progress.pop()
            qinfo = self._determine_next_query() 
            return self.qinfo2dict(qinfo)

    def receive_trial_result(self, parameter_id, parameters, value, **kwargs):
        """
        Record an observation of the objective function
        Parameters
        ----------
        parameter_id : int
        parameters : dict
        value : dict/float
            if value is dict, it should have "default" key.
            value is final metrics of the trial.
        """
        if self.trial_idx < self.num_init_evals:
            self.trial_idx += 1
        report_value = self.extract(value)
        if self.num_objectives != len(report_value):
            self.num_objectives = len(report_value)
        
        qinfos = self.dict2qinfos(parameters, report_value)
        for qinfo in qinfos:
            if float('inf') in qinfo.val:
                return
        self._update_history(qinfos)
        self._add_data_to_model(qinfos) 
        
    def extract(self, value, non_opposite_key='maximize'):
        """
        Extract from reported value
        Parameters
        ----------
        value : dict
            Invalid Value:
                0.2
                {'default':0.2,'attr0':0.3,'attr1':46,'maximize':'attr2'}
            Valid Value:
                {'default':0.2,'attr0':0.3}
                {'default':0.2,'attr0':0.3,'attr1':46}
                {'default':0.2,'attr0':0.3,'attr1':10,'maximize':'attr0'}
                {'default':0.2,'attr0':0.3,'attr1':23.45,'maximize':['attr0','attr1']}
        non_opposite_key : str
            Specify values in which attribute need to take the opposite value
        
        Return
        ------
        A list containing the values from input dict
        """
        if not isinstance(value, dict):
            raise ValueError(f'{value} is in an unexpected type.')

        for attr in value:
            if not isinstance(value[attr], (float, int)) and attr != non_opposite_key:
                raise ValueError(f'{value} has an attribute {attr} with unexpected type.')

        if non_opposite_key not in value:
            return [value[attr] for attr in value]

        if not isinstance(value[non_opposite_key], (str, list)):
            raise ValueError(f'{non_opposite_key} is in an unexpected type.')

        if isinstance(value[non_opposite_key], str):
            value[non_opposite_key] = [value[non_opposite_key]]

        for x in value[non_opposite_key]:
            if not isinstance(x, str):
                raise ValueError(f'{non_opposite_key} has a value {x} with unexpected type.')
            if x not in value:
                raise ValueError(f'{non_opposite_key} has an unexpected value {x}.')
        
        for attr in value:
            if attr not in value[non_opposite_key] and attr != non_opposite_key:
                value[attr] = -value[attr]
        return [value[attr] for attr in value if attr != non_opposite_key]

    def dict2qinfos(self, parameters, report_value):
        '''
        qinfo example:
        [Namespace(caller_eval_cost=None, curr_acq='ucb', eval_time=28.06532621383667, hp_tune_method='ml', 
        point=[['sgd', True, True, True, False, True, 'global', 'NcclAllReduce'], [0.025, 128, 1, 56, 3, 0.001, 256, 4, 10, 4, 2, 2]], receive_time=128.95513558387756, result_file='./tmp_0902_020400/result_0/result.p', send_time=100.8898093700409, step_idx=3, 
        true_val=[-0.007054062949286567, 0.12229999899864197], val=[-0.007054062949286567, 0.12229999899864197], worker_id=0, working_dir='./tmp_0902_020400/working_0/tmp')]
        '''
        num_para = list()
        non_num_para = list()
        for name in parameters:
            if isinstance(parameters[name], (float, int, np.float64, np.int64)):
                if isinstance(parameters[name], np.int64):
                    parameters[name] = int(parameters[name])
                if isinstance(parameters[name], np.float64):
                    parameters[name] = float(parameters[name])
                num_para.append(parameters[name])
            else:
                if isinstance(parameters[name], np.str_):
                    parameters[name] = str(parameters[name])
                non_num_para.append(parameters[name])
        qinfos = [Namespace(
            point=[non_num_para, num_para],
            val=report_value
        )]
        return qinfos
    
    def _update_history(self, qinfos):
        for qinfo in qinfos:
            self.history.query_points.append(qinfo.point)
            self.history.query_vals.append(qinfo.val)
            self.eval_points_in_progress.append(qinfo.point)

    def sp2config(self,search_space):
        domain_vars = []
        for name in search_space:
            domain_var = {
                'name': name,
                'type': 'discrete_numeric' if isinstance(search_space[name]['_value'][0], (float, int)) else 'discrete',
                'items': search_space[name]['_value']
            }
            domain_vars.append(domain_var)
        config_params = {'domain': domain_vars}
        
        config = load_config(config_params)
        return config
    
    def _get_user_options(self):
        options = load_options([ 
            get_option_specs('init_capital', False, None, 'The capital to be used for initialisation.'),
            get_option_specs('num_init_evals', False, self.num_init_evals, 'The number of evaluations for initialisation.')
        ])
        return options
    
    def get_all_cp_moo_gp_bandit_args(self):
        return multiobjective_gp_bandit_args + exd_core_args + gp_bandit_args

    def _build_new_gps(self, num_objectives):
        # Invoke the GP fitter.
        self.gp_processors = []
        for i in range(num_objectives):
            gp_fitter = self._get_non_mf_gp_fitter(i)
            # Fits gp and adds it to gp_processor
            gp_fitter.fit_gp_for_gp_bandit(self.build_new_model_every)
            gp_processor = Namespace()
            gp_processor.gp_fitter = gp_fitter
            self.gp_processors.append(gp_processor)
    
    def _get_non_mf_gp_fitter(self, gp_idx):
        """ Returns the Non-Multi-fidelity GP Fitter. """
        options = Namespace(**vars(self.options)) # opt/multiobjective_gp_bandit.py: 285
        reg_data = self._get_moo_gp_reg_data(gp_idx)
        return CPGPFitter(reg_data[0], reg_data[1], self.domain, domain_kernel_ordering=['', ''], options=options)
    
    def _get_moo_gp_reg_data(self, obj_ind): # opt/multiobjective_gp_bandit.py: 257
        """ Returns the current data to be added to the GP. """
        # pylint: disable=no-member
        reg_X = self.history.query_points
        reg_Y = self.history.query_vals
        row_idx = [i for i,y in enumerate(reg_Y)]
        return ([reg_X[i] for i in row_idx],
                [reg_Y[i][obj_ind] for i in row_idx])
    
    def _add_data_to_model(self, qinfos):
        """ Add data to self.gp """
        if self.gps is None:
            return
        qinfos = [qinfo for qinfo in qinfos
                if qinfo.val != EVAL_ERROR_CODE and qinfo.val != float('inf')]
        if len(qinfos) == 0:
            return
        new_points = [qinfo.point for qinfo in qinfos]
        new_vals = [qinfo.val for qinfo in qinfos]
        self._add_data_to_gps((new_points, new_vals))

    def _add_data_to_gps(self, new_data):
        """ Adds data to the GP. """
        # Add data to the GP only if we will be repeating with the same GP.
        if hasattr(self, 'gp_processors') and hasattr(self.gp_processors[0], 'fit_type') and \
          self.gp_processors[0].fit_type == 'fitted_gp':
            for i, gp in enumerate(self.gps):
                if self.gp_processors[i].fit_type == 'fitted_gp':
                    vals = [y[i] for y in new_data[1]]
                    gp.add_data_multiple(new_data[0], vals)

    def _get_adaptive_ensemble_acq_probs(self):
        """ Computes the adaptive ensemble acqusitions probs. """
        num_acqs = len(self.acqs_to_use)
        uniform_sampling_probs = self.acq_uniform_sampling_prob * \
                                np.ones((num_acqs,)) / num_acqs
        acq_succ_counter = np.array([self.acq_sampling_weights[key] for
                                    key in self.acqs_to_use])
        acq_use_counter = np.array([self.acqs_to_use_counter[key] for
                                    key in self.acqs_to_use])
        acq_weights = acq_succ_counter / np.sqrt(1 + acq_use_counter)
        acq_norm_weights = acq_weights / acq_weights.sum()
        adaptive_sampling_probs = (1 - self.acq_uniform_sampling_prob) * acq_norm_weights
        ret = uniform_sampling_probs + adaptive_sampling_probs
        return ret / ret.sum()

    def _determine_next_query(self):
        """ Determine the next point for evaluation. """
        
        self.curr_acq = self._get_next_acq() # 'ts' or 'ucb'
        anc_data = self._get_ancillary_data_for_acquisition(self.curr_acq)
        select_pt_func = getattr(multiobjective_gpb_acquisitions.asy, 'tch_'+self.curr_acq) 
        # tch means the default value 'tchebychev' in opt/multiobjective_gp_bandit.py:44
        next_eval_point = select_pt_func(self.gps, anc_data)
        return next_eval_point
    
    def _get_next_acq(self):
        """ Gets the acquisition to use in the current iteration. """
        self.acq_probs = self._get_adaptive_ensemble_acq_probs()
        ret = np.random.choice(self.acqs_to_use, p=self.acq_probs)
        return ret
    
    def _get_ancillary_data_for_acquisition(self, curr_acq):
        """ Returns ancillary data for the acquisitions. """
        self._set_up_cp_acq_opt_ga()
        max_num_acq_opt_evals = self.get_acq_opt_max_evals(self.trial_idx)
        ret = Namespace(
            curr_acq=curr_acq,
            max_evals=max_num_acq_opt_evals,
            t=self.trial_idx,
            domain=self.domain,
            eval_points_in_progress=self.eval_points_in_progress,
            acq_opt_method=self.acq_opt_method,
            handle_parallel=self.options.handle_parallel,
            mf_strategy=self.options.mf_strategy,
            is_mf=False,
            obj_weights=np.abs(np.random.normal(loc=0.0, scale=10, size=(self.num_objectives,))), # opt/multiobjective_gp_bandit.py:78
            reference_point=[-1.0]*self.num_objectives # opt/multiobjective_gp_bandit.py:90
        )
        return ret

    def _set_up_cp_acq_opt_ga(self):
        """ Set up optimisation for acquisition using rand. """
        domain_types = [dom.get_type() for dom in self.domain.list_of_domains]
        if 'neural_network' in domain_types:
            # Because Neural networks can be quite expensive
            self._set_up_cp_acq_opt_with_params(1, 500, 2e4)
        else:
            self._set_up_cp_acq_opt_with_params(1, 1000, 3e4)

    def _set_up_cp_acq_opt_with_params(self, lead_const, min_iters, max_iters):
        """ Sets up optimisation for acquisition using direct. """
        if self.get_acq_opt_max_evals is None:
            dim_factor = lead_const * min(5, self.domain.get_dim())**2
            self.get_acq_opt_max_evals = lambda t: np.clip(dim_factor * np.sqrt(min(t, 1000)),
                                                            min_iters, max_iters)

    def qinfo2dict(self, qinfo):
        params = dict()
        idx0 = idx1 = 0
        for para_name in self.search_space.keys():
            if isinstance(self.search_space[para_name]['_value'][0], (float, int)):
                numerical_idx = 1 if len(qinfo) > 1 else 0
                if isinstance(qinfo[numerical_idx][idx1], np.float64):
                    qinfo[numerical_idx][idx1] = float(qinfo[numerical_idx][idx1])
                if isinstance(qinfo[numerical_idx][idx1], np.int64):
                    qinfo[numerical_idx][idx1] = int(qinfo[numerical_idx][idx1])
                params[para_name] = qinfo[numerical_idx][idx1]
                idx1 += 1
            else:
                if isinstance(qinfo[0][idx0], np.str_):
                    qinfo[0][idx0] = str(qinfo[0][idx0])
                params[para_name] = qinfo[0][idx0]
                idx0 += 1
        return params
