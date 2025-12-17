import argparse
import json
import os
import sys
import concurrent
import threading
import yaml
from tqdm import tqdm
from typing import Dict, Optional
import optuna
from optuna.storages import JournalStorage
from optuna.storages.journal import JournalFileBackend
# from multiprocessing import Pool

from postprocessing_utils import BLIS_TRAINING_FILEPATH, BLIS_REQGEN_CONFIG_FOLDER, ALPHA_METRICS_FILENAME, BETA_METRICS_FILENAME
from postprocessing_utils import run_go_binary

GO_BINARY_NAME = "simulation_worker"
GO_BINARY_PATH = os.path.join(os.path.dirname(
    os.path.abspath(__file__)), GO_BINARY_NAME)

NUM_TPE_ITERS = 500
MAX_NUM_PROCESSES = 20
    
class InferenceSimOptimizer:
    """
    A class for black box optimization of inference simulation parameters.
    
    This class provides an easy interface to configure, run, and evaluate
    optimization experiments for inference simulation models.
    """
    
    def __init__(
        self,
        alpha0: float,
        alpha1: float,
        reqgen_config_folder: str,
        training_data: Dict = None,
        pbounds: Optional[Dict] = None,
        scaling: Optional[Dict] = None,
        seed: int = 42
    ):
        """
        Initialize the InferenceSimOptimizer.
        
        Args:
            pbounds: Parameter bounds for optimization variables
            scaling: Scaling factors for parameters
        """
        # Set default parameter bounds, scaling
        self.pbounds = pbounds or {
            'beta0': (1e-4, 1e4),
            'beta1': (1e-4, 1e4),
            'beta2': (1e-4, 1e4),
        }
        self.scaling = scaling or {
            'beta0': 1,
            'beta1': 1,
            'beta2': 1
        }
        self.seed = seed
        self.reqgen_config_folder = reqgen_config_folder
        self.training_data = training_data

        self.metrics_lock = None

        self.alpha0 = alpha0
        self.alpha1 = alpha1
        
    def cost_function(self, vllm_metrics, sim_metrics):
        metric_names = ["e2e_mean_ms", "e2e_p90_ms", "ttft_mean_ms", "ttft_p90_ms"]
        total_mape = 0
        for _, metric in enumerate(metric_names):
            mape = abs(sim_metrics[metric] - vllm_metrics[metric])/vllm_metrics[metric] * 100
            total_mape += mape
        return total_mape
    
    def per_thread_cost(self, request_rate, beta_coeffs):
        """
        Run simulator per experiment thread and obtain simulator results. 
        Compare against vllm ground truth metrics and return cost per experiment
        """
        # get vllm ground truth
        vllm_metrics = [x for x in self.training_data["benchmarks"] if x["rps"] == request_rate][0]
        
        # get sim metrics
        reqgen_config_file = os.path.join(self.reqgen_config_folder, 
                                          f"requestgenconfig_RPS={round(request_rate, 3)}.yaml")
        args = {
            "max-num-running-reqs": self.training_data["vllm_config"]["max_num_seqs"], 
            "total-kv-blocks": self.training_data["vllm_config"]["total_kv_blocks"],
            "model": self.training_data["vllm_config"]["model"],
            "max-num-scheduled-tokens": self.training_data["vllm_config"]["max_num_batched_tokens"], 
            "block-size-in-tokens": 16,
            "horizon": "922337203685477580", # Golang int64 max value
            "beta-coeffs": ','.join(beta_coeffs),
            "long-prefill-token-threshold": 0,
            "alpha-coeffs": f"{self.alpha0},{self.alpha1},0",
            "log": "error"
        }
        args_list = ["run"]
        for key in args:
            args_list.extend([f"--{key}", str(args[key])])
        with open(reqgen_config_file, "r+") as f:
            workload_config = yaml.safe_load(f)
        for config in workload_config["data"]:
            config_field = f"--{config.replace("_", "-")}"
            args_list.extend([config_field, str(workload_config["data"][config])])
        args_list.extend(["--rate", str(workload_config["rate"]["rate"])])
        args_list.extend(["--max-prompts", str(workload_config["rate"]["max-requests"])])
        sim_metrics = run_go_binary(args_list, GO_BINARY_PATH, request_rate, self.metrics_lock)
        if not sim_metrics:
            return None
        cost = self.cost_function(vllm_metrics, sim_metrics)
        return cost
    
    def run_task_from_tuple(self, args_tuple):
        return self.per_thread_cost(*args_tuple)

    def multiexp_obj(self, trial: optuna.trial.Trial):
        total_cost = 0.0
        tasks = []
        self.metrics_lock = threading.Lock()
        beta0 = trial.suggest_float('beta0', *self.pbounds['beta0'])
        beta1 = trial.suggest_float('beta1', *self.pbounds['beta1'])
        beta2 = trial.suggest_float('beta2', *self.pbounds['beta2'])
        beta_coeffs = [beta0, beta1, beta2]
        beta_coeffs = list(map(str, beta_coeffs))

        for exp in self.training_data["benchmarks"]:
            rps = exp["rps"]
            tasks.append((rps, beta_coeffs))

        with concurrent.futures.ThreadPoolExecutor() as executor:
            all_costs = executor.map(self.run_task_from_tuple, tasks)
            valid_costs = [cost for cost in all_costs if cost is not None]
            total_cost = sum(valid_costs)
        return total_cost


    def optimize_multiexp(self, n_trials: int = 50):
        """
        Run TPE log sampler to find best scaling parameters.
        
        Args:
            n_trials: Number of optimization trials to run, defaults to 50
        """
        print("=" * 60)
        print("STARTING OPTIMIZATION")
        print("=" * 60)
        print(f"Number of Trials: {n_trials}")
        print("=" * 60)
        
        sampler_obj = optuna.samplers.TPESampler(seed=self.seed)
        
        self.study = optuna.create_study(sampler=sampler_obj, 
                                         direction="minimize", 
                                         load_if_exists=False,
                                         storage=None)  
        self.study.optimize(self.multiexp_obj, n_trials=n_trials)    
        self.train_score = self.study.best_value
        
        print("=" * 60)
        print("OPTIMIZATION COMPLETED")
        print("=" * 60)
        print(f"Best Training Error: {self.study.best_value}")
        print(f"Best Parameters:")
        for param in self.study.best_params:
            print(f"Scaling for {param}: {self.study.best_params[param]}")
        print("=" * 60)

    def multitrial_obj(self, trial: optuna.trial.Trial):
        print(f"Running trial {trial.number=} in process {os.getpid()}")
        tasks = []
        beta0 = trial.suggest_float('beta0', *self.pbounds['beta0'])
        beta1 = trial.suggest_float('beta1', *self.pbounds['beta1'])
        beta2 = trial.suggest_float('beta2', *self.pbounds['beta2'])
        beta_coeffs = [beta0, beta1, beta2]
        beta_coeffs = list(map(str, beta_coeffs))

        for exp in self.training_data["benchmarks"]:
            rps = exp["rps"]
            tasks.append((rps, beta_coeffs))
        
        all_costs = []
        for task in tqdm(tasks):
            all_costs.append(self.per_thread_cost(*task))
        valid_costs = [cost for cost in all_costs if cost is not None]
        return sum(valid_costs)
       
    def optimize_multitrial(self):
        """
        Run optimization study.
        
        Args:
            n_trials: Number of optimization trials to run, defaults to 100
            sampler: Sampler to use for optimization, defaults to "TPESampler"
        """
        sampler_obj = optuna.samplers.GridSampler(self.search_space, seed=self.seed)
        self.study = optuna.create_study(sampler=sampler_obj, 
                                         direction="minimize",
                                         study_name="journal_storage_multitrial",
                                         storage=JournalStorage(JournalFileBackend(file_path="./journal.log")),
                                         load_if_exists=True)  
        self.study.optimize(self.multitrial_obj, n_trials=1)    
        self.train_score = self.study.best_value
        
        print("=" * 60)
        print("OPTIMIZATION COMPLETED")
        print("=" * 60)
        print(f"Best Training Error: {self.study.best_value}")
        print(f"Best Parameters:")
        for param in self.study.best_params:
            print(f"{param}: {self.study.best_params[param] * self.scaling[param]}")
        print("=" * 60)

    def get_best_trial(self):
        """
        Get the best trial from the current study.
        
        Returns:
            Dictionary of best trial
        """
        if self.study is None:
            raise ValueError("No study available. Run optimize() first.")
        
        best_trial = {
            "best_loss": self.study.best_trial.value,
            "best_params": self.study.best_trial.params
        }
        print(f"Best trial: {best_trial}")
        return best_trial

    def visualize_study(self):
        """Visualize the optimization study results."""
        if self.study is None:
            raise ValueError("No study available. Run optimize() first.")
        
        import optuna.visualization
        return optuna.visualization.plot_optimization_history(self.study)

def with_inp(args):
    i, optimizer = args
    optimizer.optimize_multitrial()

def train_beta_model(results_path, model_path):
    # read pretrained alpha coeffs
    alpha_metrics_filename = os.path.join(model_path, ALPHA_METRICS_FILENAME)
    try:
        with open(alpha_metrics_filename, 'r') as f:
            alpha_metrics = json.load(f)
            alpha0, alpha1 = alpha_metrics["coeffs"]
    except:
        print("Could not load alpha metrics file. Could not fetch alpha coeffs.")
        sys.exit()

    # read heuristic bounds & benchmark metrics for betas
    training_data_filename = os.path.join(results_path, BLIS_TRAINING_FILEPATH)
    try:
        with open(training_data_filename, 'r') as f:
            training_data = json.load(f)
    except:
        print("Could not load BLIS training data.")
        sys.exit()

    heuristics_bounds = {
        "beta0": (0, 2 * training_data["bounds"]["beta0"]),
        "beta1": (0, 2 * training_data["bounds"]["beta1"]),
        "beta2": (0, 2 * training_data["bounds"]["beta2"])
    }

    reqgen_config_folder = os.path.join(results_path, BLIS_REQGEN_CONFIG_FOLDER)

    # Initialize optimizer
    optimizer = InferenceSimOptimizer(
        alpha0=alpha0,
        alpha1=alpha1,
        reqgen_config_folder=reqgen_config_folder,
        training_data=training_data,
        pbounds=heuristics_bounds
    )

    # Sequential TPE sampling
    optimizer.optimize_multiexp(n_trials=NUM_TPE_ITERS)
    best_params = optimizer.get_best_trial()

    # Parallel Grid sampling
    # optimizer.search_space = {
    #     'beta0': list(np.arange(0, heuristics_bounds["beta0"][1], heuristics_bounds["beta0"][1]/20)),
    #     'beta1': list(np.arange(0, heuristics_bounds["beta1"][1], heuristics_bounds["beta1"][1]/20)),
    #     'beta2': list(np.arange(0, heuristics_bounds["beta2"][1], heuristics_bounds["beta2"][1]/20)),
    # }
    # num_GS_iters = len(optimizer.search_space["beta0"]) * len(optimizer.search_space["beta1"]) * len(optimizer.search_space["beta2"])
    
    # with Pool(processes=MAX_NUM_PROCESSES) as pool: 
    #     pool.map(with_inp, ((i, optimizer) for i in range(num_GS_iters)))

    # best_params = optimizer.get_best_trial()

    # save best optimizer parameters
    beta_metrics_filename = os.path.join(model_path, BETA_METRICS_FILENAME)
    with open(beta_metrics_filename, 'w+') as file:
        json.dump(best_params, file, indent=4)
    print(f"Beta model metrics and coefficients saved to {beta_metrics_filename}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Read and parse traces JSON file.")
    parser.add_argument("--model_path", 
                        help="Folder path to save trained beta models")
    parser.add_argument("--train_results_path",
                            default=".", 
                            help="Location to get training data from.")
    args = parser.parse_args()
    train_beta_model(args.train_results_path, args.model_path)

    