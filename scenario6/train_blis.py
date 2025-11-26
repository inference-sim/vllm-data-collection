import argparse
import os
import yaml
import concurrent
import threading
from tqdm import tqdm
from typing import Dict, Optional
import pandas as pd
import optuna
from optuna.storages import JournalStorage
from optuna.storages.journal import JournalFileBackend
# from multiprocessing import Pool
from filelock import FileLock

from postprocessing_utils import run_go_binary

GO_BINARY_NAME = "simulation_worker"
GO_BINARY_PATH = os.path.join(os.path.dirname(
    os.path.abspath(__file__)), GO_BINARY_NAME)

NUM_TPE_ITERS = 2000
MAX_NUM_PROCESSES = 20
MILLISECONDS_TO_MICROSECONDS_CONVERSION = 1e3
# modify this list to change which metrics contribute to cost fn
METRICS_IN_LOSS = ["ttft_mean", "ttft_p90", "e2e_mean", "e2e_p90"]
    
class InferenceSimOptimizer:
    """
    A class for black box optimization of inference simulation parameters.
    
    This class provides an easy interface to configure, run, and evaluate
    optimization experiments for inference simulation models.
    """
    
    def __init__(
        self,
        training_df: pd.DataFrame = None,
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
            'alpha0': (1e-4, 1e4),
            'alpha1': (1e-4, 1e4),
            'alpha2': (1e-4, 1e4),
            'beta0': (1e-4, 1e4),
            'beta1': (1e-4, 1e4),
            'beta2': (1e-4, 1e4),
        }
        self.scaling = scaling or {
            'alpha0': 1,
            'alpha1': 1,
            'alpha2': 1,
            'beta0': 1,
            'beta1': 1,
            'beta2': 1
        }
        self.seed = seed
        self.training_df = training_df

        self.metrics_lock = None
        
    def cost_function(self, vllm_metrics, sim_metrics):
        total_mape = 0
        for _, metric in enumerate(METRICS_IN_LOSS):
            mape = abs(sim_metrics[f"{metric}_ms"] - vllm_metrics[metric])/vllm_metrics[metric] * 100
            total_mape += mape
        return total_mape
    
    def per_thread_cost(self, vllm_exp, blis_cmd, alpha_coeffs, beta_coeffs):
        """
        Run simulator per experiment thread and obtain simulator results. 
        Compare against vllm ground truth metrics and return cost per experiment
        """
        blis_args = ["run"]
        blis_args.extend(blis_cmd.split(" "))
        extra_args_with_coeffs = {
            "block-size-in-tokens": 16,
            "long-prefill-token-threshold": 0,
            "horizon": "922337203685477580", # Golang int64 max value
            "beta-coeffs": ','.join(beta_coeffs),
            "alpha-coeffs": ','.join(alpha_coeffs),
            "max-prompts": 100,
            "log": "error"
        }
        for key in extra_args_with_coeffs:
            blis_args.extend([f"--{key}", str(extra_args_with_coeffs[key])])
        sim_metrics = run_go_binary(blis_args, GO_BINARY_PATH, self.metrics_lock)
        if not sim_metrics:
            return None
        cost = self.cost_function(vllm_exp, sim_metrics)
        return cost
    
    def run_task_from_tuple(self, args_tuple):
        return self.per_thread_cost(*args_tuple)

    def multiexp_obj(self, trial: optuna.trial.Trial):
        total_cost = 0.0
        tasks = []
        self.metrics_lock = threading.Lock()
        alpha0 = trial.suggest_float('alpha0', *self.pbounds['alpha0'])
        alpha1 = trial.suggest_float('alpha1', *self.pbounds['alpha1'])
        alpha2 = trial.suggest_float('alpha2', *self.pbounds['alpha2'])
        beta0 = trial.suggest_float('beta0', *self.pbounds['beta0'])
        beta1 = trial.suggest_float('beta1', *self.pbounds['beta1'])
        beta2 = trial.suggest_float('beta2', *self.pbounds['beta2'])
        alpha_coeffs = [alpha0, alpha1, alpha2]
        alpha_coeffs = list(map(str, alpha_coeffs))
        beta_coeffs = [beta0, beta1, beta2]
        beta_coeffs = list(map(str, beta_coeffs))

        for idx in range(len(self.training_df)):
            exp_dict = self.training_df.iloc[idx].to_dict()
            blis_instruction = exp_dict["blis_cmd"]
            tasks.append((exp_dict, blis_instruction, alpha_coeffs, beta_coeffs))

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
            print(f"{param}: {self.study.best_params[param]}")
        print("=" * 60)

    def multitrial_obj(self, trial: optuna.trial.Trial):
        print(f"Running trial {trial.number=} in process {os.getpid()}")
        tasks = []
        alpha0 = trial.suggest_float('alpha0', *self.pbounds['alpha0'])
        alpha1 = trial.suggest_float('alpha1', *self.pbounds['alpha1'])
        alpha2 = trial.suggest_float('alpha2', *self.pbounds['alpha2'])
        beta0 = trial.suggest_float('beta0', *self.pbounds['beta0'])
        beta1 = trial.suggest_float('beta1', *self.pbounds['beta1'])
        beta2 = trial.suggest_float('beta2', *self.pbounds['beta2'])
        alpha_coeffs = [alpha0, alpha1, alpha2]
        alpha_coeffs = list(map(str, alpha_coeffs))
        beta_coeffs = [beta0, beta1, beta2]
        beta_coeffs = list(map(str, beta_coeffs))


        for idx in range(len(self.training_df)):
            exp_dict = self.training_df.iloc[idx].to_dict()
            blis_instruction = exp_dict["blis_cmd"]
            tasks.append((exp_dict, blis_instruction, alpha_coeffs, beta_coeffs))
        
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

def get_heuristic_bounds(train_df):
    alpha0_ub = train_df["e2e_mean"].max() * MILLISECONDS_TO_MICROSECONDS_CONVERSION
    alpha1_ub = (train_df["e2e_mean"]/train_df["mean_input_tokens"]).max() * MILLISECONDS_TO_MICROSECONDS_CONVERSION
    alpha2_ub = (train_df["e2e_mean"]/train_df["mean_output_tokens"]).max() * MILLISECONDS_TO_MICROSECONDS_CONVERSION
    train_df["avg_chunk_count"] = train_df["mean_input_tokens"]/train_df['max_num_scheduled_tokens'] * MILLISECONDS_TO_MICROSECONDS_CONVERSION
    beta0_ub = (train_df["e2e_mean"]/(train_df["mean_output_tokens"] + train_df["avg_chunk_count"])).max() * MILLISECONDS_TO_MICROSECONDS_CONVERSION
    beta1_ub = (train_df["ttft_mean"]/train_df["mean_input_tokens"]).max() * MILLISECONDS_TO_MICROSECONDS_CONVERSION
    beta2_ub = ((train_df["e2e_mean"] - train_df["ttft_mean"])/train_df["mean_output_tokens"]).max() * MILLISECONDS_TO_MICROSECONDS_CONVERSION
    return alpha0_ub, alpha1_ub, alpha2_ub, beta0_ub, beta1_ub, beta2_ub

def train_blis_model(training_filepath, LLM_name, tp, gpu, vllm_version, coeffs_filepath):
    # check if coefficients already exist to prevent overwriting
    # read training CSV and filter to only train rows for LLM
    df = pd.read_excel(training_filepath)
    train_df = df[(df["train_test"] == "train") & (df["saturated"] == False) & (df["model_hf_repo"] == LLM_name) & (df["hardware_count"] == int(tp)) & (df["hardware"] == gpu) & (df["docker_image"] == vllm_version)]

    heuristic_upper_bounds = get_heuristic_bounds(train_df)
    heuristics_bounds = {
        "alpha0": (0, 2 * heuristic_upper_bounds[0]),
        "alpha1": (0, 2 * heuristic_upper_bounds[1]),
        "alpha2": (0, 2 * heuristic_upper_bounds[2]),
        "beta0": (0, 2 * heuristic_upper_bounds[3]),
        "beta1": (0, 2 * heuristic_upper_bounds[4]),
        "beta2": (0, 2 * heuristic_upper_bounds[5])
    }

    # Initialize optimizer
    optimizer = InferenceSimOptimizer(
        training_df=train_df,
        pbounds=heuristics_bounds
    )

    # Sequential TPE sampling
    optimizer.optimize_multiexp(n_trials=NUM_TPE_ITERS)
    best_params = optimizer.get_best_trial()["best_params"]
    model_results = {
        "id": LLM_name,
        "tensor_parallelism": int(tp),
        "GPU": gpu,
        "vllm_version": vllm_version,
        "best_loss": optimizer.get_best_trial()["best_loss"],
        "alpha_coeffs": [best_params["alpha0"], best_params["alpha1"], best_params["alpha2"]],
        "beta_coeffs": [best_params["beta0"], best_params["beta1"], best_params["beta2"]]
    }

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

    # read again, and save best optimizer parameters
    lock = FileLock(f"{coeffs_filepath}.lock")
    with lock:
        with open(coeffs_filepath, 'r+') as file:
            all_data = yaml.safe_load(file)
        is_updated = False
        if "models" in all_data:
            if all_data["models"] and len(all_data["models"]) > 0:
                for model in all_data["models"]:
                    # overwrite existing coeffs if exists
                    if model["id"] == LLM_name and model["GPU"] == gpu and model["tensor_parallelism"] == int(tp) and model["vllm_version"] == vllm_version:
                        print("You already have trained coefficients for this combination.", LLM_name, tp, gpu, vllm_version, "Overwriting coeffs...")
                        model["best_loss"] = model_results["best_loss"]
                        model["alpha_coeffs"] = model_results["alpha_coeffs"]
                        model["beta_coeffs"] = model_results["beta_coeffs"]
                        is_updated = True
                        break
                if not is_updated:
                    all_data["models"].append(model_results)
            else:
                all_data["models"] = [model_results]
        else:
            all_data["models"] = [model_results]
        with open(coeffs_filepath, 'w+') as file:
            yaml.dump(all_data, file, default_flow_style=False)
        print(f"BLIS model metrics and coefficients saved to {coeffs_filepath}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Read and parse traces JSON file.")
    parser.add_argument("--LLM-name", 
                        help="LLM to train BLIS coefficients for, pick one from the Excel file")
    parser.add_argument("--tp", 
                        help="TP value to train BLIS coefficients for, pick one from the Excel file")
    parser.add_argument("--GPU", 
                        help="GPU to train BLIS coefficients for, pick one from the Excel file")
    parser.add_argument("--vllm-version", 
                        help="vllm version to train BLIS coefficients for, pick one from the Excel file")
    parser.add_argument("--training-filepath",
                        default="blis_rh_final.xlsx",
                        help="Path to Excel file with GuideLLM RH data.")
    parser.add_argument("--coeffs-filepath",
                        default="coefficients.yaml", 
                        help="Path to save trained BLIS coeffs.")
    parser.add_argument("--specs-filepath",
                        default="training_specs.csv", 
                        help="Path to all combinations to train.")
    args = parser.parse_args()
    if not (args.LLM_name and args.tp and args.GPU and args.vllm_version):
        print("Args not found. Training everything")
        df = pd.read_csv(args.specs_filepath)
        for idx in range(len(df)):
            row_dict = df.iloc[idx].to_dict()
            print("############################################################################")
            print(f"Training BLIS for LLM={row_dict["LLM_name"]}, tp={row_dict["tp"]}, GPU={row_dict["GPU"]}, vllm-version={row_dict["vllm_version"]}")
            print("############################################################################")
            train_blis_model(args.training_filepath, row_dict["LLM_name"], row_dict["tp"], row_dict["GPU"], row_dict["vllm_version"], args.coeffs_filepath)
    else:
        train_blis_model(args.training_filepath, args.LLM_name, args.tp, args.GPU, args.vllm_version, args.coeffs_filepath)
    
