import argparse
import json
import os
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_absolute_error, mean_absolute_percentage_error
from sklearn.linear_model import LinearRegression, RANSACRegressor
import matplotlib.pyplot as plt
import numpy as np

def get_variables(data):
    prompt_lens = data["prompt_lens"]
    block_size = data["block_size"][0]
    prompt_lens_sq_by_b2 = [((x + block_size - 1) // block_size) * ((x + block_size - 1) // block_size) for x in prompt_lens]
    x = np.array([list(pair) for pair in zip(prompt_lens, prompt_lens_sq_by_b2)])
    y = np.array(data["e2e - network_latency"])
    return x, y

def remove_outliers(X_train, y_train):
    # filter out anomalies
    print ("Dataset size before filtering: ", X_train.shape[0])
    mean_y = np.mean(y_train)
    std_y = np.std(y_train)
    z_scores = (y_train - mean_y) / std_y
    filter_mask = np.abs(z_scores) < 2
    y_train = y_train[filter_mask]
    X_train = X_train[filter_mask]
    print ("Dataset size after filtering: ", X_train.shape[0])
    return X_train, y_train

def train_lr(model_name, scenario):
    local_folder_name = f'../results_new/{scenario}/{model_name}'
    items = os.listdir(local_folder_name)
    run_folder_name = sorted([item for item in items if os.path.isdir(os.path.join(local_folder_name, item))], reverse=True)[0]
    
    with open(f'{local_folder_name}/{run_folder_name}/results/{scenario}_output_train.json', 'r') as f:
        train_data = json.load(f)

    with open(f'{local_folder_name}/{run_folder_name}/results/{scenario}_output_test.json', 'r') as file:
        test_data = json.load(file)

    x_scaler = MinMaxScaler()

    X_train, y_train = get_variables(train_data)
    # X_train, y_train = remove_outliers(X_train, y_train)

    X_test, y_test = get_variables(test_data)
    # X_test = x_scaler.fit_transform(X_test)

    model_lr = LinearRegression(positive=True)
    model_lr.fit(X_train, y_train)
    training_score_lr = round(model_lr.score(X_train, y_train), 3)
    test_score_lr = round(model_lr.score(X_test, y_test), 3)
    training_preds_lr = model_lr.predict(X_train)
    test_preds_lr = model_lr.predict(X_test)
    training_mae_lr = round(mean_absolute_error(training_preds_lr, y_train), 3)
    training_mape_lr = round(mean_absolute_percentage_error(training_preds_lr, y_train), 3)
    test_mae_lr = round(mean_absolute_error(test_preds_lr, y_test), 3)
    test_mape_lr = round(mean_absolute_percentage_error(test_preds_lr, y_test), 3)

    # huber = HuberRegressor(positive=True, ).fit(X_train, y_train)
    # training_score_huber = round(huber.score(X_train, y_train), 3)
    # test_score_huber = round(huber.score(X_test, y_test), 3)

    ransac = RANSACRegressor(random_state=0, estimator=model_lr).fit(X_train, y_train)
    training_score_ransac = round(ransac.score(X_train, y_train), 3)
    test_score_ransac = round(ransac.score(X_test, y_test), 3)
    training_preds_ransac = ransac.predict(X_train)
    test_preds_ransac = ransac.predict(X_test)
    training_mae_ransac = round(mean_absolute_error(training_preds_ransac, y_train), 3)
    training_mape_ransac = round(mean_absolute_percentage_error(training_preds_ransac, y_train), 3)
    test_mae_ransac = round(mean_absolute_error(test_preds_ransac, y_test), 3)
    test_mape_ransac = round(mean_absolute_percentage_error(test_preds_ransac, y_test), 3)


    print (f"##################### {model_name} ############################")
    print("LR Model Train Score:", training_score_lr)
    print("LR Model Train MAE:", training_mae_lr)
    print("LR Model Train MAPE:", training_mape_lr)

    print("LR Model Test Score:", test_score_lr)
    print("LR Model Test MAE:", test_mae_lr)
    print("LR Model Test MAPE:", test_mape_lr)

    print("RANSAC Model Train Score:", training_score_ransac)
    print("RANSAC Model Train MAE:", training_mae_ransac)
    print("RANSAC Model Train MAPE:", training_mape_ransac)

    print("RANSAC Model Test Score:", test_score_ransac)
    print("RANSAC Model Test MAE:", test_mae_ransac)
    print("RANSAC Model Test MAPE:", test_mape_ransac)

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(20, 10))

    ax1.scatter(np.array(X_train)[:, 0], y_train)
    ax1.scatter(np.array(X_train)[:, 0], training_preds_lr)
    ax1.set_title(f'Train R2 LR: {training_score_lr}, RANSAC: {training_score_ransac}\nTrain MAE LR: {training_mae_lr}, RANSAC: {training_mae_ransac}\nTrain MAPE LR: {training_mape_lr}, RANSAC: {training_mape_ransac}')
    ax1.set_xlabel('$t_{IN}$')
    ax1.set_ylabel('e2e latency')
    ax1.legend(["Original", "LR predictions"])
    ax1.grid(True)

    ax2.scatter(np.array(X_test)[:, 0], y_test)
    ax2.scatter(np.array(X_test)[:, 0], test_preds_lr)
    ax2.set_title(f'Test R2 LR: {test_score_lr}, RANSAC: {test_score_ransac}\nTest MAE LR: {test_mae_lr}, RANSAC: {test_mae_ransac}\nTest MAPE LR: {test_mape_lr}, RANSAC: {test_mape_ransac}')
    ax2.set_xlabel('$t_{IN}$')
    ax2.set_ylabel('e2e latency')
    ax2.legend(["Original", "LR predictions"])
    ax2.grid(True)

    fig.suptitle(f'Results for {model_name}')

    plt.savefig(f"../plots_new/{args.scenario}/{model_name}/{model_name}_e2e_vs_input_lens.png")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Simple vLLM Benchmark Runner')
    parser.add_argument('--run_folder_name', help='latest run folder',  default="20250826-115550_scenario2")
    parser.add_argument('--scenario', help='scenario X',  default="scenario2")
    args = parser.parse_args()
    models = ["facebook/opt-125m", "Qwen/Qwen2.5-0.5B", "Qwen/Qwen2-1.5B", "Qwen/Qwen2.5-3B", "Qwen/Qwen2-7B", "Qwen/Qwen3-14B", "mistralai/Mistral-7B-Instruct-v0.1", "google/gemma-7b", "meta-llama/Llama-3.1-8B","ibm-granite/granite-3.3-8b-instruct", "mistralai/Mistral-Small-24B-Instruct-2501", "Qwen/Qwen3-32B"]
    for model in models:
        model_name = model.split("/")[-1].replace(".", "_")
        os.makedirs(f"../plots_new/{args.scenario}/{model_name}", exist_ok=True)
        train_lr(model_name, args.scenario)

## C    m     delta C*m m*delta.  e2e
## 256. 1.      0.                
## 256. 1.      0                
## 256. 1.      20.  
# ...           
## 256. 2.      40



