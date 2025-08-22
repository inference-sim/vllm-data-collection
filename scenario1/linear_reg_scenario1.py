import json
from sklearn.preprocessing import MinMaxScaler
from sklearn.linear_model import LinearRegression
import matplotlib.pyplot as plt
from scipy.stats import zscore
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

with open('results_train.json', 'r') as f:
    train_data = json.load(f)

with open('results_test.json', 'r') as file:
    test_data = json.load(file)

x_scaler = MinMaxScaler()

X_train, y_train = get_variables(train_data)
# X_train, y_train = remove_outliers(X_train, y_train)

X_test, y_test = get_variables(test_data)
# X_test = x_scaler.fit_transform(X_test)

model = LinearRegression(positive=True)
model.fit(X_train, y_train)
print("LR Model Train Score:", model.score(X_train, y_train))
print("LR Model Test Score:", model.score(X_test, y_test))

coefficients = model.coef_
intercept = model.intercept_
print("Coefficients:", str(coefficients))
print("Intercept:", intercept)

# plt.figure(figsize=(8, 6))
# plt.scatter(np.array(X_train)[:, 0], y_train)

# # Add a title and labels
# plt.title('e2e latency vs $t_{IN}$')
# plt.xlabel('$t_{IN}$')
# plt.ylabel('e2e latency')

# # Add a grid and a legend
# plt.grid(True)
# plt.legend()
# plt.savefig("e2e_minus_network_vs_tin_filtered_train.png")



