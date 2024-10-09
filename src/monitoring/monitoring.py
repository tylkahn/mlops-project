from scipy.stats import kstest
import pickle
import pandas as pd
import matplotlib.pyplot as plt
import sklearn
import os

with open('new_model.pkl', 'rb') as f:
    model = pickle.load(f)

df = pd.read_csv('data/processed_val_data.csv')
X = df.drop(['y','track_genre','Unnamed: 0','duration_ms','explicit','key','time_signature'], axis=1)
y = df['y']

X_ref = X[:100]

metrics = {}

#Kolmogorov–Smirnov test
for column in X_ref:
    print(column)
    metrics[column] = []
    for i in range(1,25):
        X_new = X[100*i:100*(i+1)]
        r = kstest(X_ref[column], X_new[column])
        metrics[column].append(r.statistic)

metrics['predictions'] = []
metrics['performance'] = []
for i in range(1,25):
    X_new = X[100*i:100*(i+1)]
    y_true = y[100*i:100*(i+1)]
    y_pred = model.predict(X_new)
    r = kstest(model.predict(X_ref), y_pred)
    metrics['predictions'].append(r.statistic)
    metrics['performance'].append(sklearn.metrics.r2_score(y_true, y_pred))
print(metrics)

results = pd.DataFrame.from_dict(metrics)
results.to_csv("monitoring.csv")

output_dir = 'plots'
os.makedirs(output_dir, exist_ok=True)

for key, array in metrics.items():
    plt.figure()
    plt.plot(array)
    plt.title(f'{key} plot')
    plt.xlabel('Index')
    plt.ylabel('Value')
    plt.ylim(bottom=0)
    
    file_path = os.path.join(output_dir, f'{key}.png')
    plt.savefig(file_path)
    plt.close()