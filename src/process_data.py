import pandas as pd
import yaml
from sklearn.model_selection import train_test_split
    
params = yaml.safe_load(open("params.yaml"))["features"]
data = params["data_path"]
test = params["test_size"]
val = params["val_size"]

df = pd.read_csv(data)

df['explicit'] = df['explicit'].apply(lambda x: int(x))

df_filter = df.drop(['Unnamed: 0', 'track_id', 'album_name', 'track_name', 'artists'], axis=1)

df_filter['y'] = df_filter['popularity']
X = df_filter.drop('popularity', axis=1)

X_train_val, X_test = train_test_split(X, test_size=test)

X_train, X_val = train_test_split(X_train_val, test_size=val)

X_train.to_csv('data/processed_train_data.csv')
X_val.to_csv('data/processed_val_data.csv')
X_test.to_csv('data/processed_test_data.csv')



