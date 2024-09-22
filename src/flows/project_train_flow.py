from metaflow import FlowSpec, step, Parameter, kubernetes

class TrainFlow(FlowSpec):
    path = Parameter('path', type=str, required=True)

    @step
    def start(self):
        import pandas as pd
        df = pd.read_csv(self.path)
        self.X = df.drop(['y','track_genre','Unnamed: 0','duration_ms','explicit','key','time_signature'], axis=1)
        self.y = df['y']

        print("Loaded data")
        self.ntrees = [20,40,60,80,100]
        self.next(self.train_rf, foreach='ntrees')

    @kubernetes(image='metaflow_env')
    @step
    def train_rf(self):
        from sklearn.ensemble import RandomForestRegressor
        import mlflow
        mlflow.set_tracking_uri('https://mlops-19873747083.us-west2.run.app')
        mlflow.set_experiment('metaflow')

        mlflow.end_run()
        with mlflow.start_run():
            self.model = RandomForestRegressor(n_estimators = self.input, max_features = 5, oob_score = True)
            self.model.fit(self.X, self.y)
            r2 = self.model.oob_score_
            mlflow.log_artifact(self.path)
            mlflow.log_metric('r2', r2)
        mlflow.end_run()
        self.next(self.choose_model)

    @step
    def choose_model(self, inputs):   
        def score(inp):
            r2 = inp.model.oob_score_
            return inp.model, r2

        self.results = sorted(map(score, inputs), key=lambda x: -x[1])
        self.model = self.results[0][0]
        self.next(self.end)

    @step
    def end(self):
        print('Scores:')
        print('\n'.join('%s %f' % res for res in self.results))
        print('Model:', self.model)


if __name__=='__main__':
    TrainFlow()