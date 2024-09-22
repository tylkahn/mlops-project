from metaflow import FlowSpec, step, Flow, Parameter

class PredictFlow(FlowSpec):
    path = Parameter('path', type=str, required=True)

    @step
    def start(self):
        import pandas as pd
        run = Flow('TrainFlow').latest_run 
        self.train_run_id = run.pathspec 
        self.model = run['end'].task.data.model
        df = pd.read_csv(self.path)
        self.X = df.drop(['y','track_genre','Unnamed: 0','duration_ms','explicit','key','time_signature'], axis=1)
        self.next(self.end)

    @step
    def end(self):
        print('Model', self.model)
        print('Predicted score', self.model.predict(self.X))

if __name__=='__main__':
    PredictFlow()