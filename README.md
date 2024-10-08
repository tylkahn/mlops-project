**MLOPS Project**

The goal of this project is to develop a pipeline for training a model that predicts the popularity of songs on Spotify given some features generated by Spotify.

The general pipeline is outlined below:

- Data is preprocessed via a script.
- The processed data is tested for quality using Greater Expectations.
- Metaflow is used to create flows for training and scoring models.
- The flows are run on GKE.
- The results of the experiments are recorded using mlflow, and the best performing model is saved and registered.
- The best performing model is used to score incoming data in batches.
- The model is monitored for drift and performance via a script. R2 and Kolmogorov–Smirnov test are used as metrics.

The pipeline is deployed using GCP.

Results:

The best model was a random forest with 100 trees and depth 5. This model had a training r2 of approximately 0.49.

<img width="855" alt="pipeline" src="https://github.com/user-attachments/assets/0699936a-bf28-43c1-b3eb-3426478a3c3a">

