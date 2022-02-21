from audioop import cross
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.preprocessing import OneHotEncoder
from sklearn.model_selection import train_test_split, cross_validate
from sklearn.compose import ColumnTransformer
from sklearn.linear_model import LinearRegression
from taxi_batch789_Lyon.encoders import DistanceTransformer, TimeFeaturesEncoder
from taxi_batch789_Lyon.utils import compute_rmse


import mlflow
from memoized_property import memoized_property
from mlflow.tracking import MlflowClient

EXPERIMENT_NAME = "[FR] [Bordeaux] [sanpigh] tests"
# Indicate mlflow to log to remote server
MLFLOW_URI = 'https://mlflow.lewagon.co/'


class Trainer():
    def __init__(self, X, y, **kwargs):
        """
            X: pandas DataFrame
            y: pandas Series
        """
        # assert X is dataframe
        # assert y is series
        self.pipeline = None
        self.X = X
        self.y = y

        self.kwargs = kwargs
        self.cross_val = self.kwargs.get("cross_val", 0)
#        self.score = self.kwargs.get('scores', 'neg_root_mean_squared_error')

        self.X_train, self.X_test, self.y_train, self.y_test = train_test_split(X,y,
                                                        test_size=0.3)
        del X, y

        self.experiment_name = EXPERIMENT_NAME
        self.mlflow_experiment_id

    def set_pipeline(self):
        """defines the pipeline as a class attribute"""
        # create distance pipeline
        dist_pipe = Pipeline([
            ('dist_trans', DistanceTransformer()),
            ('stdscaler', StandardScaler())
        ])
        # create time pipeline
        time_pipe = Pipeline([
            ('time_enc', TimeFeaturesEncoder('pickup_datetime')),
            ('ohe', OneHotEncoder(handle_unknown='ignore'))
        ])
        # create preprocessing pipeline
        preproc_pipe = ColumnTransformer([
            ('distance', dist_pipe, ["pickup_latitude", "pickup_longitude", 'dropoff_latitude', 'dropoff_longitude']),
            ('time', time_pipe, ['pickup_datetime'])
        ], remainder="drop")
        # Add the model to the pipeline
        self.pipeline = Pipeline([
            ('preproc', preproc_pipe),
            ('linear_model', LinearRegression())
        ])

    def run(self):
        """set and train the pipeline"""
        self.set_pipeline()
        self.pipeline.fit(self.X_train,self.y_train)


    def evaluate(self):

        if self.cross_val == 0:
            """evaluates the pipeline on X_test and return the RMSE"""
            y_train_pred = self.pipeline.predict(self.X_train)
            y_test_pred  = self.pipeline.predict(self.X_test)
            rmse_train_error = compute_rmse(self.y_train,y_train_pred)
            rmse_test_error  = compute_rmse(self.y_test,y_test_pred)
        else:
            results = cross_validate(self.X_train, self.y_train, cv=self.cross_val,scoring=self.score)


        model_name = self.pipeline.steps[-1][0] # get the name from the pipe
        yourname   = 'sanpigh'
        run = self.mlflow_run
        self.mlflow_client.log_metric(run.info.run_id, "rmse_train_error",
                                      rmse_train_error)
        self.mlflow_client.log_metric(run.info.run_id, "rmse_test_error",
                                      rmse_test_error)
        self.mlflow_client.log_param(run.info.run_id, "model", model_name)
        self.mlflow_client.log_param(run.info.run_id, "student_name", yourname)
        return rmse_test_error

# MLFLOW

    @memoized_property            # Crear un objeto MlflowClient virgen
    def mlflow_client(self):
        mlflow.set_tracking_uri(MLFLOW_URI)
        return MlflowClient()

    @memoized_property            # Asignar a ese objeto un experiment experimetn_name
    def mlflow_experiment_id(self):
        try:
            return self.mlflow_client.create_experiment(self.experiment_name)
        except BaseException:
            return self.mlflow_client.get_experiment_by_name(
                self.experiment_name).experiment_id

    @memoized_property
    def mlflow_run(self):
        return self.mlflow_client.create_run(self.mlflow_experiment_id)

    def mlflow_log_param(self, key, value):
        self.mlflow_client.log_param(self.mlflow_run.info.run_id, key, value)

    def mlflow_log_metric(self, key, value):
        self.mlflow_client.log_metric(self.mlflow_run.info.run_id, key, value)


if __name__ == "__main__":
    # get data
    # clean data
    # set X and y
    # hold out
    # train
    # evaluate
    print('TODO')
