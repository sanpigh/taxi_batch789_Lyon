from audioop import cross
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.preprocessing import OneHotEncoder
from sklearn.model_selection import train_test_split, cross_validate
from sklearn.compose import ColumnTransformer
from sklearn.linear_model import LinearRegression, Lasso, Ridge
from sklearn.ensemble import GradientBoostingRegressor, RandomForestRegressor
from xgboost import XGBRegressor, cv
from taxi_batch789_Lyon.encoders import DistanceTransformer, TimeFeaturesEncoder
from taxi_batch789_Lyon.utils import compute_rmse, save_model


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
        #        self.X = X
        #        self.y = y

        self.kwargs = kwargs
        self.cv = self.kwargs.get("cv", 0)
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
        model = self.get_estimator()
        self.pipeline = Pipeline([
            ('preproc', preproc_pipe),
            (model.__class__.__name__, model) #model name and model itself (object)
        ])

    def get_estimator(self):
        estimator = self.kwargs.get('estimator', 'Linear')
        if estimator == "Lasso":
            model = Lasso()
        elif estimator == "Linear":
            model = LinearRegression()
        elif estimator == "Ridge":
            model = Ridge()
        elif estimator == "GBM":
            model = GradientBoostingRegressor()
        elif estimator == "RandomForest":
            model = RandomForestRegressor()
            self.model_params = {  # 'n_estimators': [int(x) for x in np.linspace(start = 50, stop = 200, num = 10)],
                'max_features': ['auto', 'sqrt']}
            # 'max_depth' : [int(x) for x in np.linspace(10, 110, num = 11)]}
        elif estimator == "xgboost":
            model = XGBRegressor(objective='reg:squarederror', n_jobs=-1, max_depth=10, learning_rate=0.05,
                                 gamma=3)
#            self.model_params = {'max_depth': range(10, 20, 2),
#                                 'n_estimators': range(60, 220, 40),
#                                 'learning_rate': [0.1, 0.01, 0.05]
#                                 }
        else:
            model = LinearRegression()
        return model



    def run_train(self):
        """set and train the pipeline on the train set"""
        self.set_pipeline()
        self.pipeline.fit(self.X_train,self.y_train)
        save_model(self.pipeline)


    def evaluate(self):
        """evaluates the pipeline on train and test sets and return the RMSE.
        If cv>1 evaluates also on validation set"""
        y_test_pred = self.pipeline.predict(self.X_test)
        rmse_test_error = compute_rmse(self.y_test, y_test_pred)


        run = self.mlflow_run
        yourname = 'sanpigh'
        model_name = self.pipeline.steps[-1][0]  # get the name from the pipe

        if self.cv > 1:
            cv_result = cross_validate(self.pipeline,
                                       self.X_train,
                                       self.y_train,
                                       cv=self.cv,
                                       scoring='neg_root_mean_squared_error',
                                       return_train_score=True,
                                       #n_jobs=-1 # generates problems with xgboost
                                       )

            rmse_train_error = -cv_result['train_score'].mean()
            rmse_val_error   = -cv_result['test_score'].mean()
            self.mlflow_client.log_param(run.info.run_id, "cross-validated train and val scores",
                                          'Yes')
            self.mlflow_client.log_metric(run.info.run_id, "rmse_val_error",
                                          rmse_val_error)
        else:
            y_train_pred = self.pipeline.predict(self.X_train)
            rmse_train_error = compute_rmse(self.y_train,y_train_pred)
            self.mlflow_client.log_param(run.info.run_id, "cross-validated train and val scores",
                                          'No')

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
