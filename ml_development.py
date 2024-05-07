# Databricks notebook source
from datetime import datetime, timedelta
import pandas as pd
from random import randint

from sklearn.model_selection import train_test_split


# Model Pre-Processing & Imputation
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler, KBinsDiscretizer, Binarizer, OneHotEncoder, OrdinalEncoder, FunctionTransformer, OneHotEncoder, LabelBinarizer
import category_encoders as ce
from sklearn.impute import SimpleImputer
from sklearn.metrics import roc_auc_score, RocCurveDisplay, classification_report, explained_variance_score

# Classification Models
from sklearn.linear_model import LogisticRegression, LinearRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from xgboost import XGBClassifier
from sklearn.model_selection import GridSearchCV


# Experiment Tracking
import mlflow

# COMMAND ----------

class MLData:
    def __init__(self, df, target, X_vars):
        self.df = df
        self.target = target
        self.X_vars = X_vars
    
    def split_random(self, random_state=42, test_size=0.25, stratify=None):
        self.X_train, self.X_test, self.y_train, self.y_test = train_test_split(self.df[self.X_vars], self.df[self.target], test_size=test_size, random_state=random_state, 
                                                                                stratify = stratify)

    def split_by_dates(self, val_start, test_start, end_date):

        train_df = self.df[self.df['create_date'] < pd.to_datetime(val_start)]
        self.X_train = train_df[self.X_vars]
        self.y_train = train_df[self.target].astype(int)

        val_df = self.df[(self.df['create_date'] < pd.to_datetime(test_start)) & (self.df['create_date'] >= pd.to_datetime(val_start))]
        self.X_val = val_df[self.X_vars]
        self.y_val = val_df[self.target].astype(int)
        
        test_df = self.df[self.df['create_date'] >= pd.to_datetime(test_start)]
        self.X_test = test_df[self.X_vars]
        self.y_test = test_df[self.target].astype(int)
        
    def describe(self):
        print("Shapes of Train X and y ", self.X_train.shape, self.y_train.shape)
        print("Train DF Target Distribution:")
        print(self.y_train.value_counts(normalize=False))
        print(self.y_train.value_counts(normalize=True))

        try:
            print("Shapes of Val X and y ", self.X_val.shape, self.y_val.shape)
            print("Val DF Target Distribution:")
            print(self.y_val.value_counts(normalize=False))
            print(self.y_val.value_counts(normalize=True))
        except AttributeError:
            pass

        print("Shapes of Test X and y ", self.X_test.shape, self.y_test.shape)
        print("Test DF Target Distribution:")
        print(self.y_test.value_counts(normalize=False))
        print(self.y_test.value_counts(normalize=True))


# COMMAND ----------

class MLExperiment:
    def __init__(self, data: MLData, preprocessor, estimator, transformer_col_function=None):
        self.data = data
        self.preprocessor = preprocessor
        self.estimator = estimator
        self.transformer_col_function = transformer_col_function

    def start_log_to_mlflow(self):
        mlflow.end_run()
        mlflow.start_run(run_name=str(randint(225000000000, 225999999999)), experiment_id=EXPERIMENT_ID)
        # mlflow.sklearn.autolog()
        mlflow.set_tag('description', DESCRIPTION)
        mlflow.log_metric('BaselineAUC', BASELINE_AUC)
        mlflow.set_tag("DEPENDENCY_STACK", library_common_get_cluster_dependency_stack_version())

    def fit_preprocessor(self):
        self.preprocessor.fit(self.data.X_train)

    def fit_estimator(self):
        self.estimator.fit(self.X_train_transformed, self.data.y_train)

    @property
    def X_train_transformed(self):
        return self.preprocessor.transform(self.data.X_train)

    @property
    def X_val_transformed(self):
        return self.preprocessor.transform(self.data.X_val)

    @property
    def X_test_transformed(self):
        return self.preprocessor.transform(self.data.X_test)

    @property
    def y_train_predict_prob(self):
        return self.estimator.predict_proba(self.X_train_transformed)[:,1]

    @property
    def y_val_predict_prob(self):
        return self.estimator.predict_proba(self.X_val_transformed)[:,1]

    @property
    def y_test_predict_prob(self):
        return self.estimator.predict_proba(self.X_test_transformed)[:,1]

    @property
    def X_transformed_column_names(self):
        return self.transformer_col_function(self.preprocessor)

    @property
    def auc_score_train(self):
        return roc_auc_score(self.data.y_train, self.y_train_predict_prob)

    @property
    def auc_score_val(self):
        return roc_auc_score(self.data.y_val, self.y_val_predict_prob)

    @property
    def auc_score_test(self):
        return roc_auc_score(self.data.y_test, self.y_test_predict_prob)

    def show_roc_train(self):
        RocCurveDisplay.from_estimator(self.estimator, self.X_train_transformed, self.data.y_train)

    def show_roc_val(self):
        RocCurveDisplay.from_estimator(self.estimator, self.X_val_transformed, self.data.y_val)

    def show_roc_test(self):
        RocCurveDisplay.from_estimator(self.estimator, self.X_test_transformed, self.data.y_test)

    @property
    def best_estimator(self):
        try:
            return self.estimator.best_estimator_
        except:
            return self.estimator

    @property
    def feature_importances(self):
        try:
            return pd.DataFrame(self.best_estimator.feature_importances_, index = self.X_transformed_column_names, columns=['importance']).sort_values('importance', ascending=False)
        except:
            try:
                return self.best_estimator.feature_importances_
            except:
                return ['Feature Importance Not Available']

    def log_train_to_mlflow(self):
        mlflow.log_metric('TrainAUC', self.auc_score_train)
        
    def log_val_to_mlflow(self):
        mlflow.log_metric('ValAUC', self.auc_score_val)
                
    def log_test_to_mlflow(self):
        mlflow.log_metric('TrainAUC', self.auc_score_test)
        
    def log_feature_importance_to_mlflow(self):
        try:
          mlflow.set_tag('feature_importance', self.feature_importances.to_dict())
        except AttributeError:
          mlflow.set_tag('feature_importance', self.feature_importances)
          
    def log_model_parameters_to_mlflow(self):
         mlflow.log_params(self.best_estimator.get_params())
        
    def log_feature_names_to_mlflow(self):
        mlflow.set_tag("feature_names", ", ".join(self.data.X_vars)) 
        
    def log_mlflow_end(self):
        mlflow.end_run()

    def run_and_log_train_and_val(self):
        print('Fit Preprocessor')
        self.fit_preprocessor()
        print('Fit Estimator')
        self.fit_estimator()
#        self.show_roc_train()
#        self.show_roc_val()
        print('Log to ML Flow')
        self.start_log_to_mlflow()
        self.log_train_to_mlflow()
        self.log_val_to_mlflow()
        self.log_feature_importance_to_mlflow()
        self.log_feature_names_to_mlflow()
        self.log_model_parameters_to_mlflow()
        self.log_mlflow_end()

    def run_and_log_train_val_test(self):
        self.fit_preprocessor()
        self.fit_estimator()
        self.show_roc_train()
        self.show_roc_val()
        self.show_roc_test()
        self.start_log_to_mlflow()
        self.log_train_to_mlflow()
        self.log_val_to_mlflow()
        self.log_test_to_mlflow()
        self.log_feature_importance_to_mlflow()
        self.log_feature_names_to_mlflow()
        self.log_model_parameters_to_mlflow()
        self.log_mlflow_end()



# COMMAND ----------

def get_xgboost_gridsearch():
    estimator = XGBClassifier(
    objective= 'binary:logistic',
    nthread=4,
    seed=42
    )

    parameters = {
      'max_depth': range(2, 10, 1),
      'n_estimators': [50, 100, 150, 200],
      'learning_rate': [.01, .02, .03, .05, .07, .1, .2, .3]
    }

    grid_search = GridSearchCV(
      estimator=estimator,
      param_grid=parameters,
      scoring = 'roc_auc',
      n_jobs = 10,
      cv = 4,
      verbose=True
    )
    return grid_search


class LinearRegressionExperiment:
    def __init__(self, data, x, y, fit_intercept):
        self.data = data
        self.x_columns = x
        self.x = data[x]
        self.y = data[y]
        self.estimator = LinearRegression(fit_intercept=fit_intercept)

        self.fit_estimator()
        print("Coef:", list(zip(self.x_columns, self.coefs)))
        print("Intercept:", self.intercept)
        print("Explained Variance:", self.explained_variance)

    def fit_estimator(self):
        self.estimator.fit(self.x, self.y)

    @property
    def coefs(self):
        return self.estimator.coef_

    @property
    def y_prediction(self):
        return self.estimator.predict(self.x)

    @property
    def explained_variance(self):
        return explained_variance_score(self.y, self.y_prediction)

    @property
    def intercept(self):
        return self.estimator.intercept_
