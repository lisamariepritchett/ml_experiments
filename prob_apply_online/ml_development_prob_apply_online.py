# Databricks notebook source
from datetime import datetime, timedelta
import pandas as pd
from random import randint
# Model Pre-Processing & Imputation
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler, KBinsDiscretizer, Binarizer, OneHotEncoder, OrdinalEncoder, FunctionTransformer, OneHotEncoder, LabelBinarizer
import category_encoders as ce
from sklearn.impute import SimpleImputer
from sklearn.metrics import roc_auc_score, RocCurveDisplay, classification_report

# Classification Models
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from xgboost import XGBClassifier
from sklearn.model_selection import GridSearchCV


# Experiment Tracking
import mlflow

# COMMAND ----------

# MAGIC %run "./query_prod_digital_submit"

# COMMAND ----------

def save_df_to_databricks(df, table_name, schema_name = 'lisapritchett'):
  '''save a pandas dataframe to databricks as schema_name.table_name'''
  spark.createDataFrame(df).write.mode("overwrite").saveAsTable("{}.{}".format(schema_name, table_name))
  

def get_df_from_databricks(table_name, schema_name = 'lisapritchett'):
  '''return a pandas dataframe saved in databricks schema_name.table_name'''
  return spark.sql("SELECT * FROM {schema}.{table}".format(schema=schema_name, table=table_name)).toPandas()


def check_df_for_primary_key(df, key_col):
  try:
    assert len(df) == len(df['lead_id'].unique())
    print("Yes, {} is a primary key with {} rows".format(key_col, len(df)))
  except:
    print("No, {} is not a primary key".format(key_col))

# COMMAND ----------

class MLData:
    def __init__(self, df, start_date, val_start, test_start, end_date, target, X_vars, pkey):
        check_df_for_primary_key(df, pkey)
        self.df = df.set_index(pkey)
        self.date_dict = {'start_date': start_date,
                          'val_start': val_start,
                          'test_start': test_start,
                          'end_date': end_date
                          }
        self.target = target
        self.X_vars = X_vars
        self.pkey = pkey

    def split_by_dates(self):
        val_start = self.date_dict['val_start']
        test_start = self.date_dict['test_start']

        train_df = self.df[self.df['create_date'] < pd.to_datetime(val_start)]
        self.X_train = train_df[self.X_vars]
        self.y_train = train_df[self.target]

        val_df = self.df[(self.df['create_date'] < pd.to_datetime(test_start)) & (self.df['create_date'] >= pd.to_datetime(val_start))]
        self.X_val = val_df[self.X_vars]
        self.y_val = val_df[self.target]
        
        test_df = self.df[self.df['create_date'] >= pd.to_datetime(test_start)]
        self.X_test = test_df[self.X_vars]
        self.y_test = test_df[self.target]
        
    def describe(self):
#         print("Time period for training data: " + str(self.date_dict['start_date']) + " -- " + str(end_train))
#         print("Time period for test data: " + str(end_train) + " -- " + str(end_date))

        print("Shapes of Train X and y ", self.X_train.shape, self.y_train.shape)
        print("Shapes of Val X and y ", self.X_val.shape, self.y_val.shape)
        print("Shapes of Test X and y ", self.X_test.shape, self.y_test.shape)

        print("Train DF Target Distribution:")
        print(self.y_train.value_counts(normalize=False))
        print(self.y_train.value_counts(normalize=True))

        print("Val DF Target Distribution:")
        print(self.y_val.value_counts(normalize=False))
        print(self.y_val.value_counts(normalize=True))

        print("Test DF Target Distribution:")
        print(self.y_test.value_counts(normalize=False))
        print(self.y_test.value_counts(normalize=True))


# COMMAND ----------

class MLExperiment:
    def __init__(self, data: MLData, preprocessor, estimator, transformer_col_function, baseline_auc):
        self.data = data
        self.preprocessor = preprocessor
        self.estimator = estimator
        self.transformer_col_function = transformer_col_function
        self.transformed_col_names = None
        self.X_transformed = None
        self.baseline_auc = baseline_auc
        
    def start_log_to_mlflow(self):
        mlflow.end_run()
        mlflow.start_run(run_name=str(randint(225000000000, 225999999999)), experiment_id=EXPERIMENT_ID)
        # mlflow.sklearn.autolog()
        mlflow.set_tag('description', DESCRIPTION)
        mlflow.log_metric('BaselineAUC', self.baseline_auc)
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
            return pd.Series(self.best_estimator.feature_importances_, index = self.X_transformed_column_names)
        except:
            try:
                return self.best_estimator.feature_importances_
            except:
                return ['']

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

# choose dates for splits
anchor_date = datetime.strptime('2022-10-30', '%Y-%m-%d').date()
START_DATE = anchor_date - timedelta(days=180)
VAL_START = anchor_date - timedelta(days=60)
TEST_START = anchor_date - timedelta(days=30)
END_DATE = anchor_date - timedelta(days=1)


# COMMAND ----------

# get data from sql or from saved
# full_df = get_model_dataset(START_DATE, END_DATE)
# save_df_to_databricks(full_df, 'prob_digital_submit_life_dev_oct30_more_features')
full_df = get_df_from_databricks('prob_digital_submit_life_dev_oct30_more_features')


# COMMAND ----------

# identify columns
TARGET = 'target'
X_VARS = true_boolean_cols + float_cols + int_cols + str_cols
PKEY = 'lead_id'


# COMMAND ----------

FLOAT_COLS_USE = float_cols
INT_COLS_USE = int_cols
BOOLEAN_COLS_USE = true_boolean_cols
STRING_COLS_USE = ['individual_gender']

# COMMAND ----------

def greater_than_zero(x):
  ''' If the value is greater than zero return 1 otherwise return 0'''
  if x > 0:
    return 1
  else:
    return 0
  

def encode_browser(x):
  if x in ['Chrome', 'Facebook', 'Safair', 'Webview/iOS']:
    return x
  else:
    return 'other'
  

def is_true(x):
  if x:
    return 1
  else:
    return 0
  
  
def encode_gender(x):
  if x in ['M', 'F']:
    return x
  else:
    return 'other'
  
  
def encode_domain(x):
  if x in ['lifeinsurance.net', 'nationalfamily.com', 'assurance.com']:
    return x
  else:
    return 'other'
  
  
def encode_verify_email(x):
  if x == 'exact match':
    return 0
  elif x == 'close match':
    return 1
  else: 
    return 2

# COMMAND ----------

# build preprocessor
float_transformer = Pipeline(steps = [
  ('imputer', SimpleImputer(strategy="mean"))
])

integer_transformer = Pipeline(steps = [
  ('imputer', SimpleImputer(strategy="constant", fill_value = 0))
  , ('function_greater_than_zero', FunctionTransformer(np.vectorize(greater_than_zero), validate=False))
])

boolean_transformer = Pipeline(steps = [
  ('function_is_true', FunctionTransformer(np.vectorize(is_true), validate=False))
])

gender_transformer = Pipeline(steps = [
  ('function_encode_gender', FunctionTransformer(np.vectorize(encode_gender), validate=False))
  , ('one_hot_encoder', OneHotEncoder(drop='if_binary'))
])

browser_transformer = Pipeline(steps = [
  ('function_encode_browser', FunctionTransformer(np.vectorize(encode_browser), validate=False))
  , ('one_hot_encoder', OneHotEncoder(drop='if_binary'))
])

domain_transformer = Pipeline(steps = [
  ('function_encoded_domain', FunctionTransformer(np.vectorize(encode_domain), validate=False))
  , ('one_hot_encoder', OneHotEncoder(drop='if_binary'))
])

email_transformer = Pipeline(steps = [
  ('function_encode_verify_email', FunctionTransformer(np.vectorize(encode_verify_email), validate=False))
  , ('one_hot_encoder', OneHotEncoder(drop='if_binary'))
])


preprocessor = ColumnTransformer(
    transformers=[
        ("float_transformer", float_transformer, FLOAT_COLS_USE),
        ("integer_transformer", integer_transformer, INT_COLS_USE),
        ("boolean_transformer", boolean_transformer, BOOLEAN_COLS_USE),
        ("gender_transformer", gender_transformer, ['individual_gender']),
        ("browser_transformer", browser_transformer, ['direct_lead_browser']),
        ("domain_transformer", domain_transformer, ['direct_lead_domain']),
        ("email_transformer", email_transformer, ['ebureau_verify_email']),
    ]
)

# COMMAND ----------

def get_column_names_transformed(preprocessor):
    """ Customize this function to get the columns from preprocessor - depending on the attributes of the preprocessor this will look different. Pass this function to the MLExperiment """
    preprocessor = ml_exp1.preprocessor
    float_col_names = float_cols
    int_col_names = int_cols
    boolean_col_names = true_boolean_cols
    string_col_names = STRING_COLS_USE
    browser_col_names = ['direct_lead_browser']
    gender_col_names = list(preprocessor.named_transformers_['gender_transformer'] \
                             .named_steps['one_hot_encoder'] \
                              .get_feature_names_out(string_col_names))
    browser_cols_use = list(preprocessor.named_transformers_['browser_transformer'] \
                            .named_steps['one_hot_encoder'] \
                            .get_feature_names_out(browser_col_names))
    domain_cols_use = list(preprocessor.named_transformers_['domain_transformer'] \
                            .named_steps['one_hot_encoder'] \
                            .get_feature_names_out(browser_col_names))
    email_cols_use = list(preprocessor.named_transformers_['email_transformer'] \
                            .named_steps['one_hot_encoder'] \
                            .get_feature_names_out(browser_col_names))
    col_names_transformed = float_col_names + int_col_names + boolean_col_names + gender_col_names + browser_cols_use + domain_cols_use + email_cols_use
    
    return col_names_transformed

# COMMAND ----------

# experiment id
EXPERIMENT_ID = 422
DESCRIPTION = 'Trying more features'

MODEL_NAME = 'prob_apply_online'


# COMMAND ----------

# build estimator
estimator = XGBClassifier(max_depth=3, learning_rate = 0.05, seed = 422)


# COMMAND ----------

# build ml datasets
ml_data = MLData(df = full_df, start_date = START_DATE, val_start = VAL_START, test_start = TEST_START, end_date = END_DATE,
                 target = TARGET, X_vars = X_VARS, pkey = PKEY)
ml_data.split_by_dates()
ml_data.describe()

# COMMAND ----------

# do ml experiment
ml_exp1 = MLExperiment(ml_data, preprocessor=preprocessor, estimator=estimator, transformer_col_function=get_column_names_transformed, baseline_auc=.75)

# COMMAND ----------

ml_exp1.fit_preprocessor()
ml_exp1.fit_estimator()
ml_exp1.auc_score_train, ml_exp1.auc_score_val

# COMMAND ----------

# run all train and validation logging to mlflow
ml_exp1.run_and_log_train_and_val()

# COMMAND ----------

# MAGIC %md
# MAGIC #### XGB Grid Search 

# COMMAND ----------

estimator = XGBClassifier(
    objective= 'binary:logistic',
    nthread=4,
    seed=42
)

parameters = {
    'max_depth': range (2, 10, 1),
    'n_estimators': range(60, 220, 40),
    'learning_rate': [0.3, 0.1, 0.01, 0.05]
}

grid_search = GridSearchCV(
    estimator=estimator,
    param_grid=parameters,
    scoring = 'roc_auc',
    n_jobs = 10,
    cv = 4,
    verbose=True
)

DESCRIPTION = 'xgb-grid-search'

# COMMAND ----------

# do ml experiment
a = MLExperiment(ml_data, preprocessor=preprocessor, estimator=grid_search, transformer_col_function=get_column_names_transformed, baseline_auc=.75)

# COMMAND ----------

a.run_and_log_train_and_val()

# COMMAND ----------

# MAGIC %md
# MAGIC #### Random Forest Grid Search
# MAGIC

# COMMAND ----------

range (2, 20, 2)

# COMMAND ----------

estimator = RandomForestClassifier(random_state = 42)

parameters = {'bootstrap': [True, False],
 'max_depth': range(2,16,2),
 'max_features': ['auto', 'sqrt'],
 'min_samples_leaf': [1, 2, 4],
 'min_samples_split': [2, 5, 10],
 'n_estimators': [100, 200, 300]}

grid_search = GridSearchCV(
    estimator=estimator,
    param_grid=parameters,
    scoring = 'roc_auc',
    n_jobs = 10,
    cv = 4,
    verbose=True
)

DESCRIPTION = 'randomforest-grid-search'

# COMMAND ----------

c = MLExperiment(ml_data, preprocessor=preprocessor, estimator=grid_search, transformer_col_function=get_column_names_transformed, baseline_auc=.75)

# COMMAND ----------

c.run_and_log_train_and_val()

# COMMAND ----------

# MAGIC %md
# MAGIC #### Logistic Regression Grid Search

# COMMAND ----------

estimator = LogisticRegression()

parameters = {"C":np.logspace(-3,3,7), 
              "penalty":["l1","l2"]}

grid_search = GridSearchCV(
    estimator=estimator,
    param_grid=parameters,
    scoring = 'roc_auc',
    n_jobs = 10,
    cv = 4,
    verbose=True
)

DESCRIPTION = 'logistic-regression-grid-search'

# COMMAND ----------

# do ml experiment
b = MLExperiment(ml_data, preprocessor=preprocessor, estimator=grid_search, transformer_col_function=get_column_names_transformed, baseline_auc=.75)

# COMMAND ----------

b.run_and_log_train_and_val()

# COMMAND ----------

b.fit_preprocessor()

# COMMAND ----------

b.fit_estimator()

# COMMAND ----------

b.auc_score_train

# COMMAND ----------

b.auc_score_val

# COMMAND ----------



# COMMAND ----------



# COMMAND ----------

# MAGIC %md
# MAGIC ##### OR run each step seperately

# COMMAND ----------

ml_exp1.fit_preprocessor()

# COMMAND ----------

ml_exp1.fit_estimator()

# COMMAND ----------

# MAGIC %md
# MAGIC ##### Access the estimator and preprocessor explicitly

# COMMAND ----------

ml_exp1.estimator

# COMMAND ----------

ml_exp1.preprocessor

# COMMAND ----------


