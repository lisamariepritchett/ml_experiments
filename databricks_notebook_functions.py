# Databricks notebook source
import pandas as pd
from time import time
import matplotlib.pyplot as plt


# COMMAND ----------

pd.set_option('display.max_rows', 500)
pd.set_option("display.max_columns", 80)
pd.set_option("max_colwidth", 80)
pd.set_option('expand_frame_repr', False)


plt.style.use('ggplot')
plt.rcParams["font.size"] = 15
plt.rcParams["figure.figsize"] = [9, 5]


# COMMAND ----------


def pull_data(query):
     
    # cnxn = library_common_get_presto_connection()
    cnxn = get_connection()
    # Pull data
         
    print("Starting the data query...")
    all_data = psql.read_sql(query, cnxn)
    print("Data query done, there are " + str(len(all_data)) + " observations")

    # Close the connection to presto
    cnxn.close()
    
    return all_data
  
  

# COMMAND ----------


def shorten_col_names_remove_view_dot(df):
  short_names = []
  for col in df.columns:
    try:
      short_name = col.split('.')[1]
    except IndexError:
      short_name = col
    finally:
      short_names.append(short_name)
  
  print(short_names)
  df.columns = short_names
  return df



# COMMAND ----------

def create_or_replace_databricks_table(sdf, table_name, schema_name = 'lisapritchett'):
  sdf.write.mode("overwrite").option("overwriteSchema", "true").saveAsTable("{}.{}".format(schema_name, table_name))
  print("Data saved to databricks table {}.{}".format(schema_name, table_name))


def get_databricks_table(table_name = "df"):
    return spark.read.table(table_name)


# COMMAND ----------

def barplot(sdf, x, y):
  pdf = sdf.select(x, y).toPandas()
  sns.barplot(data=pdf, x=x, y=y)

# COMMAND ----------

def read_databricks_table(table, schema = 'lisapritchett'):
  return spark.sql("SELECT * FROM {schema}.{table}".format(schema=schema, table=table))

# COMMAND ----------

def check_sdf_for_primary_key(sdf, pkey):
  count_rows = sdf.count()
  count_distinct_pkey = sdf.select(pkey).distinct().count()
  print("total rows in spark df: ", count_rows)
  print("distinct pkey values: ", count_distinct_pkey)
  try:
    assert(count_rows /  count_distinct_pkey == 1)
    print("Yes, {} Is Confirmed to be Primary Key".format(pkey))
  except AssertionError:
    print("No, {} is not a Primary Key".format(pkey))

# COMMAND ----------

def plot_n_and_p_sdf_col(sdf, col = 'gender',  order = 'n', criterion = 'app_started', max_y = .04, type = 'bar'):
  p_name = 'p_' + criterion
  n_name = 'n_' + criterion
  
  if order == 'p':
    order = p_name
  elif order == 'n':
    order = 'n'    
    
  pd_agg = sdf.groupby(col) \
    .agg(f.count(f.col(criterion)).alias('n')
         , f.count(f.when(f.col(criterion), 1)).alias(n_name)  
        )\
    .withColumn(p_name, f.col(n_name) / f.col('n')) \
    .orderBy(f.col(order)) \
    .toPandas()
  
  plt.figure()
  if type == 'line':
    pd_agg.plot.line(x=col, y='n', title=col+' N', logy=True)
  else:
    pd_agg.plot.bar(x=col, y='n', title=col+' N', logy=True)
  plt.show()
  plt.figure()
  
  if type == 'line':
    pd_agg.plot.line(x=col, y=p_name, title=col + ' P', color='m')
  else:
    pd_agg.plot.bar(x=col, y=p_name, title=col + ' P', color='m')
  plt.ylim(0,max_y)
  plt.show()


# COMMAND ----------

def get_n_and_p(sdf, col = 'gender',  order = 'n', criterion = 'app_started'):
  p_name = 'p_' + criterion
  n_name = 'n_' + criterion
  
  if order == 'p':
    order = p_name
  elif order == 'n':
    order = 'n'    
    
  sdf_agg = sdf.groupby(col) \
    .agg(f.count(f.col(criterion)).alias('n')
         , f.count(f.when(f.col(criterion), 1)).alias(n_name)  
        )\
    .withColumn(p_name, f.round(f.col(n_name) / f.col('n'), 5)) \
    .orderBy(f.col(order)) 
  return sdf_agg


# COMMAND ----------

def plot_value_labels(df, x, y, y_offset = .01):
  
  x_series = df[x]
  y_series = df[y]
  
  for x, y in zip(x_series, y_series):
    # the position of the data label relative to the data point can be adjusted by adding/subtracting a value from the x &/ y coordinates
    plt.text(x = x, # x-coordinate position of data label
      y = y+y_offset, # y-coordinate position of data label, adjusted to be 150 below the data point
      s = '{:.2f}'.format(y), # data label, formatted to 2 decimals
      ) 

# COMMAND ----------


