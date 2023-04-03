from pyspark.sql import SparkSession
from pyspark.sql.types import DateType, FloatType, IntegerType, BooleanType, TimestampType
from pyspark.sql.functions import col,isnan, when, count, monotonically_increasing_id, udf

# COMMAND ----------

import numpy as np
import pandas as pd

import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats
import plotly.express as px
import plotly.figure_factory as ff
from plotly.subplots import make_subplots
import plotly.graph_objs as go

from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from imblearn.over_sampling import SMOTE
from collections import Counter

from sklearn.ensemble import RandomForestClassifier
from lightgbm.sklearn import LGBMClassifier
import xgboost as xgb
from sklearn.svm import SVC
from sklearn.linear_model import LogisticRegression

from sklearn.metrics import accuracy_score
from sklearn.metrics import roc_auc_score
from sklearn.metrics import classification_report
from sklearn.metrics import roc_curve, RocCurveDisplay, average_precision_score,accuracy_score,f1_score,precision_recall_curve,confusion_matrix

import mlflow
from mlflow.tracking import MlflowClient

spark = SparkSession.builder.appName('AnomalyDetection').getOrCreate()

df = spark.read.format("csv").option("header", "true").load("dbfs:/FileStore/tables/financial_payment.csv")

df.show()

# COMMAND ----------

df = df.withColumnRenamed("oldbalanceOrg", "oldBalanceOrig") \
    .withColumnRenamed("oldbalanceDest", "oldBalanceDest") \
    .withColumnRenamed("newbalanceOrig", "newBalanceOrig") \
    .withColumnRenamed("newbalanceDest", "newBalanceDest")

# COMMAND ----------

df = df.withColumn("amount", df["amount"].cast(FloatType())) \
    .withColumn("oldBalanceOrig", df["oldBalanceOrig"].cast(FloatType())) \
    .withColumn("newBalanceOrig", df["newBalanceOrig"].cast(FloatType())) \
    .withColumn("oldBalanceDest", df["oldBalanceDest"].cast(FloatType())) \
    .withColumn("newBalanceDest", df["newBalanceDest"].cast(FloatType())) \
    .withColumn("isFraud", df["isFraud"].cast(IntegerType())) \
    .withColumn("isFlaggedFraud", df["isFlaggedFraud"].cast(IntegerType()))

# COMMAND ----------

df.select([count(when(isnan(c) | col(c).isNull(), c)).alias(c) for c in df.columns]
          ).show()

# COMMAND ----------

df = df.toPandas()

# COMMAND ----------

df.step.value_counts()

# COMMAND ----------

df.dtypes

# COMMAND ----------

df['isFraud'].value_counts()

# COMMAND ----------

df.nunique()


from sklearn.preprocessing import LabelEncoder
encoder = {}
for i in df.select_dtypes('object').columns:
    encoder[i] = LabelEncoder()
    df[i] = encoder[i].fit_transform(df[i])

# COMMAND ----------

x = df.drop(columns=['isFraud'])
y = df['isFraud']

# COMMAND ----------

over_sample = SMOTE(random_state=0)
x,y = over_sample.fit_resample(x,y)

# COMMAND ----------

y.value_counts()

# COMMAND ----------

from sklearn.preprocessing import MinMaxScaler
scaler = MinMaxScaler()
x = scaler.fit_transform(x)

# COMMAND ----------

x = df[['step', 'type', 'amount', 'oldBalanceOrig', 'newBalanceOrig','nameDest', 'oldBalanceDest', 'isFlaggedFraud']]
y= df['isFraud']

from sklearn.model_selection import train_test_split

x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.20, random_state=0)

evalset = [(x_train, y_train), (x_test, y_test)]
import xgboost as xg
from xgboost import plot_importance
from mlflow.models.signature import infer_signature
from sklearn.metrics import accuracy_score
from sklearn.metrics import roc_auc_score , precision_score, recall_score
from sklearn.metrics import classification_report
from sklearn.metrics import f1_score,confusion_matrix, ConfusionMatrixDisplay
from sklearn import metrics

mlflow.autolog(disable=True)

with mlflow.start_run(run_name="fraud_prediction_XGB") as run2:
    max_depth = 7
    xgb_r8 = xg.XGBClassifier(max_depth=max_depth)
    xgb_r8.fit(x_train, y_train, eval_metric='logloss', eval_set=evalset)
    xgb_r8.score(x_test, y_test)
    xgb_r8.score(x_train, y_train)

    y_pred = xgb_r8.predict(x_test)
    print(classification_report(y_test, y_pred))
    c_m = confusion_matrix(y_test, y_pred)

    test_accuracy = accuracy_score(y_test, y_pred)
    test_precision_score = precision_score(y_test, y_pred)
    test_recall_score = recall_score(y_test, y_pred)
    test_f1_score = f1_score(y_test, y_pred)
    auc_score = roc_auc_score(y_test, y_pred)

    fig = plt.figure(figsize=(14, 9))
    ax = fig.add_subplot(111)

    colours = plt.cm.Set1(np.linspace(0, 1, 9))

    ax = plot_importance(xgb_r8, height=1, color=colours, grid=False, \
                         show_values=False, importance_type='cover', ax=ax);
    for axis in ['top', 'bottom', 'left', 'right']:
        ax.spines[axis].set_linewidth(2)

    ax.set_xlabel('importance score', size=16);
    ax.set_ylabel('features', size=16);
    ax.set_yticklabels(ax.get_yticklabels(), size=12);
    ax.set_title('Ordering of features by importance to the model learnt', size=20);

    metrics = {
        'Test_accuracy': test_accuracy,
        'Test_precision_score': test_precision_score,
        'Test_recall_score': test_recall_score,
        'Test_f1_score': test_f1_score,
        'AUC_score': auc_score
    }

    cm_display = confusion_matrix(y_test, y_pred)
    cm_display = ConfusionMatrixDisplay(cm_display).plot()
    cm_xgb = plt.gcf()

    mlflow.log_figure(cm_xgb, "Confusion-Matrix-XGB.png")
    mlflow.log_figure(fig, "FeatureImportance.png")
    mlflow.log_param('Max Depth', max_depth)
    mlflow.log_metrics(metrics)

    mlflow.set_tag('FPrediction-XGB', 'fraud_Prediction')

    signature = infer_signature(x_train, xgb_r8.predict(x_train))

    mlflow.sklearn.log_model(xgb_r8, 'FraudPrediction-XGB',
                             registered_model_name='fraud_predict_model',
                             signature=signature)
    # Save the model as an MLflow artifact
    # mlflow.sklearn.log_model(log_reg, "FraudPrediction-XGB")
    run = mlflow.active_run()

    print('Active run_id: {}'.format(run.info.run_id))

mlflow.end_run()
