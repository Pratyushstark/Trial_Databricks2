# Databricks notebook source
# notebooks/04_promote_to_staging.py

import mlflow
from mlflow.tracking import MlflowClient

# ----------------------------
# Environment
# ----------------------------
dbutils.widgets.text("env", "dev")
ENV = dbutils.widgets.get("env")

CATALOG = f"mlops_{ENV}"
SCHEMA = "raw"

# ----------------------------
# Config
# ----------------------------
DEV_EXPERIMENT = "/Shared/mlops_dev"
STAG_EXPERIMENT = "/Shared/mlops_stag"
MODEL_NAME = f"{CATALOG}.{SCHEMA}.fraud_model"
ARTIFACT_PATH = "classifier_pipeline"  # sklearn pipeline artifact
TAGS = {"team": "mlops", "project": "fraud_classification"}


client = MlflowClient()

# ----------------------------
# Get latest DEV run
# ----------------------------
# Let's get our best ml run
runs = mlflow.search_runs(
    experiment_names=[DEV_EXPERIMENT],
    order_by=["metrics.f1_score DESC"],
    max_results=1,
  # filter_string="status = 'FINISHED' and run_name='sklearn_baseline_logreg'" #filter on mlops_best_run to always use the notebook 02 to have a more predictable demo
)

if runs.empty:
    raise RuntimeError("No DEV runs found")

run_id = runs.iloc[0].run_id
model_uri = f"runs:/{run_id}/{ARTIFACT_PATH}"

# ----------------------------
# Register model
# ----------------------------
model_version = mlflow.register_model(
    model_uri=model_uri,
    name=MODEL_NAME,
    tags = TAGS
)

print(f"Registered model version: {model_version.version}")

# ----------------------------
# Transition to STAGING
# ----------------------------
client.set_registered_model_alias(
    name=MODEL_NAME,
    alias="staging",
    version= model_version.version
)

print(f"✅ Alias 'latest-model' set to version {model_version.version}")

# ----------------------------
# Setting F1 Score as a tag
# ----------------------------
# Provide more details on this specific model version
best_score = runs['metrics.f1_score'].values[0]
run_name = runs['tags.mlflow.runName'].values[0]

# We can also tag the model version with the F1 score for visibility. This will add f1 score as a tag
client.set_model_version_tag(
  name=model_version.name,
  version=model_version.version,
  key="f1_score",
  value=f"{round(best_score,4)}"
)