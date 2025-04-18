# Databricks notebook source
# MAGIC %md
# MAGIC Cluster: Serverless

# COMMAND ----------

# MAGIC %md
# MAGIC ## Install addional libraries

# COMMAND ----------

# MAGIC %pip install -U -qqqq databricks-agents==0.19.0 mlflow==2.21.3 databricks-openai==0.3.1 transformers==4.48.1 torch==2.5.1 uv
# MAGIC dbutils.library.restartPython()

# COMMAND ----------

# MAGIC %md
# MAGIC ##Load the agent class by running the notebook

# COMMAND ----------

# MAGIC %run ./chat_model_wrapper

# COMMAND ----------

# MAGIC %md
# MAGIC ## Set parameters

# COMMAND ----------

# TODO: define the catalog, schema, and model name for your UC model
catalog = "hiroshi"
schema = "models"
model_name = "my-chat-wrapper"
UC_MODEL_NAME = f"{catalog}.{schema}.{model_name}"

# COMMAND ----------

# MAGIC %md
# MAGIC ##Create a config file

# COMMAND ----------

import yaml

config_file = {
      "llm_endpoint_name": "tanuki-8b-dpo-v1-0-endpoint-pt-completion",
      "hf_model_name": "weblab-GENIAC/Tanuki-8B-dpo-v1.0"
}
config_file_name = 'cpu_endpoint_config.yaml'
try:
    with open(config_file_name, 'w') as f:
        yaml.dump(config_file, f)
except:
    print('pass to work on build job')

# COMMAND ----------

# MAGIC %md
# MAGIC ## Test the agent class on local

# COMMAND ----------

API_ROOT = dbutils.notebook.entry_point.getDbutils().notebook().getContext().apiUrl().get()
os.environ["DATABRICKS_HOST"] = API_ROOT
API_TOKEN = dbutils.notebook.entry_point.getDbutils().notebook().getContext().apiToken().get()
os.environ["DATABRICKS_TOKEN"] = API_TOKEN

input_example = [{"role": "user", "content": "日本の首都は京都でしたっけ？"}]

chat_model = MyChatCompletionWrapper()
chat_model.load_context(None)

# Normal predict
response = chat_model.predict(None, messages=input_example, params=ChatParams(temperature=0.1, max_tokens=50))
print(response.to_dict())

# Streaming predict
for event in chat_model.predict_stream(context=None, messages=input_example, params=ChatParams(temperature=0.1, max_tokens=50)):
    print(event.choices[0].delta.content)

# COMMAND ----------

# MAGIC %md
# MAGIC ## Set full path of wrapper notebook and config file

# COMMAND ----------

import os

# Specify the full path to the wrapper notebook
wrapper_notebook_path = os.path.join(os.getcwd(), "chat_model_wrapper")
print(f"Wrapper notebook path: {wrapper_notebook_path}")

# Specify the full path to the config file (.yaml)
config_file_path = os.path.join(os.getcwd(), "cpu_endpoint_config.yaml")
print(f"Config file path: {config_file_path}")

# COMMAND ----------

# MAGIC %md
# MAGIC ## Log the wrapper to MLFlow

# COMMAND ----------

import os
import mlflow
from mlflow.models.resources import DatabricksServingEndpoint

mlflow.set_registry_uri("databricks-uc")

with mlflow.start_run():
    logged_agent_info = mlflow.pyfunc.log_model(
        python_model=wrapper_notebook_path,
        model_config=config_file_path,
        artifact_path="chat_model_wrapper",
        resources=[DatabricksServingEndpoint(endpoint_name=config_file["llm_endpoint_name"])],
        registered_model_name=UC_MODEL_NAME
    )

# COMMAND ----------

# MAGIC %md
# MAGIC Pre-deployment agent validation
# MAGIC Before registering and deploying the agent, perform pre-deployment checks using the mlflow.models.predict() API. See Databricks documentation (AWS | Azure).

# COMMAND ----------

mlflow.models.predict(
    model_uri=f"runs:/{logged_agent_info.run_id}/chat_model_wrapper",
    input_data={"messages": [{"role": "user", "content": "Hello!"}]},
    env_manager="uv",
)

# COMMAND ----------

# MAGIC %md
# MAGIC ## Set alias to the registered wrapper

# COMMAND ----------

from mlflow import MlflowClient
client = MlflowClient()

# from mlflow.tracking.client import MlflowClient
def get_latest_model_version(model_name):
  client = MlflowClient()
  model_version_infos = client.search_model_versions("name = '%s'" % model_name)
  return max([int(model_version_info.version) for model_version_info in model_version_infos])

latest_version = get_latest_model_version(model_name=UC_MODEL_NAME)

# 上記のセルに登録されている正しいモデルバージョンを選択
client.set_registered_model_alias(name=UC_MODEL_NAME, alias="Champion", version=latest_version)

# COMMAND ----------

# MAGIC %md
# MAGIC ##Test the registered wrapper

# COMMAND ----------

registered_wrapper = mlflow.pyfunc.load_model(f"models:/{UC_MODEL_NAME}/{latest_version}")

registered_wrapper.predict(
  {
    "messages": input_example, 
    "temperature": 0.1, 
    "max_tokens": 10
  }
)

# COMMAND ----------

for chunk in registered_wrapper.predict_stream(
    {
    "messages": input_example, 
    "temperature": 0.1, 
    "max_tokens": 100
  }
):
    if not chunk["choices"] or not chunk["choices"][0]["delta"]["content"]:
        continue

    print(chunk["choices"][0]["delta"]["content"])


# COMMAND ----------

# MAGIC %md
# MAGIC ## Deploy the wrapper uging Mosaic AI Agent Framework

# COMMAND ----------

import os
import mlflow
from databricks import agents

deployment_info = agents.deploy(
    UC_MODEL_NAME, 
    latest_version, 
    environment_vars={
        "DATABRICKS_TOKEN": "{{secrets/hiouchiy/databricks_token}}"
    })

browser_url = mlflow.utils.databricks_utils.get_browser_hostname()
print(f"\n\nView deployment status: https://{browser_url}/ml/endpoints/{deployment_info.endpoint_name}")

review_instructions = """### My ChatCompletion Wrapper テスト

1. **多様な質問をお試しください**：
   - 実際のお客様が尋ねると予想される多様な質問を入力ください。これは、予想される質問を効果的に処理できるか否かを確認するのに役立ちます。

チャットボットの評価にお時間を割いていただき、ありがとうございます。エンドユーザーに高品質の製品をお届けするためには、皆様のご協力が不可欠です。"""

agents.set_review_instructions(UC_MODEL_NAME, review_instructions)

# COMMAND ----------

from datetime import timedelta
from databricks.sdk import WorkspaceClient
from databricks.sdk.service.serving import EndpointCoreConfigInput, ServedModelInput, ServedModelInputWorkloadSize, ServedModelInputWorkloadType

w = WorkspaceClient()
w.serving_endpoints.wait_get_serving_endpoint_not_updating(name=deployment_info.endpoint_name, timeout=timedelta(minutes=60))

# COMMAND ----------


