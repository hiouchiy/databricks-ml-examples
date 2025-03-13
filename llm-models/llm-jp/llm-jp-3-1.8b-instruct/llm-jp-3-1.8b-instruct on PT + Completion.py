# Databricks notebook source
# MAGIC %md
# MAGIC Runtime version: 15.4 ML GPU
# MAGIC
# MAGIC Instance type: Standard_NC24ads_A100_v4(Single node)以上

# COMMAND ----------

# MAGIC %md
# MAGIC ## 追加ライブラリインストール

# COMMAND ----------

# MAGIC %pip install hf_transfer
# MAGIC %pip install -U mlflow==2.19.0 threadpoolctl==3.5.0
# MAGIC dbutils.library.restartPython()

# COMMAND ----------

# MAGIC %md
# MAGIC ## モデルのダウンロードを高速化するオプション

# COMMAND ----------

import os
os.environ["HF_HUB_ENABLE_HF_TRANSFER"] = "1"

# COMMAND ----------

HF_REPO_ID = "llm-jp/llm-jp-3-1.8b-instruct"
MODEL_PATH = "/Volumes/hiroshi/models/models_from_hf/llm-jp-3-1-8b-instruct"
registered_model_name = "hiroshi.models.llm-jp-3-1-8b-instruct"
endpoint_name = 'hiroshi-llm-jp-3-1-8b-completion-pt'

# COMMAND ----------

# MAGIC %md
# MAGIC ## モデルをダウンロードし、UC Volumeに保存

# COMMAND ----------

from huggingface_hub import snapshot_download

snapshot_download(
  repo_id=HF_REPO_ID,
  local_dir="/local_disk0/hf")

!mkdir -p {MODEL_PATH}
!cp -L -R /local_disk0/hf/* {MODEL_PATH}

# COMMAND ----------

# MAGIC %md
# MAGIC ## Load a LLM model

# COMMAND ----------

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

model = AutoModelForCausalLM.from_pretrained(
  MODEL_PATH, 
  device_map="auto", 
  torch_dtype=torch.bfloat16, 
  rope_theta=10000.0 # Special setting just for LLM-JP-3
)

# COMMAND ----------

# MAGIC %md
# MAGIC ## Load a tokenizer with custom "chat_template" and run inference with a ChatCompletion-style prompt

# COMMAND ----------

# MAGIC %md
# MAGIC ## Build a pipeline and run inference again to check ChatCompletion prompt works correctly

# COMMAND ----------

tokenizer = AutoTokenizer.from_pretrained(
  MODEL_PATH,
)

instruction = "日本の首都はどこ？"

messages = [
    {"role": "user", "content": instruction},
]

prompt = tokenizer.apply_chat_template(messages, add_generation_prompt=True, tokenize=False)
print(prompt)
input_ids = tokenizer.encode(prompt, add_special_tokens=False, return_tensors="pt").to(model.device)
output_ids = model.generate(input_ids,
                            max_new_tokens=512,
                            temperature=0.1,
                            do_sample=True)
output = tokenizer.decode(output_ids[0], skip_special_tokens=False)
print(output)


# COMMAND ----------

from transformers import pipeline
from mlflow.types.llm import ChatCompletionResponse, ChatMessage, ChatParams, ChatChoice

tokenizer_without_chat_template = AutoTokenizer.from_pretrained(
    MODEL_PATH,
    chat_template=None # Disable custom chat_template as this will be completion task.
)

generator = pipeline(
    "text-generation",
    tokenizer=tokenizer_without_chat_template,
    model=model,
)
answer = generator(prompt, max_new_tokens=100)
# print(answer)
assistant_message = answer[0]["generated_text"]
print(assistant_message[len(prompt):])

# COMMAND ----------

import mlflow
from mlflow.models import infer_signature

# Define model signature including params
input_example = {"prompt": prompt}

# Log the model with its details such as artifacts, pip requirements and input example
# This may take about 1.7 minutes to complete
with mlflow.start_run(run_name="llm-jp-3-1.8b-instruct") as run:
    mlflow.transformers.log_model(
        transformers_model={
            "model": model,
            "tokenizer": tokenizer_without_chat_template,
        },
        task="llm/v1/completions",
        artifact_path="model",
        pip_requirements=[
            "threadpoolctl==3.5.0",
            "hf_transfer"
        ],
        input_example=input_example,
        registered_model_name=registered_model_name,
    )

# COMMAND ----------

from mlflow import MlflowClient
client = MlflowClient()

# from mlflow.tracking.client import MlflowClient
def get_latest_model_version(model_name):
  client = MlflowClient()
  model_version_infos = client.search_model_versions("name = '%s'" % model_name)
  return max([int(model_version_info.version) for model_version_info in model_version_infos])

latest_version = get_latest_model_version(model_name=registered_model_name)

# 上記のセルに登録されている正しいモデルバージョンを選択
client.set_registered_model_alias(name=registered_model_name, alias="Champion", version=latest_version)

# COMMAND ----------

import mlflow
model_champion_uri = "models:/{model_name}@Champion".format(model_name=registered_model_name)
 
print("Loading registered model version from URI: '{model_uri}'".format(model_uri=model_champion_uri))
champion_model = mlflow.pyfunc.load_model(model_champion_uri)

# COMMAND ----------

input_example = {"prompt": prompt}
champion_model.predict(input_example, params={"max_new_tokens": 100})

# COMMAND ----------

# サービングエンドポイントの作成または更新
from mlflow.deployments import get_deploy_client
from datetime import timedelta

import requests
import json

# Name of the registered MLflow model
model_name = registered_model_name

# Get the latest version of the MLflow model
model_version = latest_version

databricks_url = dbutils.notebook.entry_point.getDbutils().notebook().getContext().apiUrl().getOrElse(None)
token = dbutils.notebook.entry_point.getDbutils().notebook().getContext().apiToken().getOrElse(None)

headers = {"Context-Type": "text/json", "Authorization": f"Bearer {token}"}

response = requests.get(url=f"{databricks_url}/api/2.0/serving-endpoints/get-model-optimization-info/{model_name}/{model_version}", headers=headers)

print(json.dumps(response.json(), indent=4))

# COMMAND ----------

client = get_deploy_client("databricks")

endpoint = client.create_endpoint(
    name=endpoint_name,
    config={
        "served_entities": [
            {
                "entity_name": model_name,
                "entity_version": model_version,
                "min_provisioned_throughput": response.json()['throughput_chunk_size'],
                "max_provisioned_throughput": response.json()['throughput_chunk_size'],
            }
        ]
    },
)

print(json.dumps(endpoint, indent=4))

# COMMAND ----------

from databricks.sdk import WorkspaceClient
w = WorkspaceClient()
w.serving_endpoints.wait_get_serving_endpoint_not_updating(endpoint.name, timeout=timedelta(minutes=10))

# COMMAND ----------

import requests
import json

data = {
  "prompt": prompt,
  "temperature": 0.1,
  "max_tokens": 100,
  "presence_penalty": 0,
  "stop": "で"
}

databricks_host = dbutils.notebook.entry_point.getDbutils().notebook().getContext().apiUrl().get()
databricks_token = dbutils.notebook.entry_point.getDbutils().notebook().getContext().apiToken().get()
headers = {"Context-Type": "text/json", "Authorization": f"Bearer {databricks_token}"}

response = requests.post(
    url=f"{databricks_host}/serving-endpoints/{endpoint_name}/invocations", json=data, headers=headers
)

print(response.json())
# print(response.json()["choices"][0]["text"])

# COMMAND ----------

# MAGIC %md
# MAGIC ---

# COMMAND ----------

# MAGIC %pip install openai

# COMMAND ----------

import os
import openai
from openai import OpenAI

API_ROOT = dbutils.notebook.entry_point.getDbutils().notebook().getContext().apiUrl().get() 
API_TOKEN = dbutils.notebook.entry_point.getDbutils().notebook().getContext().apiToken().get()

client = OpenAI(
    api_key=API_TOKEN,
    base_url=f"{API_ROOT}/serving-endpoints"
)

response = client.completions.create(
    model=endpoint_name,
    prompt=prompt,
    max_tokens=10,
    temperature=0.1,
    presence_penalty=0
)

print("【回答】")
print(response.choices[0].text)

# COMMAND ----------


