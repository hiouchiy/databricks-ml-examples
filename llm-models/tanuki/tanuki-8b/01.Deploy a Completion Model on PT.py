# Databricks notebook source
# MAGIC %md
# MAGIC Cluster: more than A100x1

# COMMAND ----------

# MAGIC %md
# MAGIC ##Install additional libraries

# COMMAND ----------

# MAGIC %pip install -U hf_transfer mlflow==2.21.3 transformers==4.48.1 torch==2.5.1 torchvision==0.20.1
# MAGIC dbutils.library.restartPython()

# COMMAND ----------

# MAGIC %md
# MAGIC ##Set parameters

# COMMAND ----------

import os
os.environ["HF_HUB_ENABLE_HF_TRANSFER"] = "1"

# model repo name on HuggingFace
HF_REPO_ID = "weblab-GENIAC/Tanuki-8B-dpo-v1.0"

MODEL_PATH = "/Volumes/hiroshi/models/models_from_hf/tanuki-8b-dpo-v1-0"

# for UC
catalog = "hiroshi"
schema = "models"
model_name = "tanuki-8b-dpo-v1-0"
registered_model_name = f"{catalog}.{schema}.{model_name}"

# Endpoint name
endpoint_name = f'{model_name}-endpoint-pt-completion'

# COMMAND ----------

# MAGIC %md
# MAGIC ## Download model and tokenizer prior to load them.

# COMMAND ----------

from huggingface_hub import snapshot_download

snapshot_download(
  repo_id=HF_REPO_ID,
  local_dir="/local_disk0/hf")

!mkdir -p {MODEL_PATH}
!cp -L -R /local_disk0/hf/* {MODEL_PATH}

# COMMAND ----------

# MAGIC %md
# MAGIC ## Load a LLM model and a tokenizer

# COMMAND ----------

from transformers import AutoModelForCausalLM, AutoTokenizer

model = AutoModelForCausalLM.from_pretrained(
  MODEL_PATH, 
  device_map="auto", 
  torch_dtype="auto")

tokenizer = AutoTokenizer.from_pretrained(MODEL_PATH)

# COMMAND ----------

# MAGIC %md
# MAGIC ## Run inference with a Completion-style prompt

# COMMAND ----------

messages = [
    {"role": "system", "content": "あなたは誠実で優秀な日本人のアシスタントです。"},
    {
        "role": "user",
        "content": "ただ、「はい」とだけ答えて下さい。",
    },
]

prompt = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
print(prompt)

# COMMAND ----------

inputs = tokenizer(prompt, return_tensors="pt").to(model.device)
output_ids = model.generate(inputs['input_ids'],
                            max_new_tokens=512,
                            temperature=0.1,
                            do_sample=True)
output = tokenizer.decode(output_ids[0], skip_special_tokens=True)
print(output)


# COMMAND ----------

# MAGIC %md
# MAGIC ## Build a pipeline and run inference again

# COMMAND ----------

from transformers import pipeline

generator = pipeline(
    "text-generation",
    tokenizer=tokenizer,
    model=model,
)
generator(prompt, max_new_tokens=100)

# COMMAND ----------

# MAGIC %md
# MAGIC ## Log and register the model

# COMMAND ----------

import numpy as np
from transformers import pipeline
import mlflow

mlflow.set_registry_uri("databricks-uc")

with mlflow.start_run(run_name=model_name) as run:
    model_info = mlflow.transformers.log_model(
        # transformers_model=generator,
        transformers_model={
            "model": model,
            "tokenizer": tokenizer,
        },
        artifact_path="model",
        input_example={"prompt": prompt},
        task="llm/v1/completions",
        registered_model_name=registered_model_name,
    )

# COMMAND ----------

from mlflow import MlflowClient
client = MlflowClient()

def get_latest_model_version(model_name):
  client = MlflowClient()
  model_version_infos = client.search_model_versions("name = '%s'" % model_name)
  return max([int(model_version_info.version) for model_version_info in model_version_infos])

latest_version = get_latest_model_version(model_name=registered_model_name)

client.set_registered_model_alias(name=registered_model_name, alias="Champion", version=latest_version)

# COMMAND ----------

# MAGIC %md
# MAGIC ## Load the model from UC and run inference

# COMMAND ----------

model_champion_uri = "models:/{model_name}@Champion".format(model_name=registered_model_name)
 
print("Loading registered model version from URI: '{model_uri}'".format(model_uri=model_champion_uri))
champion_model = mlflow.pyfunc.load_model(model_champion_uri)

# COMMAND ----------

champion_model.predict(
  {
    "prompt": prompt, 
    "max_tokens": 1024, 
    "temperature": 0.9, 
    "do_sample": True, 
    "max_new_tokens": 1024,
  }
)

# COMMAND ----------

# MAGIC %md
# MAGIC ## Deploy the model

# COMMAND ----------

databricks_url = dbutils.notebook.entry_point.getDbutils().notebook().getContext().apiUrl().getOrElse(None)
token = dbutils.notebook.entry_point.getDbutils().notebook().getContext().apiToken().getOrElse(None)

# COMMAND ----------

from databricks.sdk import WorkspaceClient
from databricks.sdk.service.serving import EndpointCoreConfigInput, ServedModelInput, ServedModelInputWorkloadSize, ServedModelInputWorkloadType
from datetime import timedelta

import requests
import json

# Name of the registered MLflow model
model_name = registered_model_name

# Get the latest version of the MLflow model
model_version = latest_version

headers = {"Context-Type": "text/json", "Authorization": f"Bearer {token}"}

response = requests.get(url=f"{databricks_url}/api/2.0/serving-endpoints/get-model-optimization-info/{model_name}/{model_version}", headers=headers)

print(json.dumps(response.json(), indent=4))

# COMMAND ----------

from mlflow.deployments import get_deploy_client

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

from datetime import timedelta
from databricks.sdk import WorkspaceClient
from databricks.sdk.service.serving import EndpointCoreConfigInput, ServedModelInput, ServedModelInputWorkloadSize, ServedModelInputWorkloadType

w = WorkspaceClient()
w.serving_endpoints.wait_get_serving_endpoint_not_updating(name=endpoint_name, timeout=timedelta(minutes=60))

# COMMAND ----------

# MAGIC %md
# MAGIC ## Test the endpoint with REST

# COMMAND ----------

import requests
import json

data = {
  "prompt": prompt,
  "temperature": 0.1,
  "max_tokens": 1000,
}

databricks_host = dbutils.notebook.entry_point.getDbutils().notebook().getContext().apiUrl().get()
databricks_token = dbutils.notebook.entry_point.getDbutils().notebook().getContext().apiToken().get()
headers = {"Context-Type": "text/json", "Authorization": f"Bearer {databricks_token}"}

response = requests.post(
    url=f"{databricks_host}/serving-endpoints/{endpoint_name}/invocations", json=data, headers=headers
)

print(response.json())
# print(response.json()["choices"][0]["message"]["content"])

# COMMAND ----------

# MAGIC %md
# MAGIC ## Test the endpoint with OpenAI client

# COMMAND ----------

# MAGIC %pip install openai
# MAGIC %pip install -U typing_extensions
# MAGIC dbutils.library.restartPython()

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
    model="hiroshi-stockmark-13b-instruct-completion",
    prompt=["### Input:\nアメリカの首都は？\n\n### Output:\n"],
    max_tokens=200,
    temperature=0.1,
    stream=False
)

print("【回答】")
print(response.choices[0].text)
# for chunk in response:
#     # print(chunk)
#     print(chunk.choices[0].delta.content, end="")
#     # print("****************")

# COMMAND ----------

# MAGIC %md
# MAGIC ---
