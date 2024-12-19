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
# MAGIC %pip install -U mlflow==2.19.0 transformers==4.45.2 accelerate==0.31.0 vllm==0.6.5
# MAGIC dbutils.library.restartPython()

# COMMAND ----------

# MAGIC %md
# MAGIC ## モデルのダウンロードを高速化するオプション

# COMMAND ----------

import os
os.environ["HF_HUB_ENABLE_HF_TRANSFER"] = "1"

# COMMAND ----------

# MAGIC %md
# MAGIC ## モデルをダウンロードし、UC Volumeに保存

# COMMAND ----------

from huggingface_hub import snapshot_download

snapshot_download(
  repo_id="stockmark/stockmark-13b-instruct",
  local_dir="/local_disk0/hf")

!cp -L -R /local_disk0/hf/* /Volumes/hiroshi/models/stockmark-13b-instruct-vllm

# COMMAND ----------

# MAGIC %md
# MAGIC ## Load a LLM model

# COMMAND ----------

MODEL_PATH = "/Volumes/hiroshi/models/stockmark-13b-instruct-vllm"
registered_model_name = "hiroshi.models.stockmark-13b-instruct-vllm"

# COMMAND ----------

import torch
from vllm import LLM, SamplingParams

# For generative models (task=generate) only
llm = LLM(model=MODEL_PATH, task="generate", tensor_parallel_size=torch.cuda.device_count())

sampling_params = SamplingParams(
  temperature=0.5, 
  top_p=0.8, 
  top_k=5, 
  max_tokens=200, 
  presence_penalty=1.1)

# COMMAND ----------

# MAGIC %md
# MAGIC ## Run inference
# MAGIC Set custom "chat_template" and run inference with a ChatCompletion-style prompt

# COMMAND ----------

messages = [
    {"role": "user", "content": "日本の首都はどこ？"},
]
output = llm.chat(
    messages, 
    sampling_params=sampling_params, 
    chat_template="{% if messages[0]['role'] == 'system' %}{% set system_message = messages[0]['content'] %}{% set messages = messages[1:] %}{% else %}{% set system_message = '' %}{% endif %}{{　bos_token }}{{　system_message　}}{% for message in messages %}{% if message['role'] == 'user' %}{{ '### Input:\n' + message['content'] }}{% elif message['role'] == 'assistant' %}{{ '\n\n### Output:\n' + message['content'] + eos_token }}{% endif %}{% if loop.last and add_generation_prompt %}{{ '\n\n### Output:\n' }}{% endif %}{% endfor %}",
    use_tqdm=False)
print(output[0].outputs[0].text)

# COMMAND ----------

# MAGIC %md
# MAGIC ## Define pyfunc

# COMMAND ----------

import mlflow
import torch
from mlflow.types.llm import ChatCompletionResponse, ChatMessage, ChatParams, ChatChoice
from vllm import LLM, SamplingParams

class StockmarkModel(mlflow.pyfunc.PythonModel):
  
  def load_context(self, context): 

    # For generative models (task=generate) only
    self.llm = LLM(model=context.artifacts["model-path"], task="generate", tensor_parallel_size=torch.cuda.device_count())  # Name or path of your model

    self.CHAT_TEMPLATE = "{% if messages[0]['role'] == 'system' %}{% set system_message = messages[0]['content'] %}{% set messages = messages[1:] %}{% else %}{% set system_message = '' %}{% endif %}{{　bos_token }}{{　system_message　}}{% for message in messages %}{% if message['role'] == 'user' %}{{ '### Input:\n' + message['content'] }}{% elif message['role'] == 'assistant' %}{{ '\n\n### Output:\n' + message['content'] + eos_token }}{% endif %}{% if loop.last and add_generation_prompt %}{{ '\n\n### Output:\n' }}{% endif %}{% endfor %}"
    
  def predict(self, context, model_input, params=None):
    print(model_input, flush=True)
    print(params, flush=True)
    if isinstance(model_input, pd.DataFrame):
      # model_input = model_input.to_dict(orient="records")[0]
      model_input = model_input.to_dict()['messages'] # 入力の整形方法を変更
      model_input_list = list(model_input.values())

    # answer = self.generator(messages, **params)
    # answer = self.generator(model_input_list, **params) # 入力をmessagesからmodel_input_listに変更
    sampling_params = SamplingParams(
      temperature=params.get("temperature", 0.5), 
      top_p=params.get("top_p", 0.8), 
      top_k=params.get("top_k", 5), 
      max_tokens=params.get("max_new_tokens", 200), 
      presence_penalty=params.get("presence_penalty", 1.1))
    answer = self.llm.chat(
      model_input_list, 
      sampling_params=sampling_params, 
      chat_template=self.CHAT_TEMPLATE,
      use_tqdm=False)
    # assistant_message = answer[0]["generated_text"][-1]
    # assistant_message = answer[0][0]["generated_text"][-1] # assistant_messageの抽出をanwserの構造に合わせて変更
    assistant_message = answer[0].outputs[0].text

    response = ChatCompletionResponse(
        choices=[ChatChoice(index=0, message=ChatMessage(role="assistant", content=assistant_message))],
        usage={},
        model="stockmark-13b-instruct",
    )

    return response.to_dict()

# COMMAND ----------

# MAGIC %md
# MAGIC ## Log and register the pyfunc as a model

# COMMAND ----------

import numpy as np
import pandas as pd

import mlflow
from mlflow.models import infer_signature

with mlflow.start_run(run_name="stockmark-13b-instruct-vllm"):
  # 入出力スキーマの定義
  input_example = {
    "messages": [
        {
            "role": "user",
            "content": "日本の首都はどこ？",
        }
    ],
  }

  output_response = {
    'id': 'chatcmpl_e048d1af-4b9c-4cc9-941f-0311ac5aa7ab',
    'choices': [
      {
        'finish_reason': 'stop', 
        'index': 0,
        'logprobs': "",
        'message': {
          'content': '首都は東京です。',
          'role': 'assistant'
          }
        }
      ],
    'created': 1719722525,
    'model': 'dbrx-instruct-032724',
    'object': 'chat.completion',
    'usage': {'completion_tokens': 74, 'prompt_tokens': 803, 'total_tokens': 877}
  }

  params={
    "max_new_tokens": 512,
    "temperature": 0.1,
    "do_sample": True,
  }

  signature = infer_signature(
    model_input=input_example, 
    model_output=output_response, 
    params=params)
  
  logged_chain_info = mlflow.pyfunc.log_model(
    python_model=StockmarkModel(),
    artifact_path="stockmark-13b-instruct-vllm",
    signature=signature, 
    input_example=input_example,
    example_no_conversion=True,
    pip_requirements=[
      "mlflow==2.19.0", 
      "torch==2.5.1", 
      "transformers==4.45.2", 
      "accelerate==0.31.0", 
      "vllm==0.6.5", 
      "hf_transfer"],
    artifacts={'model-path': MODEL_PATH},
    registered_model_name=registered_model_name,
  )

# COMMAND ----------

# MAGIC %md
# MAGIC ## Set alias

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

# MAGIC %md
# MAGIC ## Option: Load the registered model adn run inference using pyfunc

# COMMAND ----------

import mlflow
model_champion_uri = "models:/{model_name}@Champion".format(model_name=registered_model_name)
 
print("Loading registered model version from URI: '{model_uri}'".format(model_uri=model_champion_uri))
champion_model = mlflow.pyfunc.load_model(model_champion_uri)

input_example = {
    "messages": [{"role": "user", "content": "日本の首都はどこ？"}]
  }
champion_model.predict(input_example, params=None)

# COMMAND ----------

# MAGIC %md
# MAGIC ## Deploy the model as an endpoint

# COMMAND ----------

# サービングエンドポイントの作成または更新
from datetime import timedelta
from databricks.sdk import WorkspaceClient
from databricks.sdk.service.serving import EndpointCoreConfigInput, ServedModelInput, ServedModelInputWorkloadSize, ServedModelInputWorkloadType

# サービングエンドポイントの名前を指定
serving_endpoint_name = 'hiroshi-stockmark-13b-instruct-gpu-serving-vllm'
latest_model_version = latest_version
model_name = registered_model_name

databricks_url = dbutils.notebook.entry_point.getDbutils().notebook().getContext().apiUrl().getOrElse(None)
token = dbutils.notebook.entry_point.getDbutils().notebook().getContext().apiToken().getOrElse(None)

w = WorkspaceClient()
endpoint_config = EndpointCoreConfigInput(
    name=serving_endpoint_name,
    served_models=[
        ServedModelInput(
            model_name=model_name,
            model_version=latest_model_version,
            workload_type=ServedModelInputWorkloadType.GPU_LARGE,
            workload_size=ServedModelInputWorkloadSize.SMALL,
            scale_to_zero_enabled=True
        )
    ]
)

existing_endpoint = next(
    (e for e in w.serving_endpoints.list() if e.name == serving_endpoint_name), None
)
serving_endpoint_url = f"{databricks_url}/ml/endpoints/{serving_endpoint_name}"
if existing_endpoint == None:
    print(f"Creating the endpoint {serving_endpoint_url}, this will take a few minutes to package and deploy the endpoint...")
    w.serving_endpoints.create_and_wait(name=serving_endpoint_name, config=endpoint_config, timeout=timedelta(minutes=60))
else:
    print(f"Updating the endpoint {serving_endpoint_url} to version {latest_model_version}, this will take a few minutes to package and deploy the endpoint...")
    w.serving_endpoints.update_config_and_wait(served_models=endpoint_config.served_models, name=serving_endpoint_name, timeout=timedelta(minutes=60))
    
displayHTML(f'Your Model Endpoint Serving is now available. Open the <a href="/ml/endpoints/{serving_endpoint_name}">Model Serving Endpoint page</a> for more details.')

# COMMAND ----------

# MAGIC %md
# MAGIC ## Test the endpoint

# COMMAND ----------

import requests
import json

data = {
  "messages": [{"role": "user", "content": "日本の消費税に関して小学生でもわかるように教えて。"}],
  "temperature": 0.3,
  "max_new_tokens": 500,
}

endpoint_name = "hiroshi-stockmark-13b-instruct-gpu-serving-vllm"

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
# MAGIC ---

# COMMAND ----------

# MAGIC %md
# MAGIC おまけ：OpenAIのクライアントから呼び出す

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

response = client.chat.completions.create(
    model=endpoint_name,
    messages=[{"role": "user", "content": "日本の首都はどこ？"}],
    max_tokens=100,
    temperature=0.1
)

print("【回答】")
print(response.choices[0].message.content)

# COMMAND ----------


