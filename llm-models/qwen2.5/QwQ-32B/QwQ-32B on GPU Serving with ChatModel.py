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
# MAGIC %pip install -U mlflow==2.19.0 threadpoolctl==3.5.0 transformers==4.49.0
# MAGIC dbutils.library.restartPython()

# COMMAND ----------

# MAGIC %md
# MAGIC ## モデルのダウンロードを高速化するオプション

# COMMAND ----------

import os
os.environ["HF_HUB_ENABLE_HF_TRANSFER"] = "1"

# COMMAND ----------

HF_REPO_ID = "Qwen/QwQ-32B"
MODEL_PATH = "/Volumes/hiroshi/models/models_from_hf/QwQ-32B"
registered_model_name = "hiroshi.models.qwq-32b"
endpoint_name = 'hiroshi-qwq-32b-instruct-gpu-serving'

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

model = AutoModelForCausalLM.from_pretrained(MODEL_PATH, device_map="auto", torch_dtype="auto")

# COMMAND ----------

# MAGIC %md
# MAGIC ## Load a tokenizer with custom "chat_template" and run inference with a ChatCompletion-style prompt

# COMMAND ----------


tokenizer = AutoTokenizer.from_pretrained(
  MODEL_PATH,
)

instruction = "日本の首都はどこ？"

messages = [
    {"role": "user", "content": instruction},
]

input_ids = tokenizer.apply_chat_template(messages, add_generation_prompt=True, return_tensors="pt").to(model.device)
# inputs = tokenizer(messages, return_tensors="pt").to(model.device)
output_ids = model.generate(input_ids,
                            max_new_tokens=512,
                            temperature=0.1,
                            do_sample=True)
output = tokenizer.decode(output_ids[0], skip_special_tokens=True)
print(output)


# COMMAND ----------

# MAGIC %md
# MAGIC ## Build a pipeline and run inference again to check ChatCompletion prompt works correctly

# COMMAND ----------

from transformers import pipeline
from mlflow.types.llm import ChatCompletionResponse, ChatMessage, ChatParams, ChatChoice

generator = pipeline(
    "text-generation",
    tokenizer=tokenizer,
    model=model,
)
answer = generator(messages, max_new_tokens=1000)
assistant_message = answer[0]["generated_text"][-1]

output = ChatCompletionResponse(
    choices=[ChatChoice(index=0, message=ChatMessage(**assistant_message))],
    usage={},
    model="Qwen/QwQ-32B",
)
output.to_dict()

# COMMAND ----------

# MAGIC %md
# MAGIC ## パイプラインを削除し、メモリーを解放します

# COMMAND ----------

import torch
import gc

del generator
gc.collect()
torch.cuda.empty_cache()

# COMMAND ----------

# MAGIC %md
# MAGIC ## モデルをMLFlowに登録する

# COMMAND ----------

import os
from dataclasses import dataclass
from typing import Optional, Dict, List, Generator
from mlflow.pyfunc import ChatModel
from mlflow.types.llm import (
    # Non-streaming helper classes
    ChatCompletionRequest,
    ChatCompletionResponse,
    ChatCompletionChunk,
    ChatMessage,
    ChatChoice,
    ChatParams,    
    # Helper classes for streaming agent output
    ChatChoiceDelta,
    ChatChunkChoice,
)
import pandas as pd
import mlflow
from openai import OpenAI
from transformers import AutoTokenizer

class QwQ32BChatCompletionModel(ChatModel):
    """
    Defines a custom agent that processes ChatCompletionRequests
    and returns ChatCompletionResponses.
    """
    def __init__(self):
        """
        コンストラクタ
        """

    
    def load_context(self, context):
        """
        コンテキストの設定
        """

        model = AutoModelForCausalLM.from_pretrained(
          context.artifacts["model-path"], 
          device_map="auto", 
          torch_dtype=torch.bfloat16
        )

        tokenizer = AutoTokenizer.from_pretrained(
          context.artifacts["model-path"],
        )
        
        self.generator = pipeline(
          "text-generation",
          tokenizer=tokenizer,
          model=model,
        )

    def _build_prompt(self, message):
        """
        プロンプトフォーマットの変換
        """

        converted_prompt = self.tokenizer.apply_chat_template(
            message,
            add_generation_prompt=True,
            add_special_tokens=False,
            tokenize=False)

        return converted_prompt
            
    @mlflow.trace(name="predict")
    def predict(self, context, messages: list[ChatMessage], params: ChatParams) -> ChatCompletionResponse:
        """
        推論メイン
        """

        # LLMに回答を生成させる
        with mlflow.start_span(name="generate_answer", span_type="LLM") as span:
            messages_list = []
            for message in messages:
                messages_list.append(message.to_dict())

            param_dict = params.to_dict()
            for param_name in ['max_tokens', 'stop', 'n', 'stream']:
                value = param_dict.pop(param_name, None)
                if param_name == 'max_tokens':
                    param_dict['max_new_tokens'] = value

            answer = self.generator(messages_list, **param_dict) # 入力をmessagesからmodel_input_listに変更
            assistant_message = answer[0]["generated_text"][-1] # assistant_messageの抽出をanwserの構造に合わせて変更
            span.set_inputs({"messages": messages, "params": params})
            span.set_outputs({"answer": answer})
        
        
        # 回答データを整形して返す.
        # ChatCompletionResponseの形式で返さないと後々エラーとなる。
        with mlflow.start_span(name="create_response") as span:
            response = ChatCompletionResponse(
                choices=[ChatChoice(index=0, message=ChatMessage(**assistant_message))],
                usage={},
                model=HF_REPO_ID,
            )
            span.set_inputs({"original_answer": assistant_message})
            span.set_outputs({"response": response})

        return response

# COMMAND ----------

import os
import mlflow

mlflow.set_registry_uri("databricks-uc")

with mlflow.start_run():
    logged_agent_info = mlflow.pyfunc.log_model(
        python_model=QwQ32BChatCompletionModel(),
        artifact_path="model",
        pip_requirements=[
          "mlflow==2.19.0", 
          "torch==2.3.1", 
          "transformers==4.49.0", 
          "accelerate==0.31.0", 
          "threadpoolctl==3.5.0",
          "hf_transfer"],
        artifacts={'model-path': MODEL_PATH},
        registered_model_name=registered_model_name
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

input_example = {
    "messages": [{"role": "user", "content": "日本の首都はどこ？"}]
  }
champion_model.predict(input_example, params=None)

# COMMAND ----------

# サービングエンドポイントの作成または更新
from datetime import timedelta
from databricks.sdk import WorkspaceClient
from databricks.sdk.service.serving import EndpointCoreConfigInput, ServedModelInput, ServedModelInputWorkloadSize, ServedModelInputWorkloadType

# サービングエンドポイントの名前を指定
serving_endpoint_name = endpoint_name
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
    w.serving_endpoints.create_and_wait(name=serving_endpoint_name, config=endpoint_config, timeout=timedelta(minutes=180))
else:
    print(f"Updating the endpoint {serving_endpoint_url} to version {latest_model_version}, this will take a few minutes to package and deploy the endpoint...")
    w.serving_endpoints.update_config_and_wait(served_models=endpoint_config.served_models, name=serving_endpoint_name, timeout=timedelta(minutes=180))
    
displayHTML(f'Your Model Endpoint Serving is now available. Open the <a href="/ml/endpoints/{serving_endpoint_name}">Model Serving Endpoint page</a> for more details.')

# COMMAND ----------

import requests
import json

data = {
  "messages": [{"role": "user", "content": "日本の首都はどこ？"}],
  "temperature": 0.1,
  "max_new_tokens": 10,
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

response = client.chat.completions.create(
    model=endpoint_name,
    messages=[{"role": "user", "content": "日本の首都はどこ？"}],
    max_tokens=10,
    temperature=0.1
)

print("【回答】")
print(response.choices[0].message.content)

# COMMAND ----------


