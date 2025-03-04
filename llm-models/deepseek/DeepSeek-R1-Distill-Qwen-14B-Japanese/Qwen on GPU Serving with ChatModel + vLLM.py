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
# MAGIC %pip install -U mlflow==2.19.0 threadpoolctl==3.5.0 transformers==4.49.0 vllm==0.7.3
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
  repo_id="cyberagent/DeepSeek-R1-Distill-Qwen-14B-Japanese",
  local_dir="/local_disk0/hf")

!mkdir -p /Volumes/hiroshi/models/models_from_hf/DeepSeek-R1-Distill-Qwen-14B-Japanese
!cp -L -R /local_disk0/hf/* /Volumes/hiroshi/models/models_from_hf/DeepSeek-R1-Distill-Qwen-14B-Japanese

# COMMAND ----------

# MAGIC %md
# MAGIC ## Load a LLM model

# COMMAND ----------

MODEL_PATH = "/Volumes/hiroshi/models/models_from_hf/DeepSeek-R1-Distill-Qwen-14B-Japanese"
registered_model_name = "hiroshi.models.deepseek-r1-distill-qwen-14b-japanese"

# COMMAND ----------

import torch
from vllm import LLM, SamplingParams

# For generative models (task=generate) only
llm = LLM(model=MODEL_PATH, task="generate", tensor_parallel_size=torch.cuda.device_count())

sampling_params = SamplingParams(
  temperature=0.1, 
  top_p=0.8, 
  top_k=5, 
  max_tokens=512, 
  presence_penalty=1.1)

# COMMAND ----------

messages = [
    {"role": "user", "content": "日本の首都はどこ？"},
]
output = llm.chat(
    messages, 
    sampling_params=sampling_params, 
    use_tqdm=False)
print(output[0].outputs[0].text)

# COMMAND ----------

from mlflow.types.llm import ChatCompletionResponse, ChatMessage, ChatParams, ChatChoice
output = ChatCompletionResponse(
    choices=[ChatChoice(index=0, message=ChatMessage(role="assistant", content=output[0].outputs[0].text))],
    usage={},
    model="cyberagent/DeepSeek-R1-Distill-Qwen-14B-Japanese",
)
output.to_dict()

# COMMAND ----------

import torch
import gc

# パイプラインを削除し、メモリーを解放します
del llm
gc.collect()
torch.cuda.empty_cache()

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
from vllm import LLM, SamplingParams

class DeepSeekQwenChatCompletionModel(ChatModel):
    """
    Defines a custom agent that processes ChatCompletionRequests
    and returns ChatCompletionResponses.
    """
    def __init__(self):
        """
        コンストラクタ
        """
        import os
        os.environ["HF_HUB_ENABLE_HF_TRANSFER"] = "1"
        self.llm = LLM(model="cyberagent/DeepSeek-R1-Distill-Qwen-14B-Japanese", task="generate", tensor_parallel_size=torch.cuda.device_count())

    
    # def load_context(self, context):
    #     """
    #     コンテキストの設定
    #     """

    #     self.llm = LLM(model=context.artifacts["model-path"], task="generate", tensor_parallel_size=torch.cuda.device_count())

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
            for param_name in ['stop', 'n', 'stream']:
                value = param_dict.pop(param_name, None)

            sampling_params = SamplingParams(
                temperature=param_dict.get("temperature", 0.5), 
                top_p=param_dict.get("top_p", 0.8), 
                top_k=param_dict.get("top_k", 5), 
                max_tokens=param_dict.get("max_tokens", 200), 
                presence_penalty=param_dict.get("presence_penalty", 1.1))
            answer = self.llm.chat(
                messages_list, 
                sampling_params=sampling_params, 
                use_tqdm=False)
            assistant_message = answer[0].outputs[0].text

            span.set_inputs({"messages": messages, "params": params})
            span.set_outputs({"answer": answer})
        
        
        # 回答データを整形して返す.
        # ChatCompletionResponseの形式で返さないと後々エラーとなる。
        with mlflow.start_span(name="create_response") as span:
            response = ChatCompletionResponse(
                choices=[ChatChoice(index=0, message=ChatMessage(role="assistant", content=assistant_message))],
                usage={},
                model="cyberagent/DeepSeek-R1-Distill-Qwen-14B-Japanese",
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
        python_model=DeepSeekQwenChatCompletionModel(),
        artifact_path="model",
        pip_requirements=[
          "mlflow==2.19.0", 
          "torch==2.5.1", 
          "transformers==4.45.2", 
          "accelerate==0.31.0", 
          "threadpoolctl==3.5.0",
          "vllm==0.7.3",
          "hf_transfer"],
        artifacts={'model-path': MODEL_PATH},
        registered_model_name=registered_model_name
    )

# COMMAND ----------

import mlflow
import torch
from mlflow.types.llm import ChatCompletionResponse, ChatMessage, ChatParams, ChatChoice
from vllm import LLM, SamplingParams

class DeepSeekQwenModel(mlflow.pyfunc.PythonModel):
  
  def load_context(self, context): 

    # For generative models (task=generate) only
    self.llm = LLM(model=context.artifacts["model-path"], task="generate", tensor_parallel_size=torch.cuda.device_count())
    
  def predict(self, context, model_input, params=None):
    print(model_input, flush=True)
    print(params, flush=True)
    if isinstance(model_input, pd.DataFrame):
      # model_input = model_input.to_dict(orient="records")[0]
      model_input = model_input.to_dict()['messages'] # 入力の整形方法を変更
      model_input_list = list(model_input.values())

    sampling_params = SamplingParams(
      temperature=params.get("temperature", 0.5), 
      top_p=params.get("top_p", 0.8), 
      top_k=params.get("top_k", 5), 
      max_tokens=params.get("max_new_tokens", 200), 
      presence_penalty=params.get("presence_penalty", 1.1))
    
    answer = self.llm.chat(
      model_input_list, 
      sampling_params=sampling_params, 
      use_tqdm=False)

    assistant_message = answer[0].outputs[0].text

    response = ChatCompletionResponse(
        choices=[ChatChoice(index=0, message=ChatMessage(role="assistant", content=assistant_message))],
        usage={},
        model="cyberagent/DeepSeek-R1-Distill-Qwen-14B-Japanese",
    )

    return response.to_dict()

# COMMAND ----------

import numpy as np
import pandas as pd

import mlflow
from mlflow.models import infer_signature

with mlflow.start_run(run_name="deepseek-qwen-instruct-vllm"):
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
    python_model=DeepSeekQwenModel(),
    artifact_path="model",
    signature=signature, 
    input_example=input_example,
    example_no_conversion=True,
    pip_requirements=[
      "mlflow==2.19.0", 
      "torch==2.5.1", 
      "transformers==4.49.0", 
      "accelerate==0.31.0", 
      "vllm==0.7.3", 
      "threadpoolctl==3.5.0",
      "hf_transfer"],
    artifacts={'model-path': MODEL_PATH},
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
endpoint_name = 'hiroshi-deepseek-qwen25-14b-instruct-gpu-vllm'
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
    w.serving_endpoints.create_and_wait(name=serving_endpoint_name, config=endpoint_config, timeout=timedelta(minutes=60))
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

endpoint_name = "hiroshi-deepseek-qwen25-14b-instruct-gpu-vllm"
response = client.chat.completions.create(
    model=endpoint_name,
    messages=[{"role": "user", "content": "日本の首都はどこ？"}],
    max_tokens=10,
    temperature=0.1
)

print("【回答】")
print(response.choices[0].message.content)

# COMMAND ----------


