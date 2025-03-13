# Databricks notebook source
# MAGIC %md
# MAGIC REF: https://docs.databricks.com/en/generative-ai/agent-framework/agent-schema.html

# COMMAND ----------

# MAGIC %pip install -U -qqqq mlflow==2.19.0 openai==1.59.9 transformers==4.48.1
# MAGIC dbutils.library.restartPython()

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

class StockmarkChatCompletionWrapper(ChatModel):
    """
    Defines a custom agent that processes ChatCompletionRequests
    and returns ChatCompletionResponses.
    """
    def __init__(self):
        """
        コンストラクタ
        """

        self.model_config = mlflow.models.ModelConfig(development_config="cpu_endpoint_config.yaml")
        self.chat_model = None
        self.tokenizer = None

    
    def load_context(self, context):
        """
        コンテキストの設定
        """

        try:
            # サービングエンドポイントのホストに"DB_MODEL_SERVING_HOST_URL"が自動設定されるので、その内容をDATABRICKS_HOSTにも設定
            os.environ["DATABRICKS_HOST"] = os.environ["DB_MODEL_SERVING_HOST_URL"]
        except:
            pass

        # LLM基盤モデルのエンドポイントのクライアントを取得
        self.chat_model = OpenAI(
            api_key=os.environ.get("DATABRICKS_TOKEN"),
            base_url=os.environ.get("DATABRICKS_HOST") + "/serving-endpoints",
        )

        self.tokenizer = AutoTokenizer.from_pretrained(
            self.model_config.get("hf_model_name"),
            chat_template="{% if messages[0]['role'] == 'system' %}{% set system_message = messages[0]['content'] %}{% set messages = messages[1:] %}{% else %}{% set system_message = '' %}{% endif %}{{　bos_token }}{{　system_message　}}{% for message in messages %}{% if message['role'] == 'user' %}{{ '### Input:\n' + message['content'] }}{% elif message['role'] == 'assistant' %}{{ '\n\n### Output:\n' + message['content'] + eos_token }}{% endif %}{% if loop.last and add_generation_prompt %}{{ '\n\n### Output:\n' }}{% endif %}{% endfor %}")

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

        # プロンプトの構築
        with mlflow.start_span(name="_build_prompt") as span:
            prompt = self._build_prompt(messages)
            span.set_inputs({"original_prompt": messages})
            span.set_outputs({"prompt": prompt})

        # LLMに回答を生成させる
        with mlflow.start_span(name="generate_answer", span_type="LLM") as span:
            response = self.chat_model.completions.create(
                model=self.model_config.get("llm_endpoint_name"),
                prompt=prompt,
                **(params.to_dict() if params is not None else {'temperature': 0.1})
            )
            span.set_inputs({"prompt": prompt, "params": params})
            span.set_outputs({"answer": response})
        
        
        # 回答データを整形して返す.
        # ChatCompletionResponseの形式で返さないと後々エラーとなる。
        with mlflow.start_span(name="create_response") as span:
            response_dict = response.to_dict()
            response_dict['choices'][0]['message'] = {
                'content': response_dict['choices'][0]['text'],
                'role': 'assistant'
            }
            span.set_inputs({"original_answer": response})
            span.set_outputs({"response": response})

        return ChatCompletionResponse.from_dict(response_dict)

    def _create_chat_completion_chunk(self, content) -> ChatCompletionChunk:
        """Helper for constructing a ChatCompletionChunk instance for wrapping streaming agent output"""
        return ChatCompletionChunk(
                choices=[ChatChunkChoice(
                    delta=ChatChoiceDelta(
                        role="assistant",
                        content=content
                    )
                )]
            )

    def predict_stream(
        self, context, messages: List[ChatMessage], params: ChatParams
    ) -> Generator[ChatCompletionChunk, None, None]:
        last_user_question_text = messages[-1].content
        yield self._create_chat_completion_chunk(f"Echoing back your last question, word by word. You sent me this request from a client with type: ")
        for word in last_user_question_text.split(" "):
            yield self._create_chat_completion_chunk(word)
    


# COMMAND ----------

import mlflow
wrapper = StockmarkChatCompletionWrapper()
mlflow.models.set_model(wrapper)

# COMMAND ----------

# API_ROOT = dbutils.notebook.entry_point.getDbutils().notebook().getContext().apiUrl().get()
# os.environ["DATABRICKS_HOST"] = API_ROOT
# API_TOKEN = dbutils.notebook.entry_point.getDbutils().notebook().getContext().apiToken().get()
# os.environ["DATABRICKS_TOKEN"] = API_TOKEN

# wrapper.load_context(None)

# input_messages = [ChatMessage(role="user", content="東京の首都は京都だっけ？")]
# params_with_custom_inputs = ChatParams(temperature=0.1)
# response = wrapper.predict(context=None, messages=input_messages, params=params_with_custom_inputs)
# print(response.to_dict())
