# Databricks notebook source
# MAGIC %md
# MAGIC # Databricksで`multilingual-e5-large`モデルをサービングする
# MAGIC
# MAGIC このノートブックでは、[multilingual-e5-large model](https://huggingface.co/intfloat/multilingual-e5-large)を `sentence_transformers` フレーバーで MLFLow にロギングし、Unity Catalog でモデルを管理し、サービングエンドポイントを作成する方法を示します。
# MAGIC
# MAGIC このノートブックの環境
# MAGIC - ランタイム: サーバーレスノートブック
# MAGIC

# COMMAND ----------

# MAGIC %md
# MAGIC ## ライブラリのインストール

# COMMAND ----------

# MAGIC %pip install mlflow==2.21.3 torch==2.5.1 sentence_transformers==4.1.0
# MAGIC dbutils.library.restartPython()

# COMMAND ----------

# MAGIC %md
# MAGIC ## 変数をセット

# COMMAND ----------

CATALOG = "hiroshi"
SCHEMA = "models"
MODEL_NAME = "multilingual-e5-large"
ENDPOINT_NAME = MODEL_NAME + "-embedding"

# COMMAND ----------

# MAGIC %md
# MAGIC ## モデルをMLFlowに記録する

# COMMAND ----------

# MAGIC %md
# MAGIC まずはモデルをダウンロード

# COMMAND ----------

from sentence_transformers import SentenceTransformer
model_name = "intfloat/multilingual-e5-large"

model = SentenceTransformer(model_name)

# COMMAND ----------

# MAGIC %md
# MAGIC 続いて、モデルのシグネイチャー（入力と出力の形式）を定義

# COMMAND ----------

import mlflow
import pandas as pd

# 入出力スキーマの定義
sentences = ["これは例文です", "各文章は変換されます"]
signature = mlflow.models.infer_signature(
    sentences,
    model.encode(sentences),
)

signature

# COMMAND ----------

# MAGIC %md
# MAGIC MLFlowの`sentence_transformers` フレーバーで MLFlow にロギング

# COMMAND ----------

# MLFlowのSentence Transformerフレーバーを使って登録
with mlflow.start_run() as run:  
    mlflow.sentence_transformers.log_model(
      model, 
      artifact_path=MODEL_NAME, 
      signature=signature,
      input_example=sentences,
      extra_pip_requirements=["torchvision==0.20.1"]) # Obviously install Torchvision with specific version corresponding to Torch's version to avoid a deployment error

# COMMAND ----------

# MAGIC %md
# MAGIC ## モデルを Unity Catalog に登録する
# MAGIC  デフォルトでは、MLflowはDatabricksワークスペースのモデルレジストリにモデルを登録します。代わりにUnity Catalogにモデルを登録するには、[ドキュメント](https://docs.databricks.com/machine-learning/manage-model-lifecycle/index.html)に従い、レジストリサーバーをDatabricks Unity Catalogに設定します。
# MAGIC
# MAGIC  Unity Catalogにモデルを登録するには、ワークスペースでUnity Catalogが有効になっている必要があるなど、[いくつかの要件](https://docs.databricks.com/machine-learning/manage-model-lifecycle/index.html#requirements)があります。
# MAGIC

# COMMAND ----------

# Unityカタログにモデルを登録するためにMLflow Pythonクライアントを設定する
import mlflow
mlflow.set_registry_uri("databricks-uc")

# COMMAND ----------

# SQLクエリをPython文字列として構築
use_catalog_query = f"USE CATALOG {CATALOG};"
create_schema_query = f"CREATE SCHEMA IF NOT EXISTS {SCHEMA};"

# SQLクエリを実行
spark.sql(use_catalog_query)
spark.sql(create_schema_query)

# COMMAND ----------

# Unityカタログへのモデル登録

registered_name = f"{CATALOG}.{SCHEMA}.{MODEL_NAME}" # UCモデル名は<カタログ名>.<スキーマ名>.<モデル名>のパターンに従っており、カタログ名、スキーマ名、登録モデル名に対応していることに注意してください。
result = mlflow.register_model(
    "runs:/"+run.info.run_id+"/"+MODEL_NAME,
    registered_name,
)

# COMMAND ----------

from mlflow import MlflowClient
client = MlflowClient()

# 上記のセルに登録されている正しいモデルバージョンを選択
client.set_registered_model_alias(name=registered_name, alias="Champion", version=result.version)

# COMMAND ----------

# MAGIC %md
# MAGIC ## Unityカタログからモデルを読み込む

# COMMAND ----------

import mlflow
import pandas as pd

loaded_model = mlflow.pyfunc.load_model(f"models:/{registered_name}@Champion")

# ロードされたモデルを使って予測を立てる
loaded_model.predict(
  ["MLとは何か？", "大規模言語モデルとは何か？"],
)

# COMMAND ----------

# MAGIC %md
# MAGIC ## モデル提供エンドポイントの作成
# MAGIC モデルが登録されたら、APIを使用してDatabricks GPU Model Serving Endpointを作成し、`multilingual-e5-large`モデルをサービングしていきます。
# MAGIC
# MAGIC 以下のデプロイにはGPUモデルサービングが必要です。GPU モデルサービングの詳細については、Databricks チームにお問い合わせいただくか、サインアップ [こちら](https://docs.google.com/forms/d/1-GWIlfjlIaclqDz6BPODI2j1Xg4f4WbFvBXyebBpN-Y/edit) してください。

# COMMAND ----------

databricks_url = dbutils.notebook.entry_point.getDbutils().notebook().getContext().apiUrl().getOrElse(None)
token = dbutils.notebook.entry_point.getDbutils().notebook().getContext().apiToken().getOrElse(None)

# COMMAND ----------

# MAGIC %md
# MAGIC デプロイ開始

# COMMAND ----------

from mlflow.deployments import get_deploy_client

model_version = result  # mlflow.register_modelの返された結果

serving_endpoint_name = ENDPOINT_NAME
latest_model_version = model_version.version
model_name = model_version.name

client = get_deploy_client("databricks")
endpoint = client.create_endpoint(
    name=serving_endpoint_name,
    config={
        "served_entities": [
            {
                "entity_name": model_name,
                "entity_version": latest_model_version,
                "workload_type": "GPU_SMALL",
                "workload_size": "Small",
                "scale_to_zero_enabled": True
            }
        ]
    }
)

# COMMAND ----------

# MAGIC %md
# MAGIC デプロイが完了するのをMax 120分待機。多くの場合、20-30分ほどで完了します。

# COMMAND ----------

from datetime import timedelta
from databricks.sdk import WorkspaceClient
from databricks.sdk.service.serving import EndpointCoreConfigInput, ServedModelInput, ServedModelInputWorkloadSize, ServedModelInputWorkloadType

w = WorkspaceClient()
w.serving_endpoints.wait_get_serving_endpoint_not_updating(name=serving_endpoint_name, timeout=timedelta(minutes=120))

# COMMAND ----------

# MAGIC %md
# MAGIC モデルサービングエンドポイントの準備ができたら、同じワークスペースで実行されているMLflow Deployments SDKで簡単にクエリできます。

# COMMAND ----------

import mlflow.deployments

client = mlflow.deployments.get_deploy_client("databricks")

embeddings_response = client.predict(
    endpoint=ENDPOINT_NAME,
    inputs={
        "inputs": ["おはようございます"]
    }
)
print(i)
embeddings_response['predictions']

# COMMAND ----------

import time

start = time.time()

# If you get timeout error (from the endpoint not yet being ready), then rerun this.
endpoint_response = w.serving_endpoints.query(name=ENDPOINT_NAME, dataframe_records=['こんにちは', 'おはようございます'])

end = time.time()

print(endpoint_response)
print(f'Time taken for querying endpoint in seconds: {end-start}')
