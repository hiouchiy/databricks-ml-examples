# Databricks notebook source
# MAGIC %md
# MAGIC # `multilingual-e5-large`埋め込みモデルをDatabricks上で実行する
# MAGIC
# MAGIC [multilingual-e5-large (Embedding)モデル](https://huggingface.co/intfloat/multilingual-e5-large)は、任意のテキストを検索、分類、クラスタリング、意味検索などのタスクに使用できる低次元の密なベクトルにマッピングすることができます。また、LLMのベクトルデータベースにも利用できます。
# MAGIC
# MAGIC このノートブックの環境
# MAGIC - ランタイム: 14.1 GPU ML Runtime
# MAGIC - インスタンス: AWS の `g4dn.xlarge` または Azure の `Standard_NC4as_T4_v3`

# COMMAND ----------

# MAGIC %md
# MAGIC ### Sentence-Transformersを使う
# MAGIC
# MAGIC `multilingual-e5-large`モデルでsentence-transformersを使うと、文章を埋め込みとしてエンコードすることができます。

# COMMAND ----------

from sentence_transformers import SentenceTransformer, util

model_name = "intfloat/multilingual-e5-large"
model = SentenceTransformer(model_name)

sentences = ["男が食べ物を食べている。",
  "男がパンを食べている",
  "少女が赤ん坊を抱いている",
  "男が馬に乗っている",
  "女性がヴァイオリンを弾いている",
  "二人の男が森の中を荷車を押している",
  "男が白馬に乗って囲われた地面を走っている",
  "猿が太鼓を叩いている",
  "ゴリラの着ぐるみを着た人が太鼓を叩いている",
]
embeddings = model.encode(sentences, normalize_embeddings=True)

cos_sim = util.cos_sim(embeddings, embeddings)
print("Cosine-Similarity:", cos_sim)

# COMMAND ----------

# MAGIC %md
# MAGIC s2p (短いクエリから長文)検索タスクでは、`multilingual-e5-large`モデルに対して、各クエリは `query:`という命令で始まり、各Passageは`passage:`という命令で始まるとより精度が期待できる。

# COMMAND ----------

instruction_query = "query: "
instruction_passage = "passage: "

queries = ["チーズやヨーグルトなどの食品を作る際によく使われる有機体の種類は？"]
passages = [
  "好中性菌は中程度の温度、一般的には25℃～40℃の間で最もよく育つ。好中球菌はしばしばヒトや他の動物の体内や体上に生息している。多くの病原性好中球菌の至適生育温度は、ヒトの通常の体温である37℃である。チーズ、ヨーグルト、ビール、ワインなどの食品製造において、好中球菌は重要な役割を担っている。", 
  "コリオリ効果がなければ、地球の風は北から南へ、あるいは南から北へ吹く。しかし、コリオリ効果によって、北半球では北東から南西、あるいはその逆の風が吹く。南半球では北西から南東、またはその逆の風が吹く。",
  "要約 状態変化は相変化（相転移）の一例である。すべての相変化は、系のエネルギーの変化を伴う。秩序が高い状態から秩序が低い状態（液体から気体など）への変化は吸熱的である。秩序度の低い状態から秩序度の高い状態への変化（液体から固体への変化など）は、常に発熱を伴う。固体から液体への変換は、融合（または融解）と呼ばれる。1molの物質を溶かすのに必要なエネルギーは、その物質の融解エンタルピー(ΔHfus)である。1molの物質を気化させるのに必要なエネルギー変化は、気化エンタルピー(ΔHvap)である。固体の気体への直接変換は昇華である。物質1molを昇華させるのに必要なエネルギー量は昇華エンタルピー(ΔHsub)であり、融解エンタルピーと気化エンタルピーの和である。一定の加熱速度における、物質の温度対添加熱量、または加熱時間のプロットは、加熱曲線と呼ばれる。加熱曲線は、温度の変化を相転移に関連付ける。過熱された液体（気体になるべき温度と圧力の液体）は安定しない。冷却曲線は、多くの液体が予想される温度では凍結しないため、加熱曲線の正確な逆ではありません。その代わり、過冷却液体という、通常の融点より低いところに存在する準安定な液相を形成する。過冷却液体は通常、静置すると結晶化するか、同じ物質または別の物質の種結晶を加えると結晶化が誘発される。"
  ]
query_with_instruction = [instruction_query+q for q in queries]
passage_with_instruction = [instruction_passage+p for p in passages]

q_embeddings = model.encode(query_with_instruction, normalize_embeddings=True)
p_embeddings = model.encode(passage_with_instruction, normalize_embeddings=True)

scores = util.cos_sim(q_embeddings, p_embeddings)
print("Cosine-Similarity scores:", scores)

# COMMAND ----------

# MAGIC %md
# MAGIC ### Langchainを使う
# MAGIC

# COMMAND ----------

from langchain.embeddings import HuggingFaceBgeEmbeddings

model_kwargs = {'device': 'cuda'}
encode_kwargs = {'normalize_embeddings': True} # set True to compute cosine similarity
model_norm = HuggingFaceBgeEmbeddings(
    model_name=model_name,
    model_kwargs=model_kwargs,
    encode_kwargs=encode_kwargs
)

q_embeddings = model_norm.embed_documents(query_with_instruction)
p_embeddings = model_norm.embed_documents(passage_with_instruction)

scores = util.cos_sim(q_embeddings, p_embeddings)
print("Cosine-Similarity scores:", scores)

# COMMAND ----------


