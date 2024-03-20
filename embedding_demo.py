"""
参考
https://cloud.google.com/bigquery/docs/vector-search?hl=ja#create_a_dataset
"""
import os
from typing import Optional, Union, List
from io import BytesIO

import db_dtypes
from google.cloud import bigquery
import pandas as pd
import vertexai
from vertexai.vision_models import (
    Image,
    MultiModalEmbeddingModel,
    MultiModalEmbeddingResponse,
)
from google.cloud import storage


PROJECT_ID = 'gifted-mountain-415005'
LOCATION = 'us-central1'
DATASET_ID = f'{PROJECT_ID}.vector_search'
CORRECT_TABLE_ID = f"{DATASET_ID}.correct_data"
INCORRECT_TABLE_ID = f"{DATASET_ID}.incorrect_data"


# ライブラリの初期化
vertexai.init(project=PROJECT_ID, location=LOCATION)
multimodalembedding_model = MultiModalEmbeddingModel.from_pretrained("multimodalembedding")
bigquery_client = bigquery.Client(project=PROJECT_ID)
storage_client = storage.Client(project=PROJECT_ID)


def get_image_embeddings(
    image_path: str,
    dimension: Optional[int] = 1408,
) -> MultiModalEmbeddingResponse:
    """
    画像から埋め込みを取得する関数

    :param image_path: 画像のパス
    :param dimension: 埋め込みの次元数
    :return: 埋め込み
    """

    image = Image.load_from_file(location=image_path)
    embeddings = multimodalembedding_model.get_embeddings(
        image=image,
        dimension=dimension,
    )

    return embeddings


def run_query(query:str) -> Union[pd.DataFrame, List]:
    """
    BigQuery に対しクエリを実行する関数

    :param query: クエリ
    :return: エラーがなければ"ok"、エラーがあればエラーリストを返す
    """
    query_job = bigquery_client.query(query)
    df = query_job.to_dataframe()

    return df if query_job.errors is None else query_job.errors


# シェフの作った画像（不正解画像） [TODO:ユーザが選択した画像を指定]
# incorrect_image_path = "gs://gen-ai-hackson-by-jaguer-team-a-sampe-data/incorrect-data/IMG_6741.jpg"
# incorrect_image_path = 'gs://gen-ai-hackson-by-jaguer-team-a-sampe-data/incorrect-data/cat.png'
incorrect_image_path = "gs://gen-ai-hackson-by-jaguer-team-a-sampe-data/incorrect-data/フルーツサラダ.png"
print(f"シェフの作った画像（不正解画像）: {incorrect_image_path.split('/')[4]}")



# 不正解画像の埋め込みを取得
incorrect_embeddings = get_image_embeddings(image_path=incorrect_image_path)


# データを挿入または更新
# insert_query = f"""
# MERGE `{INCORRECT_TABLE_ID}` T
# USING (SELECT '{incorrect_image_path}' AS file_path, {incorrect_embeddings.image_embedding} AS embedding) S
# ON T.file_path = S.file_path
# WHEN MATCHED THEN
#   UPDATE SET T.embedding = S.embedding
# WHEN NOT MATCHED THEN
#   INSERT (file_path, embedding)
#   VALUES(S.file_path, S.embedding);
# """
# res_insert_query = run_query(query=insert_query)
# if type(res_insert_query) == list:
#     print(f"res_insert_query error : {res_insert_query}")


# ベクトル検索を実施
# path = incorrect_image_path.split("gs://", 1)[1] if "gs://" in incorrect_image_path else incorrect_image_path
# file_name = os.path.basename(path)
vector_search_options = '{"use_brute_force":true}'

execute_vector_search_query = f"""
SELECT
  query.file_path AS query_file_path,
  base.file_path AS base_file_path,
  distance
FROM
  VECTOR_SEARCH( TABLE `{CORRECT_TABLE_ID}`,
    'embedding',
    (SELECT '{incorrect_image_path}' AS file_path, {incorrect_embeddings.image_embedding} AS embedding),
    top_k => 3,
    distance_type => 'COSINE',
    OPTIONS => '{vector_search_options}');
"""
res_execute_vector_search_query = run_query(query=execute_vector_search_query)
if type(res_execute_vector_search_query) == list:
    print(f"execute_vector_search_query error : {res_execute_vector_search_query}")

for k, correct_image in enumerate(res_execute_vector_search_query['base_file_path'], start=1):
    # print(correct_image)
    print(f"正解画像 [{k}位]: {correct_image.split('/')[4]}")



# 正解の料理画像から使われている食材を取得
correct_image_url = res_execute_vector_search_query['base_file_path'][0].split('/')[4].split('.')[0]
correct_image_path = correct_image_url.split("gs://", 1)[1] if "gs://" in correct_image_url else correct_image_url
correct_image_file_name = int(os.path.basename(correct_image_path).split('.')[0])
print(correct_image_file_name)

# TODO: Cloud Run にのせたときに、food_shokuzai_list_table → food_shokuzai_list
get_shokuzai_query = f"""
SELECT
  food_menu,
  syokuzai
FROM
  `gifted-mountain-415005.master.food_shokuzai_list_table`
WHERE
  mcd = {correct_image_file_name}
"""
res_get_shokuzai_query = run_query(query=get_shokuzai_query)
if type(res_get_shokuzai_query) == list:
    print(f"res_get_shokuzai_query error : {res_get_shokuzai_query}")
print(res_get_shokuzai_query)


# 料理名
food_menu = res_get_shokuzai_query['food_menu'].unique().tolist()[0]

# 食材リスト
syokuzai_li = res_get_shokuzai_query['syokuzai'].tolist()
syokuzai_str = ""
for syokuzai in syokuzai_li:
    syokuzai_str += f"* {syokuzai} /n "

print(type(syokuzai_str))
print(syokuzai_str)