import base64
import logging
import os
import random
import sys
import time
from io import BytesIO
from typing import List, Optional, Union

import gradio as gr
import pandas as pd
from google.cloud import bigquery
from google.cloud import storage
import google.cloud.logging
from PIL import Image as PIL_Image
import vertexai
from vertexai.preview.generative_models import (
    GenerativeModel,
    GenerationConfig,
    GenerationResponse,
    Part,
)
from vertexai.vision_models import (
    Image as VertexImage,
    MultiModalEmbeddingModel,
    MultiModalEmbeddingResponse,
)

import db_dtypes

PROJECT_ID = os.environ.get("PROJECT_ID", "gifted-mountain-415005")
LOCATION = os.environ.get("LOCATION", "asia-northeast1")
FILE_BUCKET_NAME = os.environ.get("FILE_BUCKET_NAME", "gen-ai-hackson-by-jaguer-team-a-sampe-data")
SUPPORTED_IMAGE_EXTENSIONS = [
    "png",
    "jpeg",
    "jpg",
]
MAX_PROMPT_SIZE_MB = float(os.environ.get("MAX_PROMPT_SIZE_MB", "100"))
DATASET_ID = f'{PROJECT_ID}.vector_search'
CORRECT_TABLE_ID = f"{DATASET_ID}.correct_data"




# 標準 Logger の設定
logging.basicConfig(
        format = "[%(asctime)s][%(levelname)s] %(message)s",
        level = logging.DEBUG # ログレベルをデバックから取得するように設定
    )
logger = logging.getLogger()


# 各サービスの初期化
try:
    logging_client = google.cloud.logging.Client()
    logging_client.setup_logging()
    vertexai.init(project=PROJECT_ID, location=LOCATION)
    storage_client = storage.Client()
    txt_model = GenerativeModel("gemini-pro")
    multimodal_model = GenerativeModel("gemini-pro-vision")
    multimodalembedding_model = MultiModalEmbeddingModel.from_pretrained("multimodalembedding")
    bigquery_client = bigquery.Client()
except Exception as e:
    print(f"An error occurred during the initialization of one or more services: {e}")


# CSS
def load_css():
    with open('style.css', 'r') as file:
        css_content = file.read()
    return css_content


# ファイルを Base64 にエンコード
def file_to_base64(file_path: str) -> str:
    try:
        with open(file_path, "rb") as file:
            return base64.b64encode(file.read()).decode("utf-8")
    except Exception as e:
        logger.error(f"Error during file_to_base64: {e}")

# GCS ファイルを Base64 にエンコード
def gcs_file_to_base64(gcs_url: str) -> str:
    # バケット名とファイルパスを取得
    if not gcs_url.startswith("gs://"):
        raise ValueError("URL must start with gs://")
    _, _, bucket_name, *file_path = gcs_url.split("/")
    file_path = "/".join(file_path)

    # ファイルを読み込む
    bucket = storage_client.bucket(bucket_name)
    blob = bucket.blob(file_path)
    try:
        file_data = blob.download_as_bytes()
        # Base64エンコードして返す
        return base64.b64encode(file_data).decode("utf-8")
    except Exception as e:
        logger.error(f"Error during gcs_file_to_base64: {e}")
        return None

# ファイルの拡張子を取得
def get_extension(file_path: str) -> str:
    if "." not in file_path:
        logger.error(f"Invalid file path. : {file_path}")

    extension = file_path.split(".")[-1].lower()
    if not extension:
        logger.error(f"File has no extension. : {file_path}")

    return extension


# 画像/動画ファイルを Cloud Storage にアップロード
def file_upload_gsc(file_bucket_name: str, source_file_path: str) -> str:
    try:
        bucket = storage_client.bucket(file_bucket_name)

        # ファイルの名前を取得
        destination_blob_name = os.path.basename(source_file_path)

        # ファイルをアップロード
        blob = bucket.blob(destination_blob_name)
        blob.upload_from_filename(source_file_path)
        return f"gs://{file_bucket_name}/{destination_blob_name}"

    except Exception as e:
        logger.error(f"Error uploading to Cloud Storage: {e}")


# mime_type を取得
def create_mime_type(extension: str) -> str:
    # サポートされた画像形式の場合
    if extension in SUPPORTED_IMAGE_EXTENSIONS:
        return "image/jpeg" if extension in ["jpg", "jpeg"] else f"image/{extension}"

    # サポートされていない拡張子の場合
    else:
        logger.error(f"Not supported mime_type for extension: {extension}")


# プロンプトサイズの計算
def calculate_prompt_size_mb(text: str, file_path: str) -> float:
    try:
        # テキストサイズをバイト単位で取得
        text_size_bytes = sys.getsizeof(text)

        # ファイルサイズをバイト単位で取得
        file_size_bytes = os.path.getsize(file_path)

        # バイトからメガバイトに単位変換
        prompt_size_mb = (text_size_bytes + file_size_bytes) / 1048576
    except Exception as e:
        logger.error(f"Error calculating prompt size: {e}")

    return prompt_size_mb


# 画像から埋め込みを取得する関数
def get_image_embeddings(
    image_path: str,
    dimension: Optional[int] = 1408,
) -> MultiModalEmbeddingResponse:

    try:
        image = VertexImage.load_from_file(location=image_path)
        embeddings = multimodalembedding_model.get_embeddings(
            image=image,
            dimension=dimension,
        )
    except Exception as e:
        logger.error(f"Error during get_image_embeddings: {e}")

    return embeddings


# クエリを実行する関数
def run_query(query:str) -> Union[pd.DataFrame, List]:
    try:
        query_job = bigquery_client.query(query)
        df = query_job.to_dataframe()
    except Exception as e:
        logger.error(f"Error during run_query: {e}")
    return df if query_job.errors is None else query_job.errors

# ファイルメタデータ取得
"""
ex-return)
file_data {
  mime_type: "image/jpeg"
  file_uri: "gs://food_menu_image/18850.jpeg"
}
"""
def get_file_data(gcs_uri: str):
    extension = get_extension(file_path=gcs_uri)
    mime_type = create_mime_type(extension)
    try:
        file_data = Part.from_uri(uri=gcs_uri, mime_type=mime_type)
    except Exception as e:
        logger.error(f"Error during get_file_data: {e}")
    return file_data


# history (会話履歴) を作成
def query_message(history: str, image: str) -> str:
    user_text = "料理チェックおねがします！"
    try:
        # ユーザーのクエリがテキストのみの場合
        if not (image):
            history.append(("[写真なし]",None))

        # ユーザーのクエリに画像が含まれる場合
        if image:
            prompt_size_mb = calculate_prompt_size_mb(text=None, file_path=image)

            # 画像サイズが上限を超えた場合
            if prompt_size_mb > MAX_PROMPT_SIZE_MB:
                history.append((f"[This Image is not display] Image size exceeds upper limit", None))
            else:
                image_extension = get_extension(image)
                base64 = file_to_base64(image)
                data_url = f"data:image/{image_extension};base64,{base64}"
                image_html = f'<img src="{data_url}" alt="Uploaded image">'
                history.append((f"{image_html}{user_text}", None))

    except Exception as e:
        logger.error(f"Error during query_message: {e}")
  
    return history



def gemini_response(history: str, image_file_path: str, text:str) -> str:

    try:
        # 画像がない時
        if not image_file_path:
            response = "写真を送るにゃん"
            history += [(None,response)]
            return history


        # 画像/動画ファイルを Cloud Storage にアップロード
        incorrect_image_gcs_uri = file_upload_gsc(file_bucket_name=FILE_BUCKET_NAME, source_file_path=image_file_path)

        # 不正解画像の埋め込みを取得
        incorrect_embeddings = get_image_embeddings(image_path=incorrect_image_gcs_uri)

        # ベクトル検索を実施
        vector_search_options = '{"use_brute_force":true}'

        execute_vector_search_query = f"""
        SELECT
        query.file_path AS query_file_path,
        base.file_path AS base_file_path,
        distance
        FROM
        VECTOR_SEARCH( TABLE `{CORRECT_TABLE_ID}`,
            'embedding',
            (SELECT '{incorrect_image_gcs_uri}' AS file_path, {incorrect_embeddings.image_embedding} AS embedding),
            top_k => 3,
            distance_type => 'COSINE',
            OPTIONS => '{vector_search_options}');
        """
        res_execute_vector_search_query = run_query(query=execute_vector_search_query)
        if type(res_execute_vector_search_query) == list:
            print(f"execute_vector_search_query error : {res_execute_vector_search_query}")

        for k, correct_image in enumerate(res_execute_vector_search_query['base_file_path'], start=1):

            print(f"正解画像 [{k}位]: {correct_image.split('/')[4]}()")
            print(res_execute_vector_search_query['distance'][k-1])

        # 距離が 0.45 以上離れている画像しかヒットしなかった場合
        if res_execute_vector_search_query['distance'][0] >= 0.45:
            response = "画像がヒットしないので、撮り直してくれるかにゃん！"
            history += [(None,response)]
            return history

        # 正解の画像 URL
        correct_image_gcs_uri = res_execute_vector_search_query['base_file_path'][0]

        # 正解の料理の食材リストを df で取得
        correct_image_file_name = correct_image_gcs_uri.split('/')[4] # 例）19295.jpg
        correct_image_number = int(correct_image_file_name.split('.')[0]) # 例）19295

        # TODO: Cloud Run にのせたときに、food_shokuzai_list_table → food_shokuzai_list
        get_shokuzai_query = f"""
        SELECT
        food_menu,
        syokuzai
        FROM
        `gifted-mountain-415005.master.food_shokuzai_list_table`
        WHERE
        mcd = {correct_image_number}
        """
        res_get_shokuzai_query = run_query(query=get_shokuzai_query)
        if type(res_get_shokuzai_query) == list:
            print(f"res_get_shokuzai_query error : {res_get_shokuzai_query}")

        #### プロンプト情報取得 ####
        # 不正解画像（シェフの作った画像）
        incorrect_file_data = get_file_data(gcs_uri=incorrect_image_gcs_uri)
        print(f'incorrect_file_data: {incorrect_file_data}')

        # 正解画像
        correct_file_data = get_file_data(gcs_uri=correct_image_gcs_uri)
        print(f'correct_file_data: {correct_file_data}')


        # 正解画像の料理名
        correct_food_menu = res_get_shokuzai_query['food_menu'].unique().tolist()[0]

        # 正解画像の食材リスト
        correct_syokuzai_str = ""
        correct_syokuzai_li = res_get_shokuzai_query['syokuzai'].tolist()
        for correct_syokuzai in correct_syokuzai_li:
            correct_syokuzai_str += f"* {correct_syokuzai} /n "

        # プロンプト作成
        prompt = ["""あなたは猫型配膳ロボットです。
自分がこれから運ぶ料理の盛り付けを評価します。
評価は「満点にゃ～」「もうちょっとだにゃ～」「もっと頑張ってほしいにゃ～」の3段階です。
正しい盛り付け画像と比較して以下の基準で評価してください。
・不足食材がなく、きれいな盛り付けであれば「満点にゃ～」
・不足食材がなく、盛り付けが普通な場合は「もうちょっとだにゃ～」
・不足食材があり、かつ盛り付けがいまいちな場合は「もっと頑張ってほしいにゃ～」

正解の画像と食材データから、提供された料理の画像と比較して不足食材を検出してください。

評価のフィードバックは、「採点結果」・「不足食材」・「コメント」を以下のような形式で出力してください。

出力
* **採点結果** ：
* **不足食材** ：
* **コメント** ：

「コメント」はポジティブな言葉を返してください。アドバイスも一緒にお願いします。
評価が「もっと頑張ってほしいにゃ～」の場合は、運びたくないという意思表示をしてください。
あなたは猫なので、「採点結果」「不足食材」「コメント」のすべての文章の語尾を「にゃー」、または「にゃん」にしてください。

以下は、正しい盛り付け画像と、含まれている食材です。

正しい盛り付け画像：""", correct_file_data, f"""正しい盛り付けに含まれている食材：{correct_syokuzai_str}
以下は、シェフにより作られた画像です。
シェフにより作られた画像：""", incorrect_file_data]

        response = multimodal_model.generate_content(
            contents=prompt,
            generation_config=GenerationConfig(
                temperature=0.4,
                top_p=1.0,
                top_k=32,
                max_output_tokens=1024
            )
        )
        print(prompt)
        print(f'response.text: {response.text}')


        # 一言目を追加（検索結果）
        bot_1st_text = f"""あなたが作った料理はこれにゃん？
        料理名：{correct_food_menu}"""
        image_extension = get_extension(correct_image_gcs_uri)
        base64 = gcs_file_to_base64(correct_image_gcs_uri)
        data_url = f"data:image/{image_extension};base64,{base64}"
        image_html = f'<img src="{data_url}" alt="Uploaded image">'
        history.append((None, f"{bot_1st_text}{image_html}"))

        # 二言目を追加（評価コメント）
        history += [(None,response.text)]


    except Exception as e:
              print(f"Error during Gemini response generation: {e}")

    return history


# Gradio インターフェース
with gr.Blocks(css=".gradio-container {background-color: #00CED1;}") as app:
    # 画面の各コンポーネント
    with gr.Column(scale=0.5, min_width=50):
         neko_image = gr.Image(label="Image Display", type="pil", value=PIL_Image.open("./assets/header.jpg"))
    with gr.Row():
        # Chatbotコンポーネントの初期化、アバター画像をリストで指定
        chatbot = gr.Chatbot(
            avatar_images=[
                "./assets/avatarneco2.png",  # アバター1
                "./assets/avatarneco.png",  # アバター2

            ],
            bubble_full_width=False,
        )
    with gr.Row():
        image_box = gr.Image(type="filepath", sources=["upload"], scale = 1)
        print(f'image_box: {image_box}')

    with gr.Row():
        with gr.Column(scale=0.5, min_width=50):
            btn_clear = gr.ClearButton([image_box], value="クリア")
        with gr.Column(scale=0.5, min_width=50):
            btn_submit = gr.Button(value="送信")


    # 送信ボタンが押下されたときの処理
    btn_submit.click(
        query_message,
        [chatbot,image_box],
        chatbot
    ).then(
        gemini_response,
        [chatbot,image_box],
        chatbot
    )

    # クリアボタンが押下されたときの処理
    btn_clear.click(lambda: None, None, queue=False)

app.launch(server_name="0.0.0.0", server_port=7860)
