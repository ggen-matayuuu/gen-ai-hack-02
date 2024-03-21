FROM python:3.10-slim
  
# ログメッセージの即時表示設定
ENV PYTHONUNBUFFERED True

# ローカルコードをコンテナイメージにコピー
ENV APP_HOME /app
WORKDIR $APP_HOME
COPY . ./

# 依存関係のインストール
RUN pip install --no-cache-dir -r requirements.txt

# ポート指定
EXPOSE 7860
  
# コンテナ起動時のコマンド
CMD ["python", "app.py"]