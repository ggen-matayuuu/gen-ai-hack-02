# 生成 AI ハッカソン #2 チームA
## デプロイ手順

```
# 環境変数の設定
PROJECT_ID="gifted-mountain-415005"
REGION="asia-northeast1"
LOCATION="asia-northeast1"
FILE_BUCKET_NAME="gen-ai-hackson-by-jaguer-team-a-sampe-data"
AR_REPO="cloud-run-source-deploy"
SERVICE_NAME="neko-bot"

# 認証
gcloud auth login --billing-project=gifted-mountain-415005

# API の有効化
gcloud services enable --project=$PROJECT_ID  run.googleapis.com \
  artifactregistry.googleapis.com \
  cloudbuild.googleapis.com \
  compute.googleapis.com \
  aiplatform.googleapis.com

# Artifacts repositories 作成
gcloud artifacts repositories create $AR_REPO \
  --location=$LOCATION \
  --repository-format=Docker \
  --project=$PROJECT_ID

# イメージの作成＆更新
gcloud builds submit --tag $LOCATION-docker.pkg.dev/$PROJECT_ID/$AR_REPO/$SERVICE_NAME \
  --project=$PROJECT_ID

# Cloud Run デプロイ
gcloud run deploy $SERVICE_NAME --port 7860 \
  --allow-unauthenticated \
  --image $LOCATION-docker.pkg.dev/$PROJECT_ID/cloud-run-source-deploy/$SERVICE_NAME \
  --region=$LOCATION \
  --project=$PROJECT_ID \
  --memory=8Gi \
  --cpu=2 \
  --set-env-vars=PROJECT_ID=$PROJECT_ID \
  --set-env-vars=LOCATION=$LOCATION \
  --set-env-vars=FILE_BUCKET_NAME=$FILE_BUCKET_NAME 

```