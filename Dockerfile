FROM python:3.10-slim
  
COPY ./app /app
  
WORKDIR /app
  
RUN pip install --no-cache-dir -r requirements.txt
  
EXPOSE 7860
  
CMD ["python", "app.py"]