FROM python:3.11-slim

WORKDIR /app

COPY requirements.txt .

RUN pip install --no-cache-dir --timeout=120 -r requirements.txt

COPY . .

EXPOSE 7860

ENV API_BASE_URL="https://api-inference.huggingface.co/v1"
ENV MODEL_NAME="meta-llama/Llama-3.3-70B-Instruct"
ENV HF_TOKEN=""
ENV OPENAI_API_KEY=""

CMD ["uvicorn", "app:app", "--host", "0.0.0.0", "--port", "7860"]