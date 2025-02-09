FROM python:3.8-slim

RUN apt-get update && apt-get install -y build-essential

ENV TMPDIR=/tmp

WORKDIR /app


RUN python -m pip install --upgrade pip
COPY requirements.txt .
RUN pip install --no-cache-dir --default-timeout=100 -r requirements.txt || pip install -r requirements.txt

COPY . .

CMD ["python", "src/inference/run_inference.py"]
