# official Python image
FROM python:3.8-slim

WORKDIR /app

# Copy the requirements file and install dependencies
COPY ../../requirements.txt .
RUN pip install --upgrade pip && pip install -r requirements.txt

# Copy source code and data
COPY ../../src /app/src
COPY ../../data /app/data
COPY ../../outputs /app/outputs

CMD ["python", "src/inference/run_inference.py"]
