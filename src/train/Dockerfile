# official Python image
FROM python:3.8-slim

# working directory
WORKDIR /app

# copy the requirements file and install dependencies
COPY ../../requirements.txt .
RUN pip install --upgrade pip && pip install -r requirements.txt

# source code
COPY ../../src /app/src
COPY ../../data /app/data

CMD ["python", "src/train/train.py"]
