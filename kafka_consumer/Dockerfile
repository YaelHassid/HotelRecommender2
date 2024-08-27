# Use the official Python 3.10 image as the base image
FROM python:3.10

# Set the working directory
WORKDIR /app

# Copy the requirements file and install dependencies
COPY requirements.txt /app/
RUN pip install --no-cache-dir -r requirements.txt

# Install system dependencies
RUN apt-get update && \
    apt-get install -y \
    gcc \
    libssl-dev \
    && apt-get clean && \
    rm -rf /var/lib/apt/lists/*

# Copy the application code to the container
COPY . /app/

# Set the entrypoint for the container
CMD ["python", "kafka_consumer.py"]
