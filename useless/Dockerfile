# Use python:3.11-slim as the base image
FROM python:3.11-slim

# Set the working directory
WORKDIR /my_app

# RUN mkdir -p /tmp/sockets

RUN apt-get update && apt-get install -y \
    build-essential \
    python3-dev \
    libffi-dev \
    libssl-dev \
    && rm -rf /var/lib/apt/lists/*

# Copy requirements.txt and install dependencies
COPY requirements.txt .
RUN pip install --upgrade pip
RUN pip install --no-cache-dir -r requirements.txt

# Copy the application code
COPY ./src /my_app
# COPY ./nginx.conf /etc/nginx/nginx.conf

# Expose the application port
EXPOSE 5000

# Command to run the uWSGI server
# CMD ["uwsgi", "--ini", "/src/uwsgi.ini"]
CMD ["python", "-m","chat.app"]

