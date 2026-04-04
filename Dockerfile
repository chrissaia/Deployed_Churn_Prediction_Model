# this specifies the base image that the build will extend.
FROM python:3.11-slim
WORKDIR /app

# Install the application dependencies
COPY requirements.txt .

RUN pip install --upgrade pip \
    && pip install --no-cache-dir -r requirements.txt \
    && apt-get clean && rm -rf /var/lib/apt/lists/*


# tells the builder to copy all files from the host and put them into the container imagw
COPY . .

# Explicitly copy model (in case .dockerignore excluded mlruns)
#NOTE: destination changed to /app/src/serving/model to match inference.py's path
COPY src/serving/model /app/src/serving/model
# Copy MLflow run (artifacts + metadata) to the flat /app/model convenience path
COPY src/serving/model/m-287af1f0418a4910a8c9fb9fb678b2c4/artifacts/model /app/model
COPY src/serving/model/m-287af1f0418a4910a8c9fb9fb678b2c4/artifacts/feature_columns.json /app/model/feature_columns.json
COPY src/serving/model/m-287af1f0418a4910a8c9fb9fb678b2c4/artifacts/preprocessing.pkl /app/model/preprocessing.pkl

# l
ENV PYTHONUNBUFFERED=1 \
    PYTHONPATH=/app/src \
    MPLCONFIGDIR=/tmp/matplotlib

# sets configuration on the image that indicates a port the image would like to expose.
EXPOSE 8080


# Setup an app user so the container doesn't run as the root user
RUN useradd app
# sets the default user for all subsequent instructions.
USER app

# sets the default command a container using this image will run.
CMD ["python", "-m", "uvicorn", "src.app.main:app", "--host", "0.0.0.0", "--port", "8080"]