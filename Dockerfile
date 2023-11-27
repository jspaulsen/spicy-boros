# python 3.10 latest
FROM python:3.10-slim-bookworm

ENV MODEL_PATH="/app/models"
ENV MODEL_NAME="jondurbin/spicyboros-7b-2.2"


RUN \
    mkdir /app

WORKDIR /app

COPY Pipfile Pipfile.lock /app/


# Install package dependencies
RUN \
    pip install pipenv && \
    pipenv requirements > requirements.txt && \
    pip install -r requirements.txt


COPY download_models.py /app/


# Download models
RUN \
    python download_models.py


# Copy over source directory app into /app (so it's /app/app)
COPY app /app/app


# Entrypoint is the main.py file
ENTRYPOINT [ "python", "-m" ]
CMD ["app.main"]


# Copy necessary files to the container
# COPY requirements.txt .
# COPY main.py .
# COPY download_models.py .
