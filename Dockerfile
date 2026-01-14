ARG BASE_IMAGE
FROM ${BASE_IMAGE}

WORKDIR /workspace

COPY requirements.txt /workspace/requirements.txt
RUN python3 -m pip install --no-cache-dir uv \
  && uv pip install --system --no-cache -r /workspace/requirements.txt

COPY . /workspace

ENV PYTHONUNBUFFERED=1
