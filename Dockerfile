FROM python:3.10.16-slim-bookworm@sha256:f9fd9a142c9e3bc54d906053b756eb7e7e386ee1cf784d82c251cf640c502512
ENV PYTHONDONTWRITEBYTECODE=1 PYTHONUNBUFFERED=1 OMP_NUM_THREADS=4 OPENBLAS_NUM_THREADS=4
WORKDIR /app
COPY requirements.txt .
RUN pip install --no-cache-dir --upgrade pip==25.1.1
RUN pip install --no-cache-dir -r requirements.txt \
    'https://download.pytorch.org/whl/cpu/torch-2.5.1%2Bcpu-cp310-cp310-linux_x86_64.whl' \
    'https://download.pytorch.org/whl/cpu/torchvision-0.20.1%2Bcpu-cp310-cp310-linux_x86_64.whl'
ENV HF_HUB_OFFLINE=1
RUN pip install --no-cache-dir timm==1.0.22 huggingface-hub==0.35.3 safetensors==0.6.2 PyYAML==6.0.3
RUN pip install --no-cache-dir segmentation-models-pytorch==0.5.0 && pip check
RUN apt-get update && apt-get install -y --no-install-recommends fonts-dejavu-core libgomp1 && rm -rf /var/lib/apt/lists/*
COPY dxa dxa
COPY web web
COPY vendor/vertebra_landmark_detection vendor/vertebra_landmark_detection
COPY scripts/deploy scripts/deploy
COPY runtime-data/app/ /app/
COPY runtime-data/data/ /data/
RUN useradd -m -u 10001 dxa && mkdir -p /runtime/jobs /runtime/annotations && chown -R dxa:dxa /runtime
ENV DXA_JOBS=/runtime/jobs DXA_ANNOTATIONS=/runtime/annotations DXA_AUTH_FILE=/run/secrets/dxa-auth.json
USER dxa
EXPOSE 8095
HEALTHCHECK --interval=30s --timeout=5s --start-period=30s CMD ["python", "scripts/deploy/healthcheck.py"]
CMD ["python", "-m", "uvicorn", "dxa.api:app", "--host", "0.0.0.0", "--port", "8095"]
