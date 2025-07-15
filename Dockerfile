# CUDA-enabled runtime image
FROM nvidia/cuda:12.6.2-runtime-ubuntu22.04

WORKDIR /workspace
ENV PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1

# ── OS & build deps ───────────────────────────────────────────
#  * sox & libsox-dev   → provides the CLI that python-sox needs
#  * build-essential    → lets pip compile any stragglers (llvmlite, etc.)
RUN apt-get update && apt-get install -y --no-install-recommends \
        python3-pip python3-dev python3-venv \
        git curl build-essential sox libsox-dev \
        ffmpeg libsndfile1 && \
    rm -rf /var/lib/apt/lists/* && \
    ln -s /usr/bin/python3 /usr/bin/python && \
    python -m pip install --upgrade pip setuptools wheel

# ── Python deps ───────────────────────────────────────────────
# Pre-install NumPy so packages that `import numpy`
# during their setup don’t explode.
RUN python -m pip install numpy==1.26.4

RUN python -m pip install \
      torch==2.5.1+cu124 \
      torchaudio==2.5.1+cu124 \
      --extra-index-url https://download.pytorch.org/whl/cu124

COPY requirements.txt .
RUN python -m pip install -r requirements.txt

# ── App source & warm-ups ─────────────────────────────────────
COPY . .

# RUN python - <<'PY'
# from df.enhance import init_df
# init_df()
# PY

# RUN mkdir -p /root/.cache/torch/hub && \
#     git clone https://github.com/snakers4/silero-vad.git \
#         /root/.cache/torch/hub/snakers4_silero-vad_master && \
#     mkdir -p /workspace/temp


EXPOSE 7006
CMD ["sh", "-c", "python main.py --host 0.0.0.0 --port 7006 --workers ${UVICORN_WORKERS:-1}"]
