FROM python:3.12-slim

WORKDIR /app
ENV PYTHONDONTWRITEBYTECODE=1 PYTHONUNBUFFERED=1 HOME=/home/paperpilot
ENV PATH="/home/paperpilot/.local/bin:${PATH}"
# setpriv drops root privileges after the mounted volume's ownership is fixed.
RUN apt-get update && apt-get install -y --no-install-recommends util-linux && rm -rf /var/lib/apt/lists/*
RUN useradd --create-home paperpilot && mkdir /data && chown paperpilot:paperpilot /app
COPY requirements.txt ./
COPY step1_parse.py step2_chunk.py step3_embed.py step4_answer.py step7_api.py step9_summary.py step10_notes.py step11_cache.py step12_hybrid.py deploy.py auth.py ./
COPY entrypoint.sh ./
USER paperpilot
# Chroma caches ONNX at Path.home()/.cache/chroma/onnx_models. Building as the
# serving user keeps both that model and Chroma's read-write SQLite accessible.
RUN pip install --user --no-cache-dir -r requirements.txt
RUN python deploy.py build
USER root
EXPOSE 8080
ENTRYPOINT ["/bin/sh", "/app/entrypoint.sh"]
