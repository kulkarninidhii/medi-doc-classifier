FROM python:3.11-slim

ENV PYTHONUNBUFFERED=1 PIP_NO_CACHE_DIR=1

RUN apt-get update && apt-get install -y --no-install-recommends \
    tesseract-ocr \
    poppler-utils \
    libgl1 \
 && rm -rf /var/lib/apt/lists/*

RUN pip install \
    pymupdf==1.24.4 \
    pytesseract==0.3.10 \
    pdf2image==1.17.0 \
    Pillow==10.3.0 \
    openai>=1.45.0 \
    packaging==24.1 \
    pandas==2.2.2   \
    httpx==0.27.0


WORKDIR /app

COPY extractor.py /app/extractor.py
COPY classification.py /app/classification.py

COPY run_extractor_batch.py /app/run_extractor_batch.py

ENTRYPOINT ["python", "-m"]
CMD ["run_extractor_batch"]
