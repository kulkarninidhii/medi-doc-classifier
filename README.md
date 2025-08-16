# Medical Document Classification Pipeline

An automated pipeline for classifying medical documents using OCR, NLP, and LLM technologies. Built with Apache Airflow, Docker, and OpenAI GPT models.

## 🎯 Overview

This pipeline automatically processes PDF documents from S3, extracts text using OCR when needed, classifies them into medical document types (prescriptions, physician notes, orders, etc.), and stores results in PostgreSQL with real-time visualization through Streamlit.

## 🛠️ Tech Stack

- **Orchestration**: Apache Airflow 2.x
- **Containerization**: Docker & Docker Compose
- **Databases**: PostgreSQL 15
- **Language**: Python 3.9+
- **ML/AI**: OpenAI GPT-4 models
- **OCR**: Tesseract + PyMuPDF
- **Storage**: AWS S3
- **Visualization**: Streamlit
- **Processing Libraries**: 
  - PyMuPDF (fitz) - PDF text extraction
  - Pytesseract - OCR processing
  - Pandas - Data manipulation
  - Psycopg3 - PostgreSQL adapter

## 📁 Project Structure

```
ashvinai/
├── airflow/
│   ├── dags/
│   │   └── doc_pipeline.py      # Main DAG definition
│   └── Dockerfile                # Airflow container configuration
├── db/
│   └── init.sql                  # Database schema initialization
├── src/
│   ├── extractor.py              # PDF text extraction with OCR
│   ├── classification.py         # 4-step LLM classification
│   ├── run_extractor_batch.py    # Batch extraction runner
│   └── Dockerfile                # Document processor container
├── streamlit/
│   ├── visuals.py                # Dashboard application
│   ├── requirements.txt          # Python dependencies
│   └── Dockerfile                # Streamlit container
├── docker-compose.yaml           # Service orchestration
├── .env                          # Environment variables (create this)
└── README.md                     # This file
```

## 📋 Requirements

### System Requirements
- Docker Engine 20.10+
- Docker Compose v3.9+
- 8GB RAM minimum
- 10GB free disk space
- Unix-based OS (Linux/macOS) or WSL2 on Windows

### External Services
- AWS Account with S3 access
- OpenAI API key with GPT-4 access

## 🚀 Local Setup

### 1. Clone Repository

```bash
git clone <repository-url>
cd ashvinai
```

### 2. Create Environment File

Create `.env` file in the root directory:

```bash
# Database Configuration
RESULTS_DSN=postgresql+psycopg://results:results@results-db:5432/results

# AWS Configuration
S3_BUCKET=your-bucket-name
S3_PREFIX=documents/input
AWS_ACCESS_KEY_ID=your-access-key
AWS_SECRET_ACCESS_KEY=your-secret-key
AWS_DEFAULT_REGION=us-east-1
S3_ENDPOINT_URL=  # Optional: for MinIO or S3-compatible services

# OpenAI Configuration
MODEL_NAME=gpt-4.1-mini  # Options: gpt-4.1, gpt-4.1-mini, gpt-5
OPENAI_API_KEY=sk-your-api-key-here

# Optional Configuration
RUN_ID=  # Leave empty for auto-generation
```

### 3. Create Required Directories

```bash
# Create execution volume mount point
mkdir -p .exec
chmod 755 .exec
```

### 4. Build Docker Images

```bash
# Build all services
docker-compose build

# Or build individually
docker-compose build airflow
docker-compose build docproc-image
docker-compose build streamlit-viz
```

### 5. Start Services

```bash
# Start all services in detached mode
docker-compose up -d

# Check if services are running
docker-compose ps

```

### 6. Access Services

| Service | URL | Credentials |
|---------|-----|-------------|
| Airflow UI | http://localhost:8080 | admin / admin |
| Streamlit Dashboard | http://localhost:8501 | No auth |
| PostgreSQL Results | localhost:5433 | results / results |

## 🏃 Running the Pipeline

### Upload Documents to S3

```bash
# Upload PDFs to your S3 bucket
aws s3 cp /path/to/pdfs/ s3://your-bucket-name/documents/input/ --recursive
```

### Trigger Pipeline

1. Open Airflow UI: http://localhost:8080
2. Login with admin/admin
3. Navigate to DAGs page
4. Find `doc_classification_pipeline`
5. Toggle ON (if not enabled)
6. Click trigger button (▶️)

### Monitor Progress

- **Airflow UI**: View task progress and logs
- **Streamlit**: Real-time visualization at http://localhost:8501
- **Logs**: `docker-compose logs -f airflow`

## 🧪 Testing

### Test PDF Extraction

```bash
# Create test directory
mkdir -p samples

# Copy a PDF to samples
cp /path/to/test.pdf samples/

# Run extraction
docker run --rm -v $(pwd)/samples:/data docproc:latest \
    python /app/extractor.py /data/test.pdf --outdir /data/output
```

### Test Classification

```bash
# Ensure OPENAI_API_KEY is set
docker run --rm \
    -e OPENAI_API_KEY=$OPENAI_API_KEY \
    -v $(pwd)/samples:/data \
    docproc:latest \
    python /app/classification.py /data/output --outdir /data/results
```

## 📚 Documentation

For detailed documentation including architecture, workflows, and scaling considerations, see [DOCUMENTATION.md](./DOCUMENTATION.md)

---

**Note**: Ensure compliance with HIPAA and healthcare data regulations when processing medical documents.