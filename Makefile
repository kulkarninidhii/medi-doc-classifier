# Makefile for Classification Pipeline

.PHONY: setup build run

# Setup environment
setup:
	@mkdir -p .exec
	@chmod 755 .exec
	@if [ ! -f .env ]; then \
		echo "Creating .env file..."; \
		echo "# Database Configuration" > .env; \
		echo "RESULTS_DSN=postgresql+psycopg://results:results@results-db:5432/results" >> .env; \
		echo "" >> .env; \
		echo "# AWS Configuration" >> .env; \
		echo "S3_BUCKET=your-bucket-name" >> .env; \
		echo "S3_PREFIX=documents/input" >> .env; \
		echo "AWS_ACCESS_KEY_ID=your-access-key" >> .env; \
		echo "AWS_SECRET_ACCESS_KEY=your-secret-key" >> .env; \
		echo "AWS_DEFAULT_REGION=us-east-1" >> .env; \
		echo "" >> .env; \
		echo "# OpenAI Configuration" >> .env; \
		echo "MODEL_NAME=gpt-4.1-mini" >> .env; \
		echo "OPENAI_API_KEY=sk-your-api-key-here" >> .env; \
		echo "" >> .env; \
		echo "✅ .env file created - please update with your credentials"; \
	else \
		echo "✅ .env file already exists"; \
	fi

# Build Docker images
build: setup
	docker-compose build --no-cache

# Run all services
run: setup
	docker-compose up -d
	@echo ""
	@echo "✅ Services started!"
	@echo "📊 Airflow UI: http://localhost:8080 (admin/admin)"
	@echo "📈 Streamlit: http://localhost:8501"