FROM python:3.15-slim

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1

# Apply Debian security updates; upgrade pip/setuptools (known CVEs in the base versions).
RUN apt-get update && apt-get upgrade -y && rm -rf /var/lib/apt/lists/* \
 && pip install --upgrade pip setuptools wheel

WORKDIR /app
RUN pip install "streamlit>=1.35" "pandas>=2.1" "requests>=2.31" "altair>=5" "psycopg2-binary>=2.9" "pillow>=10"

COPY dashboard ./dashboard

RUN useradd --create-home appuser
USER appuser

EXPOSE 8501
CMD ["streamlit", "run", "dashboard/app.py", "--server.address=0.0.0.0", "--server.port=8501", "--server.headless=true", "--browser.gatherUsageStats=false"]
