FROM python:3.11-slim

WORKDIR /app

# System dependencies for LIEF and pefile
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    && rm -rf /var/lib/apt/lists/*

# Copy dependency files first (Docker layer caching)
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy application code
COPY malvec/ malvec/
COPY setup.py .
COPY README.md .
RUN pip install --no-cache-dir -e .

# Non-root user for security (critical for malware analysis tool)
RUN useradd --create-home --shell /bin/bash malvec
USER malvec

# Default: show help
ENTRYPOINT ["python", "-m"]
CMD ["malvec.cli.classify", "--help"]
