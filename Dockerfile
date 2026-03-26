FROM continuumio/miniconda3:latest

WORKDIR /app

# Force Python to stream logs to Render dashboard
ENV PYTHONUNBUFFERED=1
ENV PYTHONDONTWRITEBYTECODE=1
ENV PYTHONPATH=/app

# Copy the machine contract (frozen dependency tree)
COPY conda-lock.yml .

# Install conda-lock, create environment from lockfile, then clean up
RUN conda install -c conda-forge conda-lock -y && \
    conda-lock install -n mlops conda-lock.yml && \
    apt-get update && \
    apt-get install -y curl && \
    apt-get clean && \
    rm -rf /var/lib/apt/lists/* && \
    conda clean -afy

# Activate the mlops environment
ENV PATH=/opt/conda/envs/mlops/bin:$PATH

# Copy only what the serving image needs
COPY config.yaml .
COPY src/ src/

EXPOSE 8000

# Health check so Render knows the service is alive
HEALTHCHECK --interval=30s --timeout=10s --start-period=15s --retries=3 \
    CMD curl --fail http://localhost:${PORT:-8000}/health || exit 1

# Use dynamic PORT for Render, fallback to 8000 locally
CMD ["sh", "-c", "uvicorn src.api:app --host 0.0.0.0 --port ${PORT:-8000}"]
