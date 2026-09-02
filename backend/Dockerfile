FROM python:3.12-slim

WORKDIR /app
ENV PYTHONDONTWRITEBYTECODE=1 PYTHONUNBUFFERED=1

COPY pyproject.toml /app/pyproject.toml
RUN pip install --no-cache-dir ".[kafka]"

COPY app /app/app

# Nothing here needs root, and a container that keeps it hands any code
# execution bug the whole filesystem. The uid is fixed so the Kubernetes
# fsGroup can match it and the uploads volume stays writable.
RUN useradd --uid 10001 --create-home --shell /usr/sbin/nologin atlas \
    && mkdir -p /app/data/uploads \
    && chown -R atlas:atlas /app
USER atlas

EXPOSE 8000
CMD ["uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "8000"]
