# The console and its API, with the tasks and the committed reports. Grading runs in the same image.
FROM node:20-bookworm-slim AS web
WORKDIR /web
COPY web/package.json web/package-lock.json ./
RUN npm ci --no-audit --no-fund
COPY web/ ./
RUN npm run build

FROM python:3.13-slim AS app
ENV PYTHONDONTWRITEBYTECODE=1 PYTHONUNBUFFERED=1 PIP_NO_CACHE_DIR=1 \
    FOUNDRY_DB=/data/reviews.sqlite FOUNDRY_WEB_DIR=/app/web/dist
WORKDIR /app
COPY pyproject.toml README.md ./
COPY src/ src/
RUN pip install .
COPY config/ config/
COPY tasks/ tasks/
COPY reports/ reports/
COPY --from=web /web/dist web/dist
RUN useradd --create-home --uid 10001 foundry && mkdir -p /data && chown foundry /data
USER foundry
ENV FOUNDRY_HOME=/app
EXPOSE 8000
HEALTHCHECK --interval=30s --timeout=3s CMD python -c "import urllib.request; urllib.request.urlopen('http://127.0.0.1:8000/healthz')"
CMD ["foundry", "serve", "--host", "0.0.0.0", "--port", "8000"]
