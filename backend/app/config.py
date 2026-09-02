from pydantic import Field
from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    model_config = SettingsConfigDict(env_file=".env", extra="ignore")

    openai_api_key: str = ""
    openai_base_url: str = "https://api.openai.com/v1"
    chat_model: str = "gpt-4o"
    cheap_model: str = "gpt-4o-mini"
    embedding_model: str = "text-embedding-3-small"
    embedding_dim: int = 1536

    redis_url: str = "redis://localhost:6379/0"
    qdrant_url: str = "http://localhost:6333"
    qdrant_collection: str = "atlas_chunks"

    kafka_brokers: str = ""
    kafka_topic: str = "atlas.index.jobs"
    index_stream: str = "atlas:index"

    token_budget: int = 1800
    retrieve_k: int = 24
    rerank_k: int = 6
    max_retrieve_retries: int = 2
    semantic_cache_threshold: float = 0.92

    enable_cross_encoder: bool = True
    cross_encoder_model: str = "cross-encoder/ms-marco-MiniLM-L-6-v2"

    # "principal:secret,principal:secret". Empty disables auth entirely and
    # every caller becomes the dev principal -- see app/auth.py.
    #
    # The alias is load-bearing: without it pydantic-settings binds this field
    # to API_KEYS, silently ignores the ATLAS_API_KEYS the deployment sets, and
    # the service starts with auth off while looking configured.
    api_keys: str = Field("", validation_alias="ATLAS_API_KEYS")
    rate_limit_per_minute: int = 60
    # How often the API re-checks whether the worker has reindexed. Sparse
    # retrieval can be this far behind an ingest; dense retrieval sees it at
    # once, because that lives in Qdrant rather than in process memory.
    bm25_refresh_seconds: float = 2.0
    # Port the worker exposes its scrape endpoint on. The API serves /metrics
    # from its own FastAPI app and ignores this. 0 disables.
    metrics_port: int = 9100
    # Same-origin in the container image (nginx proxies /api), so this only
    # needs the Vite dev server.
    cors_origins: str = "http://localhost:5173,http://127.0.0.1:5173"

    embedded_worker: bool = True
    log_level: str = "INFO"
    samples_dir: str = "../samples"
    upload_dir: str = "./data/uploads"


settings = Settings()
