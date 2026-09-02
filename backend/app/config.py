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
    # Calibrated, not guessed. scripts/calibrate_cache.py measures cosine
    # similarity for two sets of question pairs: the same question reworded,
    # and questions one decisive word apart. At 0.82 it catches 6 of 9
    # paraphrases with no wrong hit, and clears the worst false pair
    # (Seattle sick leave against the China cap, 0.768) by five points.
    # The old 0.92 caught 2 of 9 -- and nothing at all in practice, because
    # the stored vector described a different text than the lookup.
    semantic_cache_threshold: float = 0.82
    qa_cache_collection: str = "atlas_qa_cache"

    # Off by default: measured at zero benefit on this corpus, and the model
    # needs the optional [rerank] extra. See "What each stage actually buys".
    enable_cross_encoder: bool = False
    cross_encoder_model: str = "cross-encoder/ms-marco-MiniLM-L-6-v2"

    # "principal:secret,principal:secret". Empty disables auth entirely and
    # every caller becomes the dev principal -- see app/auth.py.
    #
    # The alias is load-bearing: without it pydantic-settings binds this field
    # to API_KEYS, silently ignores the ATLAS_API_KEYS the deployment sets, and
    # the service starts with auth off while looking configured.
    api_keys: str = Field(default="", validation_alias="ATLAS_API_KEYS")
    # 60 was a guess, and scripts/loadtest.py showed it was the binding
    # constraint long before the service was: one replica serves ~590 rps of
    # cached answers and ~1 rps once the model is in the path, so 60/min
    # rate-limited a single user reading their own follow-ups. The limit is
    # here to bound spend, not to protect the app, so it sits well above any
    # interactive session and well below what a runaway client could burn.
    rate_limit_per_minute: int = 300
    # Housekeeping tick: corpus gauges, and on a slower multiple of it the
    # paraphrase-cache sweep. Retrieval no longer waits on this -- dense and
    # sparse both live in Qdrant and are current as soon as a write lands.
    refresh_seconds: float = 2.0
    # How often to check that the catalogue and the vector store still agree.
    # It scrolls the whole collection, so it is deliberately slow and runs in a
    # thread; 0 leaves reconciliation to startup and the endpoint. Divergence
    # comes from events -- a failed ingest, a migration -- rather than drift,
    # so this is insurance, not the mechanism.
    reconcile_seconds: float = 300.0
    # Port the worker exposes its scrape endpoint on. The API serves /metrics
    # from its own FastAPI app and ignores this. 0 disables.
    metrics_port: int = 9100
    # Same-origin in the container image (nginx proxies /api), so this only
    # needs the Vite dev server.
    cors_origins: str = "http://localhost:5173,http://127.0.0.1:5173"

    # An upload was read fully into memory before anything checked its size,
    # so a body larger than the container limit was an OOM kill rather than a
    # 413. 20MiB clears the documents this is built for and sits far under the
    # 2Gi the pod is allowed.
    max_upload_bytes: int = 20 * 1024 * 1024

    embedded_worker: bool = True
    log_level: str = "INFO"
    samples_dir: str = "../samples"
    upload_dir: str = "./data/uploads"


settings = Settings()
