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

    embedded_worker: bool = True
    log_level: str = "INFO"
    samples_dir: str = "../samples"
    upload_dir: str = "./data/uploads"


settings = Settings()
