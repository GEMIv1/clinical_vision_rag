from pydantic_settings import BaseSettings

class Settings(BaseSettings):
    #OCR
    #MODEL_ENDPOINT: str
    MODEL_NAME: str
    #BACKUP_MODEL_ENDPOINT: str
    BACKUP_MODEL_NAME: str
    HF_TOKEN: str
    #Embedding
    #EMBED_ENDPOINT: str
    EMBED_MODEL: str
    #Groq LLM
    GROQ_API_KEY: str
    GROQ_MODEL: str
    #Vector DB
    CHROMA_PATH: str
    
    # HF_TOKEN: str
    # HF_MODEL: str

    class Config:
        env_file = ".env"

settings = Settings()