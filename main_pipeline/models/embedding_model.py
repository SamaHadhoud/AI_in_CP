from sentence_transformers import SentenceTransformer

class EmbeddingModelSingleton:
    _instance = None

    @classmethod
    def get_instance(cls):
        if cls._instance is None:
            print("Initializing embedding model for the first time...")
            cls._instance = cls()
        return cls._instance

    def __init__(self):
        self.model = SentenceTransformer('all-MiniLM-L6-v2')

    def encode(self, text):
        return self.model.encode(text)

def get_embedding_model():
    return EmbeddingModelSingleton.get_instance()