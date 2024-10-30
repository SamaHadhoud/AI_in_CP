# from sentence_transformers import SentenceTransformer

# class EmbeddingModelSingleton:
#     _instance = None

#     @classmethod
#     def get_instance(cls):
#         if cls._instance is None:
#             print("Initializing embedding model for the first time...")
#             cls._instance = cls()
#         return cls._instance

#     def __init__(self):
#         # self.model = SentenceTransformer('flax-sentence-embeddings/st-codesearch-distilroberta-base')
#         self.model = SentenceTransformer('all-MiniLM-L6-v2')

#     def encode(self, text):
#         return self.model.encode(text)

# def get_embedding_model():
#     return EmbeddingModelSingleton.get_instance()

from sentence_transformers import SentenceTransformer

class EmbeddingModelSingleton:
    _instances = {}

    @classmethod
    def get_instance(cls, model_type='default'):
        if model_type not in cls._instances:
            print(f"Initializing {model_type} embedding model for the first time...")
            cls._instances[model_type] = cls(model_type)
        return cls._instances[model_type]

    def __init__(self, model_type):
        self.model_type = model_type
        if model_type == 'code':
            self.model = SentenceTransformer('flax-sentence-embeddings/st-codesearch-distilroberta-base')
        else:  # default model
            self.model = SentenceTransformer('BAAI/bge-m3')

    def encode(self, text):
        return self.model.encode(text)

def get_embedding_model(model_type='default'):
    return EmbeddingModelSingleton.get_instance(model_type)