from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity
import itertools

# Load SBERT model
model_name = 'all-mpnet-base-v2'
model = SentenceTransformer(model_name)

def sbert(paragraphs):
    embeddings = model.encode(paragraphs)
    similarities = cosine_similarity(embeddings)
    return similarities


if __name__ == '__main__':

    paragraphs = [
        "The cat is sitting on the table.",
        "The feline is perched on the table.",
        "The kitty is resting on the table.",
        "My cat is hungry.",
        "my dog is hungry."
    ]
    
    similarities = sbert(paragraphs)
    for i, j in itertools.combinations(range(len(paragraphs)), 2):
        similarity_score = similarities[i][j]
        print(f"Similarity between {i+1} and paragraph {j+1}: {similarity_score}")
