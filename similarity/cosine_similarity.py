#!/usr/bin/env python
from typing import List
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity


sentences = [
    "The cat is sitting on the windowsill.",
    "The feline is perched on the windowsill.",
    "The kitty is resting on the windowsill.",
    "The cat is hungry.",
    "The cat needs food",
]



def similarity(sentences) -> List[List[float]]:
    vectorizer = TfidfVectorizer().fit_transform(sentences)
    vectors = vectorizer.toarray()
    cosine_sim = cosine_similarity(vectors)
    return cosine_sim


if __name__ == '__main__':
    print(similarity(sentences))
