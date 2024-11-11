# GradGuide AI

## Installation

install dependencies with `pip install -r requirements.txt`

## Run experimental API

```shell
fastapi dev scribe/api.py
```

## API Reference
### `QnA`

- `simple_question(question: str, context: str) -> Dict[str, Union[str, float]]`: Answers a simple question based on the provided context.

### `LLM`

- `summarize(input_text: str, max_tokens: int = 64, temperature: float = 0.3) -> str`: Summarizes the input text.
- `answer_question(question: str, context: str, max_tokens: int = 64, temperature: float = 0.3) -> str`: Answers a question based on the provided context.
- `grammar_corrector(text: str, max_tokens: int = 64, temperature: float = 0.3) -> str`: Corrects grammar and spelling in the input text.

### `Summary`

- `bart_summarize(text: str, min_length: int = 5, max_length: int = 20) -> List[str]`: Summarizes the input text using BART.
- `spacy_extract_keywords(text: str, num_keywords: int = 10) -> List[Tuple[str, int]]`: Extracts keywords from the input text using spaCy.

### `Similarity`

- `bert_similarity(sentences: List[str]) -> float`: Computes similarity between two sentences using BERT.
- `sbert_similarity(paragraphs: List[str]) -> List[List[float]]`: Computes similarity between multiple paragraphs using SBERT.
- `tfidf_cosine_similarity(sentences: List[str]) -> List[List[float]]`: Computes similarity between multiple sentences using TF-IDF.


## License
بالحب
