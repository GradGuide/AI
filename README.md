# GradGuide AI

## Installation

### Using virtualenv


make a python virtual enviroment

```shell
python -m venv .env

source .env/bin/activate
```

Install python libraries using:

```bash
pip install -r requirements
```

Then download all the models

```bash
sh ./scripts/install.sh
```

Then you are good to go.

### Using Docker

build the dockerfile

```
docker build -t scribe .
```

run it using

```bash
docker run --rm -it --net=host scribe
```

## Run experimental API

```shell
fastapi dev scribe/api.py
```

## Usage

### Summarization
```python
from scribe import Summary

# Initialize the Summary class
summarizer = Summary()

# Summarize text
text = "Artificial intelligence is transforming industries across the globe. It offers opportunities for innovation and growth."
summary = summarizer.bart_summarize(text)
print("Summary:", summary)
```

### Keyword Extraction
```python
# Extract keywords
keywords = summarizer.spacy_extract_keywords(text, num_keywords=5)
print("Keywords:", keywords)
```

### Text Similarity

#### BERT Similarity
```python
from scribe import Similarity

# Initialize the Similarity class
similarity_tool = Similarity()

# Compute similarity between two sentences using BERT
sentences = [
    "Artificial intelligence is fascinating.",
    "Machine learning is a subset of artificial intelligence."
]
similarity = similarity_tool.bert_similarity(sentences)
print("BERT Similarity:", similarity)
```

#### SBERT Similarity
```python
# Compute similarity for multiple paragraphs using SBERT
paragraphs = [
    "Artificial intelligence powers many modern applications.",
    "Deep learning and AI have revolutionized technology."
]
sbert_similarity = similarity_tool.sbert_similarity(paragraphs)
print("SBERT Similarity:", sbert_similarity)
```

#### TF-IDF Cosine Similarity
```python
# Compute similarity for sentences using TF-IDF
sentences = [
    "Natural language processing is a key area of AI.",
    "AI techniques are widely used in NLP."
]
tfidf_similarity = similarity_tool.tfidf_cosine_similarity(sentences)
print("TF-IDF Similarity:", tfidf_similarity)
```

### Question Answering
```python
from scribe import QnA

# Initialize the QnA class
qna_tool = QnA()

# Provide a question and context
question = "What is artificial intelligence?"
context = "Artificial intelligence (AI) is the simulation of human intelligence in machines that are programmed to think and learn."
answer = qna_tool.simple_question(question, context)
print("Answer:", answer)
```

### Grammar Correction with Diff
```python
from scribe import GrammarCorrector

grammar_tool = GrammarCorrector()

# Grammar correction
original_text = "This is an sentence with errors."
corrected_text = grammar_tool.correct(original_text)
print("Corrected Text:", corrected_text)

# Diff of changes
diff = grammar_tool.diff(original_text, corrected_text)
print("Diff:")
print(diff)
```

### Generative AI

#### Text Summarization
```python
from scribe import LLM

# Initialize the LLM class
llm_tool = LLM()

# Generate a summary
text = "Artificial intelligence and machine learning are rapidly advancing technologies."
summary = llm_tool.summarize(text)
print("Generated Summary:", summary)
```

#### Answer Questions
```python
# Answer a question based on context
question = "What is the use of AI in modern technology?"
context = "AI is widely used in applications such as virtual assistants, fraud detection, and personalized recommendations."
answer = llm_tool.answer_question(question, context)
print("Generated Answer:", answer)
```

#### Grammar Correction
```python
# Correct grammar and spelling
text = "AI have many aplications in todays world."
corrected_text = llm_tool.grammar_corrector(text)
print("Corrected Text:", corrected_text)
```


## License
بالحب
