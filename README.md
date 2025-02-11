# GradGuide AI

Scribe is a powerful and easy-to-use library for natural language processing
(NLP) tasks. It integrates cutting-edge AI models to provide functionalities
such as text summarization, question answering, grammar correction, and text
similarity comparison.

Built on top of Google Gemini, Transformer models, and BERT-based similarity
techniques, Scribe enables developers to leverage advanced AI capabilities with
minimal setup.

**Table of Contents**

- [GradGuide AI](#gradguide-ai)
  - [Features](#features)
  - [Installation](#installation)
    - [Using virtualenv](#using-virtualenv)
    - [Using Docker](#using-docker)
  - [Run experimental API](#run-experimental-api)
  - [Usage](#usage)
    - [Summarization](#summarization)
    - [Text Similarity](#text-similarity)
    - [Question Answering](#question-answering)
    - [Grammar Correction with Diff](#grammar-correction-with-diff)
    - [Generative AI](#generative-ai)
- [Models Information](#models-information)
- [License](#license)


## Features

- ✅ Text Summarization – Generate concise summaries of long texts.
- ✅ Question Answering – Answer questions based on given context.
- ✅ Question Generation and Answer Evaluation.
- ✅ Grammar Correction – Fix grammar and spelling mistakes.
- ✅ Text Similarity – Compare texts using BERT, SBERT, and TF-IDF.
- ✅ Multi-language Support – Works in English, Arabic, French and more!.


## Installation

You can install it using a virtualenv or Docker.

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

**Keyword Extraction**:

```python
# Extract keywords
keywords = summarizer.spacy_extract_keywords(text, num_keywords=5)
print("Keywords:", keywords)
```

### Text Similarity

**BERT Similarity:**

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

**SBERT Similarity:**

```python
# Compute similarity for multiple paragraphs using SBERT
paragraphs = [
    "Artificial intelligence powers many modern applications.",
    "Deep learning and AI have revolutionized technology."
]
sbert_similarity = similarity_tool.sbert_similarity(paragraphs)
print("SBERT Similarity:", sbert_similarity)
```

**TF-IDF Cosine Similarity:**

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

for simple one word answer.

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

for question generation and evaluation (Requires [LLM](#generative-ai) API Key)

```python
qna = QnA()

text = """
Photosynthesis is the process used by plants, algae, and some bacteria to convert light energy into chemical energy.
It occurs in the chloroplasts of plant cells, primarily using chlorophyll to capture sunlight.
The process involves the intake of carbon dioxide and water, which are
transformed into glucose and oxygen.
"""

# generate questions
questions = qna.generate_questions(text, num_questions=5)
print("Generated Questions:", questions)

# ['What is photosynthesis?', 'Which organisms utilize photosynthesis?', 'Where does photosynthesis take place in plant cells?', 'What is the primary pigment used in photosynthesis?', 'What are the inputs and outputs of photosynthesis?']


# Evaluate answers
evaluations = qna.evaluate_answers(
    questions=["What is photosynthesis?"], # or use the generated questions
    user_answers=["the plants converts light energy into chemical energy"],
    context=text
)

print(evaluations)
# [('What is photosynthesis?', 6, 'The answer is partially correct, but lacks
# detail.  It correctly identifies the conversion of light energy into chemical
# energy. However, it omits key components such as the involvement of chlorophyll,
# carbon dioxide, water, glucose, and oxygen, and the location of the process
# (chloroplasts).')]
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

```python
from scribe import LLM

# Initialize the LLM module with an API key
llm = LLM(api_key="your_api_key")

# Example 1: Summarization
text_to_summarize = (
	"Artificial intelligence (AI) is intelligence demonstrated by machines, "
	"in contrast to the natural intelligence displayed by humans and animals. "
	"Leading AI textbooks define the field as the study of intelligent agents: "
	"any device that perceives its environment and takes actions that maximize "
	"its chance of achieving its goals."
	)
summary = llm.summarize(text_to_summarize, max_tokens=50, temperature=0.5)
print("Summary:", summary)

# Example 2: Answering Questions
context = (
	"The Eiffel Tower is a wrought-iron lattice tower on the Champ de Mars in Paris, "
	"France. It is named after the engineer Gustave Eiffel, whose company "
	"designed and built the tower."
	)
question = "Who designed the Eiffel Tower?"
answer = llm.answer_question(question, context, max_tokens=30, temperature=0.3)
print("Answer:", answer)

# Example 3: Grammar Correction
incorrect_text = "She go to the market every day."
corrected_text = llm.grammar_corrector(incorrect_text, max_tokens=30, temperature=0.3)
print("Corrected Text:", corrected_text)

# Example 4: Summarization with Additional Instructions and Language Support
summary_french = llm.summarize(text_to_summarize, max_tokens=50, temperature=0.5, language="French")
print("Summary in French:", summary_french)

# Example 5: Answering a Question with Additional Instructions
question_with_instruction = llm.answer_question(
    "What is the significance of the Eiffel Tower?",
    context,
    max_tokens=50,
    temperature=0.3,
    language="English"
)
print("Detailed Answer:", question_with_instruction)
```

# Models Information

| Module           | Model Name                        | Trained On                                                                         | Accuracy / Performance                                                                               |
|------------------|-----------------------------------|------------------------------------------------------------------------------------|------------------------------------------------------------------------------------------------------|
| LLM              | `gemini-1.5-flash`                | Proprietary Google dataset                                                         | -                                                                                                    |
| Similarity       | `t5-base`                         | [C4 (Clossal Clean Crawled Corpus)](https://paperswithcode.com/dataset/c4) dataset | GLUE = ~83.38%                                                                                       |
| Summary          | `all-mpnet-base-v2`               | [MNLI](https://huggingface.co/datasets/nyu-mll/multi_nli) dataset                  | 87.47% using [nli-roberta-base](https://huggingface.co/cross-encoder/nli-roberta-base) for benchmark |
| QnA              | `roberta-base-squad2`             | [SQuAD 2.0](https://huggingface.co/datasets/rajpurkar/squad_v2) dataset            | EM = 76.87%, F1=80.9%                                                                                |
| GrammarCorrector | `flan-t5-large-grammar-synthesis` | [jfleg](https://huggingface.co/datasets/jhu-clsp/jfleg) dataset                    | GLEU ≈ 76%                                                                                           |


# License
بالحب
