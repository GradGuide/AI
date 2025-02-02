#!/bin/sh

# Download and install all models

python3 -m spacy download en_core_web_sm

python3 -c "import nltk; nltk.download('punkt'); nltk.download('punkt_tab')"
python3 -c "from scribe import Summary; Summary()"
python3 -c "from scribe import Similarity; Similarity()"
python3 -c "from scribe import QnA; QnA()"
python3 -c "from scribe import GrammarCorrector; GrammarCorrector()"

echo "[!] Done"
