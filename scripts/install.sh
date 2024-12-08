#!/bin/sh

# Download and install all models

python -m spacy download en_core_web_sm

python -c "from scribe import Summary; Summary()"
python -c "from scribe import Similarity; Similarity()"
python -c "from scribe import QnA; QnA()"
python -c "from scribe import GrammarCorrector; GrammarCorrector()"

echo "[!] Done"
