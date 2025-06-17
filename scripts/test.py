#!/usr/bin/env python
# from scribe import LLM, similarity, QnA
from scribe import LLM, QnA, Similarity
from scribe.similarity import find_common_text
from markitdown import MarkItDown
import time


def timed(name, func, *args, **kwargs):
    print(f"⏳ Starting '{name}'...")
    start_time = time.time()
    result = func(*args, **kwargs)
    end_time = time.time()
    print(f"✅ Finished '{name}' in {end_time - start_time:.2f} seconds\n")
    return result


# load pdf file
md = MarkItDown(enable_plugins=False)
result = str(md.convert("/home/youssef/Downloads/research-paper.pdf"))


print("============= SUMMARY =============")

llm = LLM()

summary_gemini = timed("Gemini Summary", llm.summarize, result)

llm = LLM(provider="ollama")

summary_ollama = timed("Ollama Summary", llm.summarize, result)

print("============= Question & Answers =============")

qna = QnA()

questions = timed("Generate Questions", qna.generate_questions, result)
answers = timed(
    "Evaluate Answers",
    qna.evaluate_answers,
    context=result,
    questions=questions,
    user_answers=["i dont know"] * 5,
)


print("============= Similarity =============")

sim = Similarity()

sim_1 = timed("find common text", find_common_text, result, result)

sim_2 = timed("sbert_similarity", sim.sbert_similarity, sim_1["common_sentences"])

sim_3 = timed(
    "sbert_similarity", sim.tfidf_cosine_similarity, sim_1["common_sentences"]
)


print("============= Grammar Correction =============")

llm = LLM()

grammar_gemini = timed(
    "Grammar Corrector", llm.grammar_corrector, result, max_tokens=0x800
)


llm = LLM(provider="ollama")

grammar_ollama = timed(
    "Grammar Corrector", llm.grammar_corrector, result, max_tokens=0x800
)


print("Done")
