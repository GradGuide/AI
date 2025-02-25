from fastapi import FastAPI, HTTPException, Query
from typing import List
import uvicorn
import logging

from .grammar import GrammarCorrector
from .llm import LLM
from .qna import QnA
from .similarity import Similarity, find_common_text
from .summary import Summary

logging.basicConfig(level=logging.ERROR)
logger = logging.getLogger(__name__)


app = FastAPI(
    title="Scribe API",
    description="Use scribe functinos through an API",
    version="1.0.0",
)

try:
    llm = LLM()
    summary_model = Summary()
    similarity_calculator = Similarity()
    qna_model = QnA()
    grammar_corrector = GrammarCorrector()
except Exception as e:
    print(f"Error initializing models: {e}")
    exit(1)

# === API Endpoints ===


@app.post("/llm/summarize")
async def llm_summarize_endpoint(
    text: str = Query(..., description="Text to summarize"),
    max_tokens: int = Query(100, description="Maximum tokens in summary"),
    temperature: float = Query(0.3, description="Temperature for LLM"),
):
    try:
        summary_text = llm.summarize(text, max_tokens, temperature)
        return {"summary": summary_text}
    except Exception as e:
        raise HTTPException(
            status_code=500, detail=f"Error summarizing: {str(e)}"
        )  # Convert error to string


@app.post("/llm/grammar_correct")
async def llm_grammar_correct_endpoint(
    text: str = Query(..., description="Text to correct"),
):
    try:
        corrected_text = llm.grammar_corrector(text)
        return {"corrected_text": corrected_text}
    except Exception as e:
        raise HTTPException(
            status_code=500, detail=f"Error correcting grammar: {str(e)}"
        )


@app.post("/llm/answer_question")
async def llm_answer_question_endpoint(
    question: str = Query(..., description="Question to answer"),
    context: str = Query(..., description="Context for answering"),
):
    try:
        answer = llm.answer_question(question, context)
        return {"answer": answer}
    except Exception as e:
        raise HTTPException(
            status_code=500, detail=f"Error answering question: {str(e)}"
        )


@app.post("/grammar/correct")
async def grammar_correct_endpoint(
    text: str = Query(..., description="Text to correct"),
):
    try:
        corrected_text = grammar_corrector.correct(text)
        return {"corrected_text": corrected_text}
    except Exception as e:
        raise HTTPException(
            status_code=500, detail=f"Error correcting grammar: {str(e)}"
        )


@app.post("/similarity/sbert")
async def sbert_similarity_endpoint(
    sentences: List[str] = Query(..., description="List of sentences"),
):
    try:
        similarity_matrix = similarity_calculator.sbert_similarity(sentences)
        return {"similarity_matrix": similarity_matrix}
    except Exception as e:
        raise HTTPException(
            status_code=500, detail=f"Error calculating similarity: {str(e)}"
        )


@app.post("/similarity/tfidf_cosine")
async def tfidf_cosine_similarity_endpoint(
    sentences: List[str] = Query(..., description="List of sentences"),
):
    try:
        similarity_matrix = similarity_calculator.tfidf_cosine_similarity(sentences)
        return {"similarity_matrix": similarity_matrix}
    except Exception as e:
        raise HTTPException(
            status_code=500,
            detail=f"Error calculating TF-IDF cosine similarity: {str(e)}",
        )


@app.post("/similarity/bert")
async def bert_similarity_endpoint(
    sentences: List[str] = Query(..., description="List of sentences"),
):
    try:
        similarity_matrix = similarity_calculator.bert_similarity(sentences)
        return {"similarity_matrix": similarity_matrix}
    except Exception as e:
        raise HTTPException(
            status_code=500, detail=f"Error calculating BERT similarity: {str(e)}"
        )


@app.post("/similarity/find_common_text")
async def find_common_text_endpoint(
    text1: str = Query(..., description="First text"),
    text2: str = Query(..., description="Second text"),
):
    try:
        common_text = find_common_text(text1, text2)
        return common_text
    except Exception as e:
        raise HTTPException(
            status_code=500, detail=f"Error finding common text: {str(e)}"
        )


@app.post("/qna/generate_questions")
async def qna_generate_questions_endpoint(
    text: str = Query(..., description="Text to generate questions from"),
    num_questions: int = Query(5, description="Number of questions"),
):
    try:
        questions = qna_model.generate_questions(text, num_questions)
        return {"questions": questions}
    except Exception as e:
        raise HTTPException(
            status_code=500, detail=f"Error generating questions: {str(e)}"
        )


@app.post("/qna/evaluate_answers")
async def qna_evaluate_answers_endpoint(
    questions: List[str] = Query(..., description="List of questions"),
    user_answers: List[str] = Query(..., description="List of user answers"),
    context: str = Query(..., description="Context for evaluating answers"),
):
    try:
        results = qna_model.evaluate_answers(questions, user_answers, context)
        return {"results": results}
    except Exception as e:
        raise HTTPException(
            status_code=500, detail=f"Error evaluating answers: {str(e)}"
        )


if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000)
