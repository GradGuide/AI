from typing import List, Optional
import google.generativeai as genai
import os


class LLM:
    def __init__(self, api_key: Optional[str] = os.environ.get("GEMINI_API_KEY")):
        self.api_key = api_key
        genai.configure(api_key=self.api_key)
        self.model = genai.GenerativeModel("gemini-1.5-flash")

    def _generate_content(
        self,
        input_text: str,
        system_instruction: str,
        max_tokens: int,
        temperature: float,
        additional_instructions: Optional[List[str]] = None,
    ) -> genai.types.GenerateContentResponse:
        instructions = [system_instruction, input_text]
        if additional_instructions:
            instructions.extend(additional_instructions)

        response = self.model.generate_content(
            instructions,
            safety_settings={
                "HATE": "BLOCK_NONE",
                "HARASSMENT": "BLOCK_NONE",
                "SEXUAL": "BLOCK_NONE",
                "DANGEROUS": "BLOCK_NONE",
            },
            generation_config=genai.GenerationConfig(
                max_output_tokens=max_tokens,
                temperature=temperature,
            ),
        )

        return response

    def summarize(
        self,
        input_text: str,
        max_tokens: int = 64,
        temperature: float = 0.3,
        additional_instructions: Optional[List[str]] = None,
    ) -> str:
        """
        Generates a summary of the provided input text using the generative AI model.

        Parameters:
        ----------
        input_text : str
            The text that needs to be summarized.
        max_tokens : int, optional
            Maximum number of tokens in the output summary.
        temperature : float, optional
            The creativity level for the response.
        """
        system_instruction = f"You are an AI that provides summaries of the input text only using {max_tokens} words."
        return self._generate_content(
            input_text,
            system_instruction,
            max_tokens,
            temperature,
            additional_instructions,
        ).text.strip()

    def answer_question(
        self,
        question: str,
        context: str,
        max_tokens: int = 64,
        temperature: float = 0.3,
    ) -> str:
        """
        Answers a given question based on the provided context.

        Parameters:
        ----------
        question : str
            The question to answer.
        context : str
            The context information to answer the question from.
        max_tokens : int, optional
            Maximum number of tokens for the generated answer.
        temperature : float, optional
            The creativity level for the response.
        """
        system_instruction = "You are an AI assistant that answer questions. Answer the following question based only on the context provided and nothing more. keep the answer onpoint and short."
        return self._generate_content(
            f"Question:\n```{question}\n```",
            system_instruction,
            max_tokens,
            temperature,
            additional_instructions=[f"Context:\n```{context}\n```"],
        ).text.strip()

    def grammar_corrector(
        self,
        text: str,
        max_tokens: int = 64,
        temperature: float = 0.3,
        additional_instructions: Optional[List[str]] = None,
    ) -> str:
        """
        Correct the grammar and spelling of a given input text.

        Parameters:
        ----------
        text : str
            The text to be corrected.
        max_tokens : int, optional
            Maximum number of tokens.
        temperature : float, optional
            The creativity level for the response.
        """
        system_instruction = "You are an AI assistant that corrects grammar and spelling. rewrite the text and change what's necessary with no errors without any explaination."
        return self._generate_content(
            f"\n```{text}```",
            system_instruction,
            max_tokens,
            temperature,
            additional_instructions,
        ).text.strip()
