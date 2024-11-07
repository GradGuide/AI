import google.generativeai as genai
import os


class LLM:
    def __init__(self, api_key: str = os.environ.get("GEMINI_API_KEY")):
        self.api_key = api_key
        genai.configure(api_key=self.api_key)
        self.model = genai.GenerativeModel("gemini-1.5-flash")

    def summarize(
        self, input_text: str, max_tokens: int = 64, temperature: float = 0.3
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

        response = self.model.generate_content(
            [system_instruction, input_text],
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

        return response.text
