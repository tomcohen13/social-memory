"""Prompts to use across pipelines"""


from langchain_core.prompts import ChatPromptTemplate

from social_memory.constants import PipelineNames


PROMPT_TEMPLATE_TRANSCRIPT = ChatPromptTemplate.from_messages(
    [
        (
            "system", 
            """You are a video analysis assistant. 
            I will provide a transcript and a question. 
            You MUST answer the question using ONLY the index (0, 1, 2, or 3) of the most likely correct answer.
            """
        ),
        
        (
            "human",
            """TRANSCRIPT:
            {transcript}
            
            QUESTION: 
            {question}
            
            ANSWER CHOICES (Choose one):
            {options}"""
        )
    ]
)

PROMPT_REGISTRY = {
    PipelineNames.LANGUAGE: PROMPT_TEMPLATE_TRANSCRIPT,
}
