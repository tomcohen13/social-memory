"""Prompts to use across pipelines"""


from langchain_core.messages import HumanMessage, SystemMessage
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.runnables import RunnableLambda

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

def _build_video_messages(inputs: dict) -> list:
    return [
        SystemMessage(content=(
            "You are a video analysis assistant. "
            "I will provide a video and a question. "
            "You MUST answer the question using ONLY the index (0, 1, 2, or 3) of the most likely correct answer."
        )),
        HumanMessage(content=[
            {"type": "media", "mime_type": "video/mp4", "data": inputs["video"]},
            {"type": "text", "text": f"QUESTION:\n{inputs['question']}\n\nANSWER CHOICES (Choose one):\n{inputs['options']}"},
        ]),
    ]


PROMPT_TEMPLATE_VIDEO = RunnableLambda(_build_video_messages)

PROMPT_REGISTRY = {
    PipelineNames.LANGUAGE: PROMPT_TEMPLATE_TRANSCRIPT,
    PipelineNames.VIDEO: PROMPT_TEMPLATE_VIDEO,
}
