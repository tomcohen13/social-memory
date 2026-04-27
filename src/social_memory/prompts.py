"""Prompts to use across pipelines"""


from langchain_core.messages import HumanMessage, SystemMessage
from langchain_core.prompts import ChatPromptTemplate, HumanMessagePromptTemplate
from langchain_core.runnables import RunnableLambda

from social_memory.constants import PipelineNames


PROMPT_TEMPLATE_TRANSCRIPT = ChatPromptTemplate.from_messages(
    [
        SystemMessage(
            content="""You are a video analysis assistant. 
            I will provide a transcript and a question. 
            You MUST answer the question using ONLY the index (0, 1, 2, or 3) of the most likely correct answer.
            """
        ),
        HumanMessagePromptTemplate.from_template(
            template="""TRANSCRIPT:
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

def _build_audio_messages(inputs: dict) -> list:
    return [
        SystemMessage(content=(
            "You are an audio analysis assistant. "
            "I will provide an audio recording and a question. "
            "You MUST answer the question using ONLY the index (0, 1, 2, or 3) of the most likely correct answer."
        )),
        HumanMessage(
            content=[
            {
                "type": "text",
                "text": f"""
                QUESTION: {inputs['question']}
                
                ANSWER CHOICES (Choose one):
                {inputs['options']}
                """
            },
            {
                "type": "input_audio",
                "input_audio": {
                    "data": inputs["audio"], # base64-encoded audio
                    "format": inputs["mime_type"],
                },
            },
        ]),
    ]

PROMPT_TEMPLATE_AUDIO = RunnableLambda(_build_audio_messages)


PROMPT_TEMPLATE_SENSORY_DEPRIVATION = ChatPromptTemplate.from_messages(
    [
        SystemMessage(
            content="""You are a question-answering assistant. 
            I will provide a question and answer choices. 
            You MUST answer the question using ONLY the index (0, 1, 2, or 3) of the most likely correct answer.
            """
        ),
        HumanMessagePromptTemplate.from_template(
            'QUESTION:\n{question}\n\n'
            'ANSWER CHOICES (Choose one):\n{options}'
        )
    ]
)

PROMPT_REGISTRY = {
    PipelineNames.LANGUAGE: PROMPT_TEMPLATE_TRANSCRIPT,
    PipelineNames.VIDEO: PROMPT_TEMPLATE_VIDEO,
    PipelineNames.AUDIO: PROMPT_TEMPLATE_AUDIO,
    "sensory_deprivation": PROMPT_TEMPLATE_SENSORY_DEPRIVATION,
    "video_clipped": PROMPT_TEMPLATE_VIDEO,
}
