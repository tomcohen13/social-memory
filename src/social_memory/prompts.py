"""Prompts to use across pipelines"""


from langchain_core.prompts import ChatPromptTemplate


QA_TEMPLATE_TRANSCRIPT = ChatPromptTemplate.from_messages(
    [
        (
            "system", 
            """You are a video analysis assistant. 
            I will provide a transcript and a question. 
            You MUST answer the question using ONLY the index of the most likely correct answer to the question.
            Do not provide explanations or extra text."""
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