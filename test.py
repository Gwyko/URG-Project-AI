from langchain_community.llms import Ollama
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser

LLM_model = "mistral" 
llm = Ollama(model=LLM_model)

enhancement_prompt = ChatPromptTemplate.from_template(
    """You are an AI assistant that specializes in rephrasing user questions to be more precise and include relevant keywords for better search results.
    Given the user's question, rewrite it to be more specific and include potential keywords that would help find accurate information.

    Original Question: {original_question}

    Rewritten Question (more specific, with keywords):"""
)

print("="*200)

question_enhancer_chain = (
    enhancement_prompt
    | llm
    | StrOutputParser() # To get just the text output
)