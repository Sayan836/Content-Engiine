from langchain_huggingface import HuggingFacePipeline
from langchain.prompts import PromptTemplate
from transformers import pipeline
from langchain_core.output_parsers import StrOutputParser


def Prompt_Template():
  prompt_template = """
  <|system|>
  You are a content analysis engine specialized in comparing and analyzing multiple documents. Your task is to identify and highlight the differences between the given documents. Provide a detailed analysis of the discrepancies, focusing on key changes, missing information, or conflicting details.

  Use the following context extracted from the documents to help you:

  {context}

  </s>
  <|user|>
  {question}
  </s>
  <|assistant|>
  """

  prompt = PromptTemplate(
    input_variables=["context", "question"],
    template=prompt_template,
  )
  return prompt