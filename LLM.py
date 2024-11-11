from transformers import AutoConfig, AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig, pipeline
from langchain_huggingface import HuggingFacePipeline
from transformers import pipeline
from langchain_core.output_parsers import StrOutputParser
from prompt import Prompt_Template

class model():
  def __init__(self, model_id,AUTH_TOKEN):
    self.model_id=model_id
    self.AUTH_TOKEN= AUTH_TOKEN
  
  def config(self):
    # Configure for 4-bit quantization (optimizes model deployment)
    bnb_config = BitsAndBytesConfig(
      bnb_4bit_compute_dtype = 'float16',
      bnb_4bit_quant_type='nf4',
      load_in_4bit=True,
    )
    model_config = AutoConfig.from_pretrained(
      self.model_id,
      use_auth_token= self.AUTH_TOKEN
    )
    return bnb_config,model_config

  def create_model(self):
    bnb_config, model_config= self.config()
    # Load model
    model = AutoModelForCausalLM.from_pretrained(
      self.model_id,
      config=model_config,
      device_map='auto',
      quantization_config=bnb_config,
      use_auth_token=self.AUTH_TOKEN
    )

    # Load tokenizer
    tokenizer = AutoTokenizer.from_pretrained(
      self.model_id,
      use_auth_token= self.AUTH_TOKEN
    )

    model.eval()
    return model, tokenizer
  
  def initialize_pipeline(self):
    model, tokenizer= self.create_model()

    text_generation_pipeline = pipeline(
      "text-generation",
      model=model,
      tokenizer=tokenizer,
      batch_size=1
    )
    llm = HuggingFacePipeline(pipeline=text_generation_pipeline)

    prompt= Prompt_Template()
    llm_chain = prompt | llm | StrOutputParser()
    return llm_chain


  
  
