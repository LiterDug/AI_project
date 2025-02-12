from transformers import AutoTokenizer, AutoModelForSeq2SeqLM, AutoConfig, pipeline, BitsAndBytesConfig
from langchain.llms import HuggingFacePipeline
import os
import torch
from langchain_google_genai import ChatGoogleGenerativeAI

def load_model(model_name, GOOGLE_API_KEY):
  llm = ChatGoogleGenerativeAI(model=model_name)
  os.environ["GOOGLE_API_KEY"] = GOOGLE_API_KEY
  return llm

def load_flan_t5_model():
    # Load model tokenzier 
    tokenizer = AutoTokenizer.from_pretrained("google/flan-t5-base")

    # Config model
    config = AutoConfig.from_pretrained("google/flan-t5-base", trust_remote_code=True)#, token=token)
    config.init_device = "cpu"
    config.temperature = 0.1

    # Quantize model
    bnb_config = BitsAndBytesConfig(
                                    load_in_4bit=True,
                                    bnb_4bit_use_double_quant=True,
                                    bnb_4bit_quant_type="nf4",
                                    bnb_4bit_compute_dtype=torch.bfloat16,
                                )

    # Load model
    model = AutoModelForSeq2SeqLM.from_pretrained("google/flan-t5-base",device_map="cpu", config=config)

    # Create pipeline
    pipe = pipeline(
        "text2text-generation",
        model=model, 
        tokenizer=tokenizer,
        max_length=300,
        temperature=0.1
    )

    # Create LangChain HF Pipeline
    llm = HuggingFacePipeline(pipeline=pipe)
    return llm



