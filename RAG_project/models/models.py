import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
import warnings
warnings.filterwarnings("ignore")

from transformers import AutoTokenizer, AutoModelForSeq2SeqLM, AutoConfig, pipeline, BitsAndBytesConfig
from langchain_community.llms import HuggingFacePipeline
import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
import torch
from langchain_google_genai import ChatGoogleGenerativeAI

import sys
sys.path.append(".")
from configs.config import Load_config
CONFIG = Load_config()

class Model_loader:
   def load_gemini(self):
      return ChatGoogleGenerativeAI(
         model= CONFIG.GOOGLE_MODEL,
         google_api_key= CONFIG.GOOGLE_API_KEY
   )

   def load_gemini_text(self):
      return ChatGoogleGenerativeAI(
         model= CONFIG.GOOGLE_TEXT_MODEL,
         google_api_key= CONFIG.GOOGLE_API_KEY
   )

   def load_flan_t5():
         model_name = CONFIG.FLAN_MODEL
         config = AutoConfig.from_pretrain(model_name)

         if CONFIG.FLAN_QUANTIZATION == "4bit":
            bnb_config = BitsAndBytesConfig(
               load_in_4bit=True,
               bnb_4bit_use_double_quant=True,
               bnb_4bit_quant_type="nf4",
               bnb_4bit_compute_dtype=torch.bfloat16,
            )
         else:
            bnb_config = None
            
         pipeline = pipeline(
            "text2text-generation",
            model= AutoTokenizer.from_pretrained(model_name,config=config),
            tokenizer= AutoTokenizer.from_pretrained(model_name),
            max_length= CONFIG.FLAN_MAX_LENGTH,
            temperature= CONFIG.FLAN_TEMPERATURE,
            quantization_config=bnb_config,
            device= "gpu" if torch.cuda.is_available() else "cpu",
         )
         
         llm = HuggingFacePipeline(pipeline)
         return llm

if __name__ == "__main__":
   # print(CONFIG.__dict__)
   llm = Model_loader.load_gemini()
   response = llm.invoke("What is the capital of France?")
   print(response.content)




#       def generate(self, input_text):
        
#         input_id = self.tokenizer(input_text, return_tensors="pt").input_ids.to(self.config.init_device)

#         output = self.model.generate(input_id)

#         return output






