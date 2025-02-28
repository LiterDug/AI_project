import yaml
import pandas as pd
import sys
import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

def load_config(path):
    with open(path, "r") as f:
        config = yaml.safe_load(f)
    return config

def load_data(path):
    data = pd.read_csv(path)
    return data

class Load_config:
    def __init__(self):
        # dataset = load_data("data/data.csv")
        config = load_config("configs/config.yml")

        # Gemini config
        self.GOOGLE_MODEL = config["llm_google"]["model"]
        self.GOOGLE_API_KEY = config["llm_google"]["api_key"]
        self.GOOGLE_TEMPERATURE = config["llm_google"]["temperature"]
        self.GOOGLE_TEXT_MODEL = config["llm_google"]["text_model"]

        # Google flan t5
        self.FLAN_MODEL = config["llm_flan_t5"]["model"]
        self.FLAN_TEMPERATURE = config["llm_flan_t5"]["temperature"]
        self.FLAN_MAX_LENGTH = config["llm_flan_t5"]["max_length"]
        self.FLAN_QUANTIZATION = config["llm_flan_t5"]["quantization"]

        # RAG config
        self.MEMORY_K = 50
    

# if __name__ == "__main__":
   # print(CONFIG.__dict__)