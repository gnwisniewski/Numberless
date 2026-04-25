from transformers import AutoProcessor, AutoModelForCausalLM
from PIL import Image
import torch

MODEL_NAME = "microsoft/Florence-2-base"

def load_model():
    processor = AutoProcessor.from_pretrained(MODEL_NAME, trust_remote_code=True)
    model = AutoModelForCausalLM.from_pretrained(MODEL_NAME, trust_remote_code=True)
    model.eval()
    return model, processor

if __name__ == "__main__":
    print("Ładowanie modelu...")
    model, processor = load_model()
    print("Model załadowany.")
