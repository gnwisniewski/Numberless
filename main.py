from transformers import AutoProcessor, AutoModelForCausalLM
from PIL import Image
import torch

MODEL_NAME = "microsoft/Florence-2-base"
TASK_PROMPT = "<OCR_WITH_REGION>"

def load_model():
    processor = AutoProcessor.from_pretrained(MODEL_NAME, trust_remote_code=True)
    model = AutoModelForCausalLM.from_pretrained(MODEL_NAME, trust_remote_code=True)
    model.eval()
    return model, processor

def run_ocr(image: Image.Image, model, processor) -> dict:
    inputs = processor(text=TASK_PROMPT, images=image, return_tensors="pt")
    with torch.no_grad():
        output_ids = model.generate(
            input_ids=inputs["input_ids"],
            pixel_values=inputs["pixel_values"],
            max_new_tokens=1024,
            num_beams=3,
        )

    generated_text = processor.batch_decode(output_ids, skip_special_tokens=False)[0]
    result = processor.post_process_generation(
        generated_text,
        task=TASK_PROMPT,
        image_size=(image.width, image.height),
    )
    return result

if __name__ == "__main__":
    import sys
    if len(sys.argv) < 2:
        print("Usage: python main.py <image_path>")
        sys.exit(1)

    image_path = sys.argv[1]
    image = Image.open(image_path).convert("RGB")
    print("Loading model...") 