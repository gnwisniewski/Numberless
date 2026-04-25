import glob, os, re, sys, cv2, torch
import numpy as np
from PIL import Image
from tqdm import tqdm
from transformers import AutoProcessor, AutoModelForCausalLM

MODEL_NAME = "microsoft/Florence-2-base"
TASK_PROMPT = "<OCR_WITH_REGION>"
NUMBER_PATTERN = re.compile(r"\d{2,}")
EMAIL_PATTERN = re.compile(r"[\w.+-]+@[\w.-]+\.\w+")
URL_PATTERN = re.compile(r"(https?://|www\.)?\S+\.(com|pl|org|net|io|eu|de|fr|uk|co|info|biz|gov|edu)(\S*)?", re.IGNORECASE)
OUTPUT_DIR = "output"

def load_model():
    processor = AutoProcessor.from_pretrained(MODEL_NAME, trust_remote_code=True)
    model = AutoModelForCausalLM.from_pretrained(MODEL_NAME, trust_remote_code=True, attn_implementation="eager")
    model.eval()
    return model, processor

def run_ocr(image: Image.Image, model, processor) -> dict:
    inputs = processor(text=TASK_PROMPT, images=image, return_tensors="pt")
    with torch.no_grad():
        output_ids = model.generate(
            input_ids=inputs["input_ids"],
            pixel_values=inputs["pixel_values"],
            max_new_tokens=1024,
            do_sample=False,
            use_cache=False,
        )
    generated_text = processor.batch_decode(output_ids, skip_special_tokens=False)[0]
    return processor.post_process_generation(
        generated_text,
        task=TASK_PROMPT,
        image_size=(image.width, image.height),
    )

def should_blur(label: str) -> bool:
    return bool(
        NUMBER_PATTERN.search(label)
        or EMAIL_PATTERN.search(label)
        or URL_PATTERN.search(label)
    )

def blur_numbers(image: Image.Image, ocr_result: dict) -> tuple[Image.Image, list[str]]:
    img_cv = cv2.cvtColor(np.array(image), cv2.COLOR_RGB2BGR)
    blurred_full = cv2.GaussianBlur(img_cv, (111, 111), 35)

    regions = ocr_result.get(TASK_PROMPT, {})
    labels = regions.get("labels", [])
    bboxes = regions.get("quad_boxes", [])
    redacted = []

    for label, bbox in zip(labels, bboxes):
        clean_label = label.replace("</s>", "").strip()
        if not should_blur(clean_label):
            continue

        pts = np.array(bbox, dtype=np.float32).reshape(4, 2).astype(np.int32)
        mask = np.zeros(img_cv.shape[:2], dtype=np.uint8)
        cv2.fillPoly(mask, [pts], 255)
        mask = cv2.dilate(mask, np.ones((25, 25), np.uint8), iterations=1)
        mask_blurred = cv2.GaussianBlur(mask, (31, 31), 15).astype(np.float32) / 255.0
        alpha = mask_blurred[:, :, np.newaxis]
        img_cv = (blurred_full * alpha + img_cv * (1 - alpha)).astype(np.uint8)
        redacted.append(clean_label)

    return Image.fromarray(cv2.cvtColor(img_cv, cv2.COLOR_BGR2RGB)), redacted

def process_image(image_path: str, model, processor):
    filename = os.path.splitext(os.path.basename(image_path))[0] + ".png"
    output_path = os.path.join(OUTPUT_DIR, filename)

    image = Image.open(image_path).convert("RGB")
    result = run_ocr(image, model, processor)
    output, redacted = blur_numbers(image, result)
    output.save(output_path)

    return output_path, redacted

if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage: python main.py <image_path> [image_path2 ...]")
        sys.exit(1)

    image_paths = []
    for pattern in sys.argv[1:]:
        matched = glob.glob(pattern)
        image_paths.extend(matched if matched else [pattern])

    os.makedirs(OUTPUT_DIR, exist_ok=True)
    print("Loading model...")
    model, processor = load_model()

    for image_path in tqdm(image_paths, desc="Processing images"):
        output_path, redacted = process_image(image_path, model, processor)
        print(f"\n{image_path} -> {output_path}")
        if redacted:
            for item in redacted:
                print(f"  redacted: {item}")
        else:
            print("  no sensitive data found")
