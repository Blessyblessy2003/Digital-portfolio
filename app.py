import streamlit as st
import torch
import numpy as np
import easyocr
from PIL import Image
from pdf2image import convert_from_bytes   # ✅ THIS LINE
from transformers import TrOCRProcessor, VisionEncoderDecoderModel

# -------------------------------
# Load OCR models (cached)
# -------------------------------
@st.cache_resource
def load_models():
    easy_reader = easyocr.Reader(['en'], gpu=torch.cuda.is_available())
    processor = TrOCRProcessor.from_pretrained("microsoft/trocr-base-handwritten")
    model = VisionEncoderDecoderModel.from_pretrained("microsoft/trocr-base-handwritten")
    return easy_reader, processor, model

easy_reader, processor, trocr_model = load_models()

# -------------------------------
# OCR functions
# -------------------------------
def trocr_ocr(image):
    pixel_values = processor(images=image, return_tensors="pt").pixel_values
    generated_ids = trocr_model.generate(pixel_values)
    return processor.batch_decode(generated_ids, skip_special_tokens=True)[0]

def easyocr_ocr(image):
    image_np = np.array(image)   # PIL → NumPy
    results = easy_reader.readtext(image_np)
    return " ".join([res[1] for res in results])

def process_image(image):
    handwritten = trocr_ocr(image)
    typed = easyocr_ocr(image)
    return f"📝 Handwritten OCR:\n{handwritten}\n\n🖨 Typed OCR:\n{typed}"

# -------------------------------
# Streamlit UI
# -------------------------------
st.set_page_config(page_title="Handwritten OCR App", layout="centered")

st.title("📄 OCR: Image / PDF → Digital Text")
st.write("Upload a handwritten or typed answer sheet")

uploaded_file = st.file_uploader(
    "Upload Image or PDF",
    type=["png", "jpg", "jpeg", "pdf"]
)

if uploaded_file is not None:
    st.success("File uploaded successfully")

    extracted_text = ""

    if uploaded_file.type == "application/pdf":
        images = convert_from_bytes(uploaded_file.read())
        for i, page in enumerate(images):
            st.image(page, caption=f"Page {i+1}", use_container_width=True)
            extracted_text += f"\n--- Page {i+1} ---\n"
            extracted_text += process_image(page)

    else:
        image = Image.open(uploaded_file).convert("RGB")
        st.image(image, caption="Uploaded Image", use_container_width=True)
        extracted_text = process_image(image)

    st.subheader("📜 Extracted Digital Text")
    st.text_area("OCR Output", extracted_text, height=300)
