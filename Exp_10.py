# Install required libraries
!pip install transformers torch pillow matplotlib sentencepiece -q

# Import libraries
import torch
import matplotlib.pyplot as plt
from PIL import Image
from google.colab import files
from transformers import BlipProcessor, BlipForQuestionAnswering

# Check device
device = "cuda" if torch.cuda.is_available() else "cpu"
print("Device:", device)

# Load BLIP VQA model
model_name = "Salesforce/blip-vqa-base"

processor = BlipProcessor.from_pretrained(model_name)
model = BlipForQuestionAnswering.from_pretrained(model_name).to(device)

# Upload image
print("\nUpload an image for analysis:")
uploaded_files = files.upload()

# Read uploaded image
image_path = next(iter(uploaded_files))
image = Image.open(image_path).convert("RGB")

# Ask user question
question = input("\nEnter a question about the uploaded image: ")

# Process inputs
inputs = processor(
    images=image,
    text=question,
    return_tensors="pt"
)

# Move tensors to GPU/CPU
inputs = {key: value.to(device) for key, value in inputs.items()}

# Generate answer
with torch.no_grad():
    generated_ids = model.generate(
        **inputs,
        max_new_tokens=30
    )

# Decode answer
answer = processor.decode(
    generated_ids[0],
    skip_special_tokens=True
)

# Display image
plt.figure(figsize=(8, 6))
plt.imshow(image)
plt.axis("off")
plt.title("Input Image")
plt.show()

# Print result
print("\nMULTIMODAL AI RESULT")
print("-" * 50)
print("Question:", question)
print("Answer:", answer)
