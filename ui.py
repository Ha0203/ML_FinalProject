import streamlit as st
from PIL import Image
import numpy as np
import io
import torch
from diffusers import DiffusionPipeline
from diffusers import StableDiffusionPipeline

# Function to generate images based on user input
def generate_images(text, size, num_images, quality):
    # Placeholder function, replace with your actual implementation
    generated_images = []
    for _ in range(num_images):
        # Use your model to generate an image based on the input text
        # Example:
        # image = model.generate_image(text, size, quality)
        # generated_images.append(image)
        pass
    return generated_images

# Function to convert PIL image to bytes
def pil_to_bytes(image):
    img_byte_array = io.BytesIO()
    image.save(img_byte_array, format='PNG')
    return img_byte_array.getvalue()

# Streamlit interface
st.title('Text-to-Image AI-based Chatbot')

# Text input box
user_input = st.text_input("Enter your text here:")

# Select models
model_opts = ["amused/amused-256", "amused/amused-512", "runwayml/stable-diffusion-v1-5"]
selected_model = st.selectbox("Select model:", model_opts)

if selected_model == 'amused/amused-256':
    print("Model selected: 256")
    pipe = DiffusionPipeline.from_pretrained('amused/amused-256', variant="fp16", torch_dtype=torch.float16)
elif selected_model == 'amused/amused-512':
    print("Model selected: 512")
    pipe = DiffusionPipeline.from_pretrained('amused/amused-512', variant="fp16", torch_dtype=torch.float16)
else: 
    print("Model selected: diffusion")
    pipe = StableDiffusionPipeline.from_pretrained("runwayml/stable-diffusion-v1-5", variant="fp16", torch_dtype=torch.float16)

# # Image size input
# image_size_options = ["1080x1080", "1280x720", "1920x1080"]
# selected_size = st.selectbox("Select image size:", image_size_options)

# Convert selected size to width and height
# width, height = map(int, selected_size.split('x'))

# Number of images input
num_images = st.number_input("Number of images to generate:", min_value=1, max_value=10, value=3, step=1)
# Image quality input
image_quality = st.slider("Select image quality:", min_value=1, max_value=100, value=50, step=1)

# Generate button
if st.button("Generate Images"):
    if user_input:
       
        # Example image
        # example_image = Image.open("example_img.jpg")
        # st.image(example_image, caption='Example Image', use_column_width=True)
        # img_bytes = pil_to_bytes(example_image)
        # st.download_button(label=f"Download Image ", data=img_bytes, file_name=f"generated_image_example.png", mime='image/png')
        pipe = pipe.to('cuda')

        with st.spinner('Generating...'):
            image = pipe(
                user_input, 
                num_images_per_prompt=num_images, 
                num_inference_steps=image_quality,
                generator=torch.Generator('cuda').manual_seed(8)
                ).images
        st.image(image  , use_column_width=True)
        # img_bytes = pil_to_bytes(image)
        # st.download_button(label=f"Download Image ", data=img_bytes, file_name=f"generated_image_example.png", mime='image/png')
        

        # # Generate images based on user input
        # generated_images = generate_images(user_input, (width, height), num_images, image_quality)
        # if generated_images:
        #     for i, img in enumerate(generated_images):
        #         st.image(img, caption=f"Generated Image {i+1}", use_column_width=True)
        #         # Download link for each image
        #         img_bytes = pil_to_bytes(img)
        #         st.download_button(label=f"Download Image {i+1}", data=img_bytes, file_name=f"generated_image_{i+1}.png", mime='image/png')
        # else:
        #     st.error("Failed to generate images.")
    else:
        st.warning("Please enter some text.")
