import streamlit as st
from PIL import Image
import numpy as np
import io
import torch
from diffusers import DiffusionPipeline
from diffusers import StableDiffusionPipeline

# Session state: remember previous state
opts = ['prompt', 'model', 'num_imgs', 'quality']

if 'imgs' not in st.session_state:
    st.session_state['imgs'] = []

if 'model1' not in st.session_state:
    st.session_state['model1'] = DiffusionPipeline.from_pretrained('amused/amused-256', variant="fp16", torch_dtype=torch.float16)

if 'model2' not in st.session_state:
    st.session_state['model2'] = DiffusionPipeline.from_pretrained('amused/amused-512', variant="fp16", torch_dtype=torch.float16)

if 'model3' not in st.session_state:
    st.session_state['model3'] = StableDiffusionPipeline.from_pretrained("runwayml/stable-diffusion-v1-5", variant="fp16", torch_dtype=torch.float16)


# Function to generate images based on user input
def generate_images(text, num_images, quality):
    # Placeholder function, replace with your actual implementation
    if torch.cuda.is_available():

        if selected_model == 'amused/amused-256':
            print("Model selected: 256")
            pipe = st.session_state['model1']
        elif selected_model == 'amused/amused-512':
            print("Model selected: 512")
            pipe = st.session_state['model2']
        else: 
            print("Model selected: diffusion")
            pipe = st.session_state['model3']

        pipe = pipe.to('cuda')

        with st.spinner('Generating...'):
            images = pipe(
                text, 
                num_images_per_prompt=num_images, 
                num_inference_steps=quality,
                generator=torch.Generator('cuda').manual_seed(8)
                ).images

            return images
    else:
        return ['example_img_1.jpg', 
            'example_img_2.jpg', 
            'example_img_3.jpg', 
            'example_img_4.jpg']

# Function to convert PIL image to bytes
def pil_to_bytes(image):
    img_byte_array = io.BytesIO()
    image.save(img_byte_array, format='PNG')
    return img_byte_array.getvalue()

st.set_page_config(layout="wide")
st.title('Text-to-Image AI-based Chatbot')

col1, col2, col3 = st.columns([1.1, 2, 2])

# Text input box
user_input = col1.text_input("Enter your text here:")


# Select models
model_opts = ["amused/amused-256", "amused/amused-512", "runwayml/stable-diffusion-v1-5"]
selected_model = col1.selectbox("Select model:", model_opts)

# Number of images input
num_images_opts = [1, 2, 4]
num_images = col1.selectbox("Number of images to generate:", num_images_opts)

# Image quality input
image_quality = col1.slider("Select image quality:", min_value=1, max_value=100, value=50, step=1)

# List to store all generated images
storage_images = []


# Generate button
if col1.button("Generate Images"):
    if user_input:
        # Placeholder for generated images of current generation
        generated_images = generate_images(user_input, num_images, image_quality)

        st.session_state['imgs'].extend(generated_images)
        
        # Display the generated images in col2
        if len(generated_images) == 1:
            # Display a single image in the center of col2
            col2.image(generated_images[0], caption=f'Generated Image 1', use_column_width=True)
            
            # Offer download button for the single image
            img_bytes = pil_to_bytes(generated_images[0])
            col2.download_button(label=f"Download Image 1", data=img_bytes, file_name=f"generated_image_1.jpg", mime='image/jpg')
        else:
            # Display the generated images in pairs
            for idx in range(0, len(generated_images), 2):
                # Create a new row to display two images side by side
                row = col2.columns(2)
                
                # Display the first image in the pair
                row[0].image(generated_images[idx], caption=f'Generated Image {idx+1}', use_column_width=True)
                
                # Check if there's a second image to display
                if idx + 1 < len(generated_images):
                    # Display the second image in the pair
                    row[1].image(generated_images[idx + 1], caption=f'Generated Image {idx+2}', use_column_width=True)
                    
                    # Offer download buttons for both images in the pair
                    img_bytes_1 = pil_to_bytes(generated_images[idx])
                    img_bytes_2 = pil_to_bytes(generated_images[idx + 1])
                    
                    row[0].download_button(label=f"Download Image {idx+1}", data=img_bytes_1, file_name=f"generated_image_{idx+1}.jpg", mime='image/jpg')
                    row[1].download_button(label=f"Download Image {idx+2}", data=img_bytes_2, file_name=f"generated_image_{idx+2}.jpg", mime='image/jpg')
                else:
                    # Display placeholder if there's no second image
                    row[1].info("No second image to display.")
    else:
        col2.warning("Please enter some text.")

# Display the number of images in storage_images in col3
col3.warning(f"Number of Images in Storage: {len(storage_images)}")


for idx in range(0, len(st.session_state['imgs']), 2):
    # Create a new row to display two images side by side
    row = col3.columns(2)

    img = st.session_state['imgs'][idx]
    
    # Display the first image in the pair
    row[0].image(img, caption=f'Generated Image {idx+1}', use_column_width=True)
    
    # Check if there's a second image to display
    if idx + 1 < len(st.session_state['imgs']):
        # Display the second image in the pair
        row[1].image(img, caption=f'Generated Image {idx+2}', use_column_width=True)
        
    else:
        # Display placeholder if there's no second image
        row[1].info("No second image to display.")
