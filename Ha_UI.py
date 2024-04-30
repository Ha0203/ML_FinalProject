import streamlit as st
from PIL import Image
import numpy as np
import io
import torch
from diffusers import DiffusionPipeline
from diffusers import StableDiffusionPipeline

st.set_page_config(layout="wide")

storage_images = []

def main():
    st.sidebar.title("Navigation")
    page = st.sidebar.radio("Go to", ["Text2Img", "Img2Img"])
    if page == "Text2Img":
        Text2Img()
    elif page == "Img2Img":
        Img2Img()


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

def Text2Img():

    st.title('Text-to-Image AI-based Chatbot')

    col1, col2 = st.columns([1, 1])
    
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
    
    # Generate button
    if col1.button("Generate Images"):
        if user_input:
            with st.spinner('Generating...'):
                # Placeholder for generated images of current generation
                generated_images = []
                
                # Generate the specified number of images
                for i in range(num_images):
                    example_image = Image.open(f"example_img_{i+1}.jpg")
                    generated_images.append(example_image)
                    storage_images.append(example_image)
                
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
            
def Img2Img():

    st.title('Image-to-Image AI-based Chatbot')

    col1, col2 = st.columns([1, 1])
    
    # File uploader for user input
    uploaded_file = col1.file_uploader("Upload an image file:", type=["jpg", "jpeg", "png"])
    
    if uploaded_file is not None:
        pil_image = Image.open(uploaded_file)
        max_size = (200, 200)
        width, height = pil_image.size

        resize_ratio = min(max_size[0] / width, max_size[1] / height)

        new_width = int(width * resize_ratio)
        new_height = int(height * resize_ratio)

        preview_image = pil_image.resize((new_width, new_height))
        col1.image(preview_image, caption="Uploaded Image Preview")

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
    
    # Generate button
    if col1.button("Generate Images"):
        if user_input:
            with st.spinner('Generating...'):
                # Placeholder for generated images of current generation
                generated_images = []
                
                # Generate the specified number of images
                for i in range(num_images):
                    example_image = Image.open(f"example_img_{i+1}.jpg")
                    generated_images.append(example_image)
                    storage_images.append(example_image)
                
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

if __name__ == "__main__":
    main()