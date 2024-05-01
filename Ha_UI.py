import streamlit as st
from PIL import Image
import numpy as np
import io
import torch
from diffusers import DiffusionPipeline
from diffusers import StableDiffusionPipeline, StableDiffusionInstructPix2PixPipeline, EulerAncestralDiscreteScheduler, StableDiffusionXLImg2ImgPipeline, StableDiffusionControlNetImg2ImgPipeline, ControlNetModel, DDIMScheduler

st.set_page_config(layout="wide")

if 'imgs' not in st.session_state:
    st.session_state['imgs'] = []

def main():
    st.sidebar.title("Navigation")
    page = st.sidebar.selectbox("Go to", ["Text2Img", "Img2Img"])

    st.title('Text-to-Image AI-based Chatbot')

    col1, col2, col3 = st.columns([2, 3, 2])
    
    if page == "Text2Img":
        Text2Img(col1, col2, col3)
    elif page == "Img2Img":
        Img2Img(col1, col2, col3)

    for idx in range(0, len(st.session_state['imgs']), 2):
        # Create a new row to display two images side by side
        row = col3.columns(2)

        img = st.session_state['imgs'][idx]
        
        # Display the first image in the pair
        row[0].image(img, caption=f'Generated Image {idx+1}', use_column_width=True)
        
        # Check if there's a second image to display
        if idx + 1 < len(st.session_state['imgs']):
            # Display the second image in the pair
            row[1].image(st.session_state['imgs'][idx + 1], caption=f'Generated Image {idx+2}', use_column_width=True)

# Function to generate images based on user input
def generate_images(text, model, num_images, quality):
    if model == 'amused/amused-256':
        print("Model selected: 256")
        # pipe = st.session_state['model1']
        pipe = DiffusionPipeline.from_pretrained('amused/amused-256', variant="fp16", torch_dtype=torch.float16)
    elif model == 'amused/amused-512':
        print("Model selected: 512")
        # pipe = st.session_state['model2']
        pipe = DiffusionPipeline.from_pretrained('amused/amused-512', variant="fp16", torch_dtype=torch.float16)
    else: 
        print("Model selected: diffusion")
        # pipe = st.session_state['model3']
        pipe = StableDiffusionPipeline.from_pretrained("runwayml/stable-diffusion-v1-5", variant="fp16", torch_dtype=torch.float16)

    pipe = pipe.to('cuda')

    
    images = pipe(
        text, 
        num_images_per_prompt=num_images, 
        num_inference_steps=quality,
        generator=torch.Generator('cuda').manual_seed(8)
        ).images

    return images

# Function to convert PIL image to bytes
def pil_to_bytes(image):
    img_byte_array = io.BytesIO()
    image.save(img_byte_array, format='PNG')
    return img_byte_array.getvalue()

def Text2Img(col1, col2, col3):
    
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
                # Placeholder for generated images of current generatio                generated_images = []   
                generated_images = generate_images(user_input, selected_model, num_images, image_quality)

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
            
def refineImage(text, model, num_imgs, quality, init_img):
    if model == 'timbrooks/instruct-pix2pix':
        print("timbrooks/instruct-pix2pix")
        # pipe = st.session_state['model4']
        pipe = StableDiffusionInstructPix2PixPipeline.from_pretrained("timbrooks/instruct-pix2pix", torch_dtype=torch.float16, safety_checker=None)
        pipe.scheduler = EulerAncestralDiscreteScheduler.from_config(pipe.scheduler.config)
       
    else: 
        print("stabilityai/stable-diffusion-xl-refiner-1.0")
        # pipe = st.session_state['model5']
        pipe = StableDiffusionXLImg2ImgPipeline.from_pretrained("stabilityai/stable-diffusion-xl-refiner-1.0", torch_dtype=torch.float16, variant="fp16", use_safetensors=True)

    pipe = pipe.to('cuda')

    images = pipe(
        text, 
        image=init_img,
        num_images_per_prompt=num_imgs, 
        num_inference_steps=quality,
        generator=torch.Generator('cuda').manual_seed(8)
        ).images

    return images
        
def Img2Img(col1, col2, col3):
    
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
    model_opts = ["timbrooks/instruct-pix2pix", "stabilityai/stable-diffusion-xl-refiner-1.0"]
    selected_model = col1.selectbox("Select model:", model_opts)

    # Number of images input
    num_images_opts = [1, 2, 4]
    num_images = col1.selectbox("Number of images to generate:", num_images_opts)

    # Image quality input
    image_quality = col1.slider("Select image quality:", min_value=1, max_value=100, value=50, step=1)
    
    # Generate button
    if col1.button("Generate Images"):
        if user_input and uploaded_file:
            with st.spinner('Generating...'):
                # Placeholder for generated images of current generation
                generated_images = refineImage(user_input, selected_model, num_images, image_quality, preview_image)
                
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
                
                st.session_state['imgs'].extend(generated_images)
        else:
            if not user_input:
                col2.warning("Please enter some text.")
            else:
                col2.warning("Please upload image")
        

if __name__ == "__main__":
    main()
