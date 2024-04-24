import streamlit as st
from PIL import Image
import io
from openai import OpenAI

# Initialize OpenAI client
client = OpenAI(
    api_key="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"
)

response = client.images.generate(
    model="dall-e-2",
    prompt="Bình minh",
    size="1024x1024",  # Adjust size as needed
    quality="standard",      # Adjust quality as needed
    n=1              # Generate one image
)

image_url = response.data[0].url

print(image_url)
