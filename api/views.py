from django.shortcuts import render
from rest_framework.decorators import api_view
from rest_framework.response import Response
from django.core.files.storage import default_storage
from django.core.files.base import ContentFile
import os
import requests
from pix2text import Pix2Text
import re

OLLAMA_URL = "http://localhost:11434/api/generate"  # Ollama API URL
model= "t1c/deepseek-math-7b-rl"

CORS_ALLOW_ALL_ORIGINS = True
CORS_ALLOW_CREDENTIALS = True

@api_view(['POST'])
def upload_image(request):
    if 'image' not in request.FILES:
        print("❌ No image uploaded!")
        return Response({'error': 'No image uploaded'}, status=400)

    image = request.FILES['image']
    file_path = default_storage.save('uploads/' + image.name, ContentFile(image.read()))
    full_file_path = os.path.join('media', file_path)

    print(f"📌 Image saved at: {full_file_path}")  # Debugging

    try:
        # Load Pix2Text model
        print("🚀 Loading Pix2Text model...")
        p2t = Pix2Text.from_config()
        print("✅ Pix2Text model loaded successfully!")

        # Recognize the equation from the image
        print("🔍 Running Pix2Text recognition...")
        extracted_equation = p2t.recognize(full_file_path, file_type='text_formula')
        print(f"✅ Extracted Equation: {extracted_equation}")

        if not extracted_equation:
            print("⚠️ WARNING: No equation extracted from image!")
            return Response({'error': 'No equation extracted from image'}, status=400)

        # Send the extracted equation to Ollama for solving
        print("📡 Sending equation to Ollama...")
        ollama_response = requests.post(OLLAMA_URL, json={
            "model": "t1c/deepseek-math-7b-rl",
            "prompt": f"Solve & Equations should be in latex format: {extracted_equation}",
            "stream": False
        })

        if ollama_response.status_code == 200:
            ollama_output = ollama_response.json().get("response", "No response from Ollama")
            print(f"✅ Ollama Output: {ollama_output}")
        else:
            ollama_output = "Error: Unable to process with Ollama"
            print(f"❌ Ollama Error: {ollama_response.text}")

        def format_latex_response(response_text):
            parts = re.split(r"(\$\$.*?\$\$|\$.*?\$)", response_text)
            formatted_lines = [
                part.strip().replace("$", "").replace("$$", "") if part.startswith("$$") else part.strip()
                for part in parts if part.strip()
            ]
            return formatted_lines


        return Response({
            'equation': extracted_equation,
            'solution': format_latex_response(ollama_output)
        })

    except Exception as e:
        print(f"❌ Error during processing: {str(e)}")
        return Response({'error': f'Failed to process image: {str(e)}'}, status=500)
