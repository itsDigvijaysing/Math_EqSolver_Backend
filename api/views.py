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
model = "t1c/deepseek-math-7b-rl"

CORS_ALLOW_ALL_ORIGINS = True
CORS_ALLOW_CREDENTIALS = True

@api_view(['POST'])
def upload_image(request):
    text_input = request.data.get('text', None)
    image = request.FILES.get('image', None)
    extracted_equation = None  # Initialize variable

    if image:
        file_path = default_storage.save('uploads/' + image.name, ContentFile(image.read()))
        full_file_path = os.path.join('media', file_path)
        print(f"📌 Image saved at: {full_file_path}")  # Debugging

        # Process the image
        print("🚀 Loading Pix2Text model...")
        p2t = Pix2Text.from_config()
        extracted_equation = p2t.recognize(full_file_path, file_type='text_formula')
        print(f"✅ Extracted Equation: {extracted_equation}")

        if not extracted_equation:
            print("⚠️ WARNING: No equation extracted from image!")
            return Response({'error': 'No equation extracted from image'}, status=400)

    elif text_input:
        extracted_equation = text_input.strip()  # Use text input directly
        print(f"📜 Text Input Received: {extracted_equation}")

    else:
        return Response({'error': 'No valid input provided (image or text required).'}, status=400)

    try:
        print("📡 Sending query to Ollama...")
        ollama_response = requests.post(OLLAMA_URL, json={
            "model": "t1c/deepseek-math-7b-rl",
            "prompt": f"Solve it and Only give short explanation: {extracted_equation}",
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
            formatted_lines = []

            for part in parts:
                part = part.strip()
                if not part:
                    continue
                
                if part.startswith("$$") and part.endswith("$$"):
                    formatted_lines.append(part.replace("$$", "").strip())  # Block equation
                elif part.startswith("$") and part.endswith("$"):
                    formatted_lines.append(part.replace("$", "").strip())  # Inline equation
                else:
                    formatted_lines.append(f"\\text{{{part}}}")  # Normal text wrapped for spacing

            return formatted_lines

        return Response({
            'equation': extracted_equation,
            'solution': format_latex_response(ollama_output)
        })

    except Exception as e:
        print(f"❌ Error during processing: {str(e)}")
        return Response({'error': f'Failed to process input: {str(e)}'}, status=500)
