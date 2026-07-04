from django.shortcuts import render
from rest_framework.decorators import api_view
from rest_framework.response import Response
from django.core.files.storage import default_storage
from django.core.files.base import ContentFile
import os
import requests
from pix2text import Pix2Text
import re

OLLAMA_URL = os.getenv("OLLAMA_URL", "http://localhost:11434/api/generate")
OLLAMA_MODEL = os.getenv("OLLAMA_MODEL", "t1c/deepseek-math-7b-rl")
OLLAMA_TIMEOUT = int(os.getenv("OLLAMA_TIMEOUT", "120"))

print("Loading Pix2Text model...")
P2T = Pix2Text.from_config()
print("Pix2Text model ready.")

@api_view(['POST'])
def upload_image(request):
    text_input = request.data.get('text', None)
    image = request.FILES.get('image', None)
    extracted_equation = None

    if image:
        file_path = default_storage.save('uploads/' + image.name, ContentFile(image.read()))
        full_file_path = os.path.join('media', file_path)
        extracted_equation = P2T.recognize(full_file_path, file_type='text_formula')

        if not extracted_equation:
            return Response({'error': 'No equation extracted from image'}, status=400)

    elif text_input:
        extracted_equation = text_input.strip()

    else:
        return Response({'error': 'No valid input provided (image or text required).'}, status=400)

    try:
        ollama_response = requests.post(
            OLLAMA_URL,
            json={
                "model": OLLAMA_MODEL,
                "prompt": f"Solve it and Only give short explanation: {extracted_equation}",
                "stream": False
            },
            timeout=OLLAMA_TIMEOUT,
        )

        if ollama_response.status_code != 200:
            return Response(
                {'error': 'Unable to process with Ollama'},
                status=502
            )

        ollama_output = ollama_response.json().get("response", "No response from Ollama")

        def format_latex_response(response_text):
            parts = re.split(r"(\$\$.*?\$\$|\$.*?\$)", response_text)
            formatted_lines = []

            for part in parts:
                part = part.strip()
                if not part:
                    continue
                
                if part.startswith("$$") and part.endswith("$$"):
                    formatted_lines.append(part.replace("$$", "").strip())
                elif part.startswith("$") and part.endswith("$"):
                    formatted_lines.append(part.replace("$", "").strip())
                else:
                    escaped = part.replace("\\", "\\\\").replace("{", "\\{").replace("}", "\\}")
                    formatted_lines.append(f"\\text{{{escaped}}}")

            return formatted_lines

        return Response({
            'equation': extracted_equation,
            'solution': format_latex_response(ollama_output)
        })

    except requests.RequestException:
        return Response({'error': 'Ollama request timed out or failed'}, status=502)
    except Exception as e:
        return Response({'error': f'Failed to process input: {str(e)}'}, status=500)
