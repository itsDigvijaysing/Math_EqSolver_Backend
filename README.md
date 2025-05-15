# Math Equation Solver

A full-stack web application that allows users to **upload an image** or **input a mathematical equation** in text, and receive a **LaTeX-rendered solution with step-by-step explanation** using an LLM (Ollama - Deepseek-Math).

---

## Features

### Equation Input
- Upload handwritten or printed **math equation images**.
- Or type the math equation directly.

### OCR & Parsing
- Uses [`Pix2Text`](https://github.com/breezedeus/pix2text) to extract formulas from images.

### AI Solver (LLM via Ollama)
- Sends extracted/typed equations to **Deepseek-Math 7B RL** model via Ollama.
- Retrieves cleanly formatted LaTeX explanations.

### LaTeX Rendering
- Frontend uses **KaTeX** to render math beautifully.
- Both inline and block-level LaTeX support.

---

## Tech Stack

### Backend
- Django + Django REST Framework
- Pix2Text for image-to-formula OCR
- Ollama (running locally) for LLM-based reasoning
- Requests (HTTP call to Ollama API)

### Frontend
- React (with Vite)
- react-router-dom (routing)
- KaTeX (math rendering)
- Custom CSS particle animation
- axios/fetch for API calls

---

## Project Structure

```

Math\_EqSolver/
├── backend/
│   ├── api/
│   │   ├── views.py         # Core logic: OCR, LLM, formatting
│   │   └── urls.py, serializers.py, models.py
│   ├── backend/             # Django settings
│   ├── media/uploads/       # Uploaded equation images
│   ├── requirements.txt
│   └── manage.py
└── frontend/
└── my-app/
├── src/components/
│   ├── UploadImage.jsx        # Main UI for image/text input
│   ├── MathRenderer.jsx       # KaTeX renderer
│   ├── HomePage.jsx           # Landing page
│   ├── Signup.jsx / Login.jsx
│   ├── Navbar.jsx / Features.jsx / Testimonials.jsx
└── public/                    # Icons & assets

```

---

## Core Flow

1. **User uploads an image** or **types an equation**.
2. Backend:
   - OCR (if image) using Pix2Text.
   - Sends prompt to Ollama API: `"Solve it and only give short explanation: {equation}"`
   - Receives explanation.
   - Parses and formats LaTeX segments.
3. Frontend:
   - Receives solution array (LaTeX blocks).
   - Renders equation + formatted solution using `MathRenderer.jsx`.

---

## Backend Setup

### Requirements

- Python 3.8+
- Django
- Pix2Text
- Ollama installed locally with `deepseek-math-7b-rl` model pulled.

### Install Dependencies

```bash
cd backend
python -m venv venv
source venv/bin/activate
pip install -r requirements.txt
````

### Start Django Server

```bash
python manage.py migrate
python manage.py runserver
```

---

## Frontend Setup

### Install Dependencies

```bash
cd frontend/my-app
npm install
```

### Start React App

```bash
npm run dev
```

---

## API Endpoint

### `POST /upload_image`

Accepts:

* `image`: image file of handwritten or printed equation.
* `text`: optional fallback if no image is uploaded.

Returns:

```json
{
  "equation": "x^2 + 3x + 2 = 0",
  "solution": [
    "\\text{This is a quadratic equation}",
    "\\text{Using the quadratic formula:}",
    "x = \\frac{-b \\pm \\sqrt{b^2 - 4ac}}{2a}",
    "\\text{Substitute } a = 1, b = 3, c = 2",
    "x = \\frac{-3 \\pm \\sqrt{1}}{2}",
    "x = -1, -2"
  ]
}
```

## Required Updates

* Signup/Login UI exists, but backend auth APIs are not yet implemented.
* No persistent user data storage.
* Ollama must be running locally at `http://localhost:11434`.
* Pix2Text model is resource-heavy; cold starts may delay response.

---

## Dependencies

### Backend

* `Django`, `djangorestframework`
* `Pix2Text`
* `requests`, `re`, `os`, `json`

### Frontend

* `React`, `react-router-dom`
* `KaTeX`
* `axios`, `prop-types`

---

## Future Improvements

* Add backend auth endpoints for signup/login.
* Improve Pix2Text feedback for failed OCR.
* Deploy Ollama to remote server with GPU support.
* Add support for solving systems of equations and graphs.
* Store user history / favorites.
