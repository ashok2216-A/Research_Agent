# PDF Chatbot FastAPI

```

## Setup and Installation

1. Create a virtual environment:
```bash
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
```

2. Install dependencies:
```bash
pip install -r requirements.txt
```

3. Set up your environment variables:
- Create a `.env` file in the root directory
- Add your API keys and configurations

## Running the Application

To run the application:
```bash
uvicorn api:app --reload
```

## Features
- PDF document processing
- AI-powered chat interactions
- Document image extraction
- Research capabilities using Mistral AI 