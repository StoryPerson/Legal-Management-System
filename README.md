# AI-Powered Legal Case Management & Precedent Search

A comprehensive AI-powered system for legal case management, featuring case classification, prioritization, and intelligent precedent search using RAG (Retrieval-Augmented Generation).

## Features

- **Case Classification**: Automatically classify court cases by category (Civil, Criminal, or Constitutional)
- **Case Prioritization**: Predict the urgency level of cases (High, Medium, Low)
- **Legal Precedent Search (RAG)**: Retrieve related case precedents using embeddings and similarity search

## Prerequisites

- Python 3.9 or higher
- pip package manager
- Groq API key (for RAG functionality)

## Installation

1. **Clone the repository**
   ```bash
   git clone <your-repository-url>
   cd "Legal Management Deployment"
   ```

2. **Create a virtual environment** (recommended)
   ```bash
   python -m venv venv
   source venv/bin/activate  # On Windows: venv\Scripts\activate
   ```

3. **Install dependencies**
   ```bash
   pip install -r requirements.txt
   ```

4. **Download NLTK data**
   ```bash
   python -c "import nltk; nltk.download('stopwords'); nltk.download('wordnet'); nltk.download('omw-1.4')"
   ```

5. **Configure environment variables**
   - Copy `.env.example` to `.env`
   - Add your Groq API key to `.env`:
     ```
     api_key=your_actual_groq_api_key
     ```

## Running the Application

```bash
streamlit run app.py
```

The application will open in your default browser at `http://localhost:8501`

## Project Structure

```
.
├── app.py                          # Main Streamlit application
├── requirements.txt                # Python dependencies
├── .env                           # Environment variables (not in git)
├── .env.example                   # Environment template
├── Case Categorization/           # Classification models
│   ├── voting_pipeline.pkl
│   └── label_encoder.pkl
├── Case Prioritization/           # Prioritization models
│   ├── stacking_pipeline.pkl
│   └── label_encoder.pkl
└── Legal_Precedent_Search/        # RAG components
    ├── chroma_db/                 # Vector database
    ├── embeddings_config.pkl
    ├── llm_config.pkl
    └── prompt_template.pkl
```

## Usage

1. **Select a tool** from the sidebar:
   - Home
   - Case Classification
   - Case Prioritization
   - Legal Precedent Search (RAG)

2. **Enter case text** or legal questions in the provided text area

3. **Click the action button** to get predictions or search results

## Deployment

### Streamlit Cloud

1. Push your code to GitHub
2. Go to [share.streamlit.io](https://share.streamlit.io)
3. Connect your GitHub repository
4. Add your `api_key` in Streamlit Cloud secrets
5. Deploy!

### Docker (Optional)

```dockerfile
FROM python:3.9-slim

WORKDIR /app
COPY requirements.txt .
RUN pip install -r requirements.txt
COPY . .

EXPOSE 8501
CMD ["streamlit", "run", "app.py"]
```

## Important Notes

- **Model Files**: The `.pkl` files contain trained models and are required for the application to work
- **API Key**: Keep your `.env` file secure and never commit it to version control
- **Large Files**: If model files exceed GitHub's size limits, consider using Git LFS or hosting them separately

## Troubleshooting

### NumPy Version Conflict
If you encounter issues with `langchain-chroma`, ensure numpy version is < 2.0:
```bash
pip install "numpy<2.0.0,>=1.26.0"
```

### Missing NLTK Data
If you see NLTK errors, download the required data:
```bash
python -c "import nltk; nltk.download('stopwords'); nltk.download('wordnet'); nltk.download('omw-1.4')"
```

## License

[Add your license here]

## Contributors

[Add contributors here]
# Legal-Management-Deployment
