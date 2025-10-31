# NeuroRAG - Mental Health AI Assistant

[![Status](https://img.shields.io/badge/status-working-brightgreen)](https://github.com/GauravPatil2515/NEURO-RAG)
[![Python](https://img.shields.io/badge/python-3.9+-blue)](https://www.python.org/)
[![License](https://img.shields.io/badge/license-MIT-green)](LICENSE)

AI-powered Mental Health Question Answering System using RAG (Retrieval-Augmented Generation).

Built by **Gaurav Patil** | B.Tech Computer Engineering | India

## Features

- **Natural Language Search** - Ask questions in plain English about ICD-10 mental health codes
- **Fast Response** - Instant answers using semantic search
- **Beautiful Dashboard** - Modern, responsive web interface
- **100% Private** - Runs locally, no data sent to external APIs
- **AI-Powered** - Uses advanced RAG pipeline with FAISS
- **Real-time Stats** - Monitor system health and performance

## Quick Start

### Super Quick (3 Steps)

1. Navigate to project folder
2. Double-click START_SERVER.bat
3. Open browser to <http://127.0.0.1:5000>

Done! Start asking questions!

## Prerequisites

- Python 3.10 or higher
- 8GB RAM recommended
- Windows/Linux/Mac

## Installation

### Step 1: Clone the Repository

```bash
git clone https://github.com/GauravPatil2515/NEURO-RAG.git
cd NEURO-RAG
```

### Step 2: Install Dependencies

```bash
pip install -r requirements.txt
```

### Step 3: Start the Server

**Windows:**

```bash
START_SERVER.bat
```

**Linux/Mac:**

```bash
python run_server.py
```

### Step 4: Open Dashboard

Navigate to <http://127.0.0.1:5000>

## Usage Examples

- "What is the code for Recurrent depressive disorder?"
- "What are the diagnostic criteria for OCD?"
- "Tell me about bipolar disorder"
- "Explain schizophrenia classification"

## Tech Stack

| Component | Technology |
|-----------|------------|
| Frontend | HTML5, CSS3, JavaScript |
| Backend | Flask (Python) |
| RAG Framework | LangChain |
| Vector DB | FAISS |
| Embeddings | all-MiniLM-L6-v2 |
| LLM | Falcon-RW-1B |
| Data | ICD-10 Chapter V |

## Testing

Run the comprehensive test suite:

```bash
python tests/test_complete.py
```

## API Endpoints

- **GET /** - Dashboard homepage
- **GET /health** - Health check endpoint
- **POST /api/search** - Search for mental health information
- **GET /api/stats** - Get system statistics

## Performance

- Search Speed: Less than 2 seconds
- Memory Usage: Approximately 2GB RAM
- Accuracy: High (RAG-based retrieval)
- Offline: 100% local execution

## Author

Gaurav Patil

B.Tech Computer Engineering, India

- GitHub: [GauravPatil2515](https://github.com/GauravPatil2515)
- Project: [NEURO-RAG](https://github.com/GauravPatil2515/NEURO-RAG)

## License

This project is open-source under the MIT License.

Built with AI and Open Source Technology
