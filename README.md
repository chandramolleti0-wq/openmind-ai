# OpenMind AI 

**An AI-powered knowledge assistant with a modern chat interface, local language model support, and web-connected knowledge retrieval.**

OpenMind AI is a Python-based AI assistant designed to help users ask questions, retrieve relevant information, interact with saved knowledge, and explore AI capabilities through a web interface.

## Features

- **Interactive Chat Interface** — Ask questions through a web-based chat UI.
- **Local AI Model Support** — Run a compatible local language model on supported hardware.
- **Knowledge Retrieval** — Retrieve relevant information from saved knowledge.
- **Web-Connected Knowledge** — Add website content and retrieve information from supported public web sources.
- **Document Uploads** — Support for PDF, TXT, and Markdown documents, subject to configured limits.
- **API-Key Authentication** — Protect supported API endpoints with authentication.
- **Chat History** — View and manage previous conversations when chat-history storage is enabled.
- **FastAPI Backend** — Python backend with interactive API documentation.
- **Modern UI** — Clean interface designed for a smooth user experience.

## Technology Stack

- Python
- FastAPI
- Uvicorn
- HTML, CSS, and JavaScript
- ONNX Runtime GenAI
- Phi-3 Mini model (local inference configuration)
- Git and GitHub

## Requirements

- Python 3.11 or 3.12
- Git
- Windows, Linux, or another compatible Python environment
- Compatible model files for local inference
- NVIDIA GPU and compatible CUDA dependencies if using the GPU configuration

**Note:** Local GPU inference depends on compatible hardware, drivers, CUDA libraries, and installed Python packages.

## Run Locally

### 1. Clone the repository

```bash
git clone https://github.com/chandramolleti0-wq/openmind-ai.git
cd openmind-ai
```

### 2. Create a virtual environment

**Windows PowerShell:**

```powershell
py -3.12 -m venv .venv
.\.venv\Scripts\Activate.ps1
```

### 3. Install dependencies

Use the dependency file appropriate for your environment.

For the standard setup:

```powershell
python -m pip install --upgrade pip
pip install -r requirements.txt
```

For a compatible local CUDA setup, check `requirements-local-cuda.txt` and install the required GPU dependencies.

### 4. Configure environment variables

Create a local `.env` file based on `.env.example`, if your application uses these settings.

Never commit real API keys, passwords, or other secrets to GitHub.

### 5. Start the application

```powershell
python -m uvicorn app:app --host 127.0.0.1 --port 8001
```

Open the application at:

http://127.0.0.1:8001

If your configuration uses another port, open the URL for that port.

## API Authentication

The application may require an API key for protected endpoints.

- Keep your API key private.
- Store secrets in environment variables or an appropriate server-side configuration.
- Do not place private API keys in public frontend JavaScript.
- Do not upload `.env` files or private key files to GitHub.

Refer to the application's configuration and API documentation for the authentication method currently enabled.

## API Documentation

When the server is running, visit:

- **Swagger UI:** `http://127.0.0.1:8001/docs`
- **Health endpoint:** `http://127.0.0.1:8001/health` (if enabled)

Available endpoints depend on the current application implementation.

## Deployment

OpenMind AI can be prepared for deployment on a compatible hosting platform.

Before deploying:

1. Check the configuration in `render.yaml`.
2. Configure required environment variables in the hosting dashboard.
3. Verify that the Python dependency files match the selected runtime.
4. Configure persistent storage if chat history or knowledge data must survive restarts.
5. Confirm that the selected hosting plan supports the application's model-inference requirements.

**GPU deployment note:** The local Phi-3 CUDA configuration will not automatically work on a standard CPU-only hosting instance. GPU inference may require a separate GPU-enabled environment or a compatible hosted inference service.

**Storage note:** Temporary hosting filesystems may lose locally stored chat history and knowledge data after restarts or redeployments. Use persistent storage or a suitable external database when required.

## Security

- Never commit API keys, tokens, passwords, or private configuration files.
- Keep `.env`, `.venv/`, `data/`, and `models/` out of version control when they contain local data, secrets, or large model files.
- Use HTTPS for public deployments.
- Apply authentication, rate limits, and appropriate access controls before exposing the application publicly.
- Follow the licenses and usage terms of third-party models and dependencies.

## Roadmap

- Improve chat-history management.
- Enhance conversational response quality.
- Improve document ingestion and retrieval.
- Add persistent storage for conversations.
- Improve deployment compatibility.
- Continue refining the user interface and accessibility.

## Contributing

Contributions, bug reports, and suggestions are welcome.

1. Fork the repository.
2. Create a feature branch.
3. Make your changes.
4. Test the changes locally.
5. Submit a pull request.

## License

This project is licensed under the MIT License. See the `LICENSE` file for details.

Third-party models, datasets, and dependencies may have separate licenses.

## Author

**Chandra Sekhar Molleti**

GitHub: https://github.com/chandramolleti0-wq

---

*OpenMind AI — Building a smarter way to explore knowledge with AI.*
