# LLMsec

**LLMsec** evaluates and compares the code generation safety of leading Large Language Models (LLMs) such as OpenAI GPT-4.1, DeepSeek V3, and others. The project generates code solutions for programming tasks using these models and analyzes them for security vulnerabilities with [Trivy](https://github.com/aquasecurity/trivy). Detected vulnerabilities are classified using CVSS-based severity ratings to benchmark LLMs by the number and criticality of threats in their generated code.

## Project Overview

- **Model Coverage:** Supports testing with OpenAI GPT-4.1, DeepSeek V3, TBC.
- **Security Assessment:** Uses Trivy to scan generated code for vulnerabilities, classifying them by CVSS severity: Unknown, Low, Medium, High, Critical.
- **Experiment Scope:** Runs on a set of programming problems (e.g., CRUD, web apps, auth flows) described in natural language and implemented in various tech stacks. Analyzes and compares models based on the number and severity of vulnerabilities, and identifies patterns such as supply chain attack risks.

## Getting Started

### Installation

1. **Clone the repository:**
   ```bash
   git clone https://github.com/yourusername/LLMsec.git
   cd LLMsec
   ```

2. **Install dependencies using Poetry:**
   ```bash
   poetry install
   ```

3. **Set up environment variables:**
   - Copy `.env.sample` to `.env` and fill in the required values.

### Running the Server

Start the FastAPI server with Uvicorn:

```bash
poetry run python src/server.py
```

The interactive API docs will be available at [http://localhost:8000](http://localhost:8000) (Swagger UI).
