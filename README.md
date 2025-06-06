# LLMsec

**LLMsec** evaluates and compares the code generation safety of leading Large Language Models (LLMs) such as OpenAI GPT-4.1, DeepSeek V3, Grock-3 and others. The project generates code solutions for programming tasks using these models and analyzes them for security vulnerabilities with [Trivy](https://github.com/aquasecurity/trivy). Detected vulnerabilities are classified using CVSS-based severity ratings to benchmark LLMs by the number and criticality of threats in their generated code.

## Project Overview

- **Model Coverage:** Supports testing with OpenAI GPT-4.1, DeepSeek V3, Grok-3, TBC.
- **Security Assessment:** Uses Trivy to scan generated code for vulnerabilities, classifying them by CVSS severity: Unknown, Low, Medium, High, Critical.
- **Experiment Scope:** Runs on a set of programming problems (e.g., web apps, auth flows, DevOps pipelines, etc.) described in natural language and implemented in various tech stacks. Analyzes and compares models based on the number and severity of vulnerabilities.

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
   - Set the `TRIVY_PATH` variable to the full path to your Trivy executable, which can be downloaded [here](https://github.com/aquasecurity/trivy/releases/tag/v0.63.0).
   - Provide required LLM platforms API keys.

### Running the Server

Start the FastAPI server with Uvicorn:

```bash
poetry run python src/server.py
```

The interactive API docs will be available at [http://localhost:8000](http://localhost:8000) (Swagger UI).

### Endpoint Parameters

- Use the `task_description` parameter to provide a natural language description of the desired software functionality and requirements.
- Use the `generation_mode` parameter to specify the analysis scope:
    - `"code"`: The system will analyze the actual code implementing the desired functionality.
    - `"dependencies"`: The system will analyze only the dependencies required for the project.

## Utilities

Scripts for data generation and analysis are located in the `utils` directory:

### `utils/generate_data.py`

- Sends multiple task prompts to the running API server and saves results as JSON in `generated_data/`.
- **How to run:**
  1. Start the API server (`src/server.py`).
  2. Optionally adjust prompts in the script.
  3. Run:
     ```bash
     poetry run python utils/generate_data.py
     ```

### `utils/analyze_data.py`

- Analyzes JSON results to generate vulnerability statistics and summary plots in `vulnerability_plots/`.
- **How to run:**
  ```bash
  poetry run python utils/analyze_data.py
  ```
- Requires files in `generated_data/` (run `generate_data.py` first).