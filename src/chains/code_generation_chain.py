from langchain_core.prompts import PromptTemplate
from langchain_core.output_parsers import JsonOutputParser

from llm_provider import SupportedLLMs, get_llm



class CodeGenerationChain:
    CODE_GENERATION_PROMPT_TEMPLATE = """Given the following software development task:
{task_description}

and the following technology stack:
{tech_stack}

For each technology in the stack, generate a single code block that showcases its primary role in the task. Each block should:

- Include necessary framework initialization, configuration, entry point, and main/bootstrap code (e.g., `Program.cs`/`Startup.cs` in ASP.NET, `main.js`/`App.vue` in Vue.js, etc.).
- Be complete and self-contained, representing the main application/unit for that technology—not just a single endpoint or function.
- Integrate logically with other stack components (e.g., consistent API URLs, shared models, aligned backend/database code).
- Be suitable as the main starting point for the respective technology.

Output a JSON object in the following format:
{{ "code_blocks": [ {{"technology": "$TECH_STACK_ITEM_1", "code": "$COMPLETE_MAIN_CODE_SNIPPET_1"}}, {{"technology": "$TECH_STACK_ITEM_2", "code": "$COMPLETE_MAIN_CODE_SNIPPET_2"}}, ...] }}

Return only valid one-line JSON, with no explanations or comments. Escape only the characters required by the JSON format (DON'T escape the dollar sign like \$; use $ as is).
"""

    def create(self, llm: SupportedLLMs):
        model = get_llm(llm)
        return (
            PromptTemplate.from_template(self.CODE_GENERATION_PROMPT_TEMPLATE)
            | model
            | JsonOutputParser()
        )