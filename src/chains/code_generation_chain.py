from enum import Enum
from langchain_core.prompts import PromptTemplate
from langchain_core.output_parsers import JsonOutputParser
from langchain_deepseek import ChatDeepSeek
from langchain_openai import ChatOpenAI


class SupportedLLMs(str, Enum):
    OPENAI_41 = "gpt-4.1"
    DEEPSEEK_V3 = "deepseek-v3"

class CodeGenerationChain:
    CODE_GENERATION_PROMPT_TEMPLATE = """Given the following software development task:
```
{task_description}
```

and the following technology stack:
```
{tech_stack}
```

For each technology in the stack, generate a single code block that showcases its primary role in the task. Each block should:

- Include necessary framework initialization, configuration, entry point, and main/bootstrap code (e.g., `Program.cs`/`Startup.cs` in ASP.NET, `main.js`/`App.vue` in Vue.js, etc.).
- Be complete and self-contained, representing the main application/unit for that technology—not just a single endpoint or function.
- Integrate logically with other stack components (e.g., consistent API URLs, shared models, aligned backend/database code).
- Be suitable as the main starting point for the respective technology.

Output a JSON object in the following format:
```json
{{
    "code_blocks": [
        {{"technology": "<TECH_STACK_ITEM_1>", "code": "<COMPLETE_MAIN_CODE_SNIPPET_1>"}},
        {{"technology": "<TECH_STACK_ITEM_2>", "code": "<COMPLETE_MAIN_CODE_SNIPPET_2>"}},
        ...
    ]
}}
```

Return only valid JSON, with no explanations or comments.
"""

    def create(self, llm: SupportedLLMs):
        if llm == SupportedLLMs.OPENAI_41:
            llm = ChatOpenAI(model_name="gpt-4.1", temperature=0.0)
        elif llm == SupportedLLMs.DEEPSEEK_V3:
            llm = ChatDeepSeek(model_name="deepseek-chat", temperature=0.0)
        return (
            PromptTemplate.from_template(self.CODE_GENERATION_PROMPT_TEMPLATE)
            | llm
            | JsonOutputParser()
        )