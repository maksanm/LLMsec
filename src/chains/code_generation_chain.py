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

For each technology listed in the technology stack, generate a single relevant code block that represents its core contribution to solving the given task. For example, if the stack is "Vue.js, ASP.NET WebApi, SQL Server", there should be one code block for Vue.js (e.g., a frontend component or page), one for ASP.NET WebApi (e.g., a key controller or endpoint), and one for SQL Server (e.g., a table definition or sample query).

Output a JSON object in the following format:
```json
{{
    "code_blocks": [
        {{"technology": "<TECH_STACK_ITEM_1>", "code": "<VALID_CODE_SNIPPET_FOR_THIS_TECHNOLOGY>"}},
        {{"technology": "<TECH_STACK_ITEM_2>", "code": "<VALID_CODE_SNIPPET_FOR_THIS_TECHNOLOGY>"}},
        ...
    ]
}}
```

Return only valid JSON, with no extra explanation.
"""

    def create(self, llm: SupportedLLMs):
        if llm == SupportedLLMs.OPENAI_41:
            llm = ChatOpenAI(model_name="gpt-4.1", temperature=0.5)
        elif llm == SupportedLLMs.DEEPSEEK_V3:
            llm = ChatDeepSeek(model_name="deepseek-chat", temperature=0.5)
        return (
            PromptTemplate.from_template(self.CODE_GENERATION_PROMPT_TEMPLATE)
            | llm
            | JsonOutputParser()
        )