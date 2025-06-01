from langchain_core.prompts import PromptTemplate
from langchain_core.output_parsers import JsonOutputParser

from llm_provider import SupportedLLMs, get_llm, get_lowcost_llm


class CodeGenerationChain:
    CODE_GENERATION_PROMPT_TEMPLATE = """Given the following software development task:
{task_description}

and the following technology stack:
{tech_stack}

For each technology in the stack, generate as many code blocks as needed to represent the key files required for its role in the task. For each code block:

- Each code block should correspond to a distinct, essential file required for that technology’s role in the solution (e.g., application files, dependecy files, configs, modules, etc.).
- Together, the code blocks for each technology must cover the main logic and structure needed for integration into the overall system.
- Ensure code blocks are logically consistent and interconnected across technologies where relevant (e.g., shared APIs, data models, configuration).
- Preserve any necessary project structure and organization within the code itself, but provide only the filename (not a relative or absolute path) in the filename field.

Output a JSON object in the following format (note the grouping by technology):
`{{"code_blocks": [ {{"technology": "$TECH_STACK_ITEM_1", "blocks": [{{"filename": "$FILENAME_NOT_A_PATH_1", "code": "$CODE_FOR_FILE_1"}}, {{"filename": "$FILENAME_NOT_A_PATH_2", "code": "$CODE_FOR_FILE_2"}}]}}, {{"technology": "$TECH_STACK_ITEM_2", "blocks": [{{"filename": "$FILENAME_NOT_A_PATH_3", "code": "$CODE_FOR_FILE_3"}}]}}, ... ] }}`

Return only valid, single-line JSON with no explanations or comments. Escape only characters required by JSON formatting. Do NOT escape the dollar sign; use `$` as is."""

    def create(self, llm: SupportedLLMs):
        model = get_llm(llm)
        return (
            PromptTemplate.from_template(self.CODE_GENERATION_PROMPT_TEMPLATE)
            | model
            | JsonOutputParser()
        )