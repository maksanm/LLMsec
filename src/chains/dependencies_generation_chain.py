from langchain_core.prompts import PromptTemplate
from langchain_core.output_parsers import JsonOutputParser

from llm_provider import SupportedLLMs, get_llm, get_lowcost_llm


class DependenciesGenerationChain:
    DEPENDENCIES_GENERATION_PROMPT_TEMPLATE = """Given the following software development task:
{task_description}

and the following technology stack:
{tech_stack}

For each technology, generate the primary dependency/requirement file (e.g., requirements.txt for Python, .csproj for .NET, package.json for Node.js) as a single code block.

**Instructions:**
- Create the correct dependency file for each technology as used at a project root.
- List all required packages/dependencies, EACH WITH AN EXPLICIT VERSION.
- Output only one code block per technology, with the complete content of the file, and proper, legitimate filename.
- Escape only characters required by JSON formatting (e.g. do NOT escape ', $).
- Return only valid JSON in this format:
`{{"code_blocks": [ {{"technology": "$TECH_STACK_ITEM_1", "blocks": [{{"filename": "$DEPENDENCY_FILE_NAME_1", "code": "$COMPLETE_DEPS_FILE_CONTENT_1"}}]}}, {{"technology": "$TECH_STACK_ITEM_2", "blocks": [{{"filename": "$DEPENDENCY_FILE_NAME_2", "code": "$COMPLETE_DEPS_FILE_CONTENT_2"}}]}}, ... ] }}`

All code blocks must be valid for immediate use with appropriate tools (e.g., pip, dotnet, npm, mvn, gem, composer, etc.).
"""

    def create(self, llm: SupportedLLMs):
        model = get_llm(llm)
        return (
            PromptTemplate.from_template(self.DEPENDENCIES_GENERATION_PROMPT_TEMPLATE)
            | model
            | JsonOutputParser()
        )