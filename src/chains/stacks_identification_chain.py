import os
from langchain_core.prompts import PromptTemplate
from langchain_core.output_parsers import JsonOutputParser
from langchain_core.runnables import RunnablePassthrough

from utils import SupportedLLMs, get_llm


class StacksIdentificationChain:
    STACKS_IDENTIFICATION_PROMPT_TEMPLATE = """Given the following programming task, identify up to {TECH_STACKS_LIMIT} of the most popular and distinct technology stacks suitable for implementing the task.

Each tech stack should be a single string, listing the main technologies (such as frameworks and languages) required for the key components (for example: frontend framework, backend language/framework, and database). Do not repeat any specific technology across different stacks. Only choose widely used and relevant technologies for the task.

Return your answer as a JSON object containing an array called "tech_stacks". For example, if the limit is 2 stacks and the task is to create a web service:
{{
    "tech_stacks": [
        "Angular, ASP.NET, SQL Server",
        "React, Java Spring, MySQL"
    ]
}}

User programming task:
{task_description}

Response:
"""

    def create(self, llm: SupportedLLMs):
        model = get_llm(llm)
        return (
            RunnablePassthrough.assign(
                TECH_STACKS_LIMIT=lambda _: os.getenv("TECH_STACKS_LIMIT")
            )
            | PromptTemplate.from_template(self.STACKS_IDENTIFICATION_PROMPT_TEMPLATE)
            | model
            | JsonOutputParser()
        )