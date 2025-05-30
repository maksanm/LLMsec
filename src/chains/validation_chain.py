from langchain_core.prompts import PromptTemplate
from langchain_core.output_parsers import StrOutputParser

from llm_provider import SupportedLLMs, get_lowcost_llm


class ValidationChain:
    VALIDATION_PROMPT_TEMPLATE = """Validate if the user request specifies a clear programming task, such as creating a web service, writing a DevOps pipeline, building a database, implementing an authorization flow, or other software development tasks. Return true string if it does, otherwise return false.

User request:
{task_description}

Response:
"""

    def create(self, llm: SupportedLLMs):
        model = get_lowcost_llm(llm)
        return (
            PromptTemplate.from_template(self.VALIDATION_PROMPT_TEMPLATE)
            | model
            | StrOutputParser()
        )