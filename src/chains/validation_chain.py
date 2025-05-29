import os
from langchain_core.prompts import PromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_deepseek import ChatDeepSeek
from langchain_openai import ChatOpenAI


class ValidationChain:
    VALIDATION_PROMPT_TEMPLATE = """Validate if the user request specifies a clear programming task, such as creating a web service, writing a DevOps pipeline, building a database, implementing an authorization flow, or other software development tasks. Return 'True' if it does, otherwise return 'False'.

User request:
```
{task_description}
```

Response:
"""

    def create(self):
        if os.getenv("OPENAI_API_KEY"):
            llm = ChatOpenAI(model_name="gpt-4.1-nano", temperature=0.0)
        elif os.getenv("DEEPSEEK_API_KEY"):
            llm = ChatDeepSeek(model_name="deepseek-chat", temperature=0.0)
        else:
            raise Exception("No API key provided for any supported model")
        return (
            PromptTemplate.from_template(self.VALIDATION_PROMPT_TEMPLATE)
            | llm
            | StrOutputParser()
        )