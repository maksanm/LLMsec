from enum import Enum

class SupportedLLMs(Enum):
    OPENAI_41 = 1
    DEEPSEEK_V3 = 2


class CodeGenerationAgent:

    def __init__(self, llm: SupportedLLMs):
        return

    def invoke(self, state):
        return {}