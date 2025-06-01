
from chains.validation_chain import ValidationChain
from llm_provider import SupportedLLMs


class ValidationAgent:

    def __init__(self, llm: SupportedLLMs):
        self.validation_chain = ValidationChain().create(llm)

    def invoke(self, state):
        is_valid_str = self.validation_chain.invoke(state)
        is_valid = self._parse_bool(is_valid_str)
        return {
            "is_valid": is_valid
        }

    def _parse_bool(self, str_):
        if str_.lower() == "true":
            return True
        elif str_ .lower() == "false":
            return False
        raise Exception(f"Unable to parse the LLM output: {str_}")

