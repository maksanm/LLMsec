
from chains.validation_chain import ValidationChain


class ValidationAgent:

    def __init__(self):
        self.validation_chain = ValidationChain().create()

    def invoke(self, state):
        is_valid_str = self.validation_chain.invoke(state)
        is_valid = self._parse_bool(is_valid_str)
        if not is_valid:
            return {
                "is_valid": False
            }
        else:
            return {
                "is_valid": True,
            }

    def _parse_bool(self, str_):
        if str_ == "True":
            return True
        elif str_ == "False":
            return False
        raise Exception("Unable to parse the LLM output")

