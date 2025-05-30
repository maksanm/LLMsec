
from chains.stacks_identification_chain import StacksIdentificationChain
from llm_provider import SupportedLLMs


class StacksIdentificationAgent:

    def __init__(self, llm: SupportedLLMs):
        self.stacks_identification_chain = StacksIdentificationChain().create(llm)

    def invoke(self, state):
        return {
            "tech_stacks": self.stacks_identification_chain.invoke(state)["tech_stacks"]
        }
