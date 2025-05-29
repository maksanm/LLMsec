
from chains.stacks_identification_chain import StacksIdentificationChain


class StacksIdentificationAgent:

    def __init__(self):
        self.stacks_identification_chain = StacksIdentificationChain().create()

    def invoke(self, state):
        return {
            "tech_stacks": self.stacks_identification_chain.invoke(state)["tech_stacks"]
        }
