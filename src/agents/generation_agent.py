from enum import Enum
from chains.code_generation_chain import CodeGenerationChain, SupportedLLMs
from chains.dependencies_generation_chain import DependenciesGenerationChain


class GenerationModes(str, Enum):
    CODE = "code"
    DEPENDENCIES = "dependencies"

class GenerationAgent:

    def __init__(self, llm: SupportedLLMs):
        self.llm = llm
        self.code_generation_chain = CodeGenerationChain().create(llm)
        self.dependencies_generation_chain = DependenciesGenerationChain().create(llm)

    def invoke(self, state):
        try:
            match state["generation_mode"]:
                case GenerationModes.CODE:
                    code_blocks = self.code_generation_chain.invoke(state)["code_blocks"]
                case GenerationModes.DEPENDENCIES:
                    code_blocks = self.dependencies_generation_chain.invoke(state)["code_blocks"]
                case _:
                    raise ValueError(f"Unknown generation_mode: {state['generation_mode']}.")
            return {
                "llm_codeblocks_pairs": [(self.llm, code_blocks)]
            }
        except Exception as e:
            print((f"Exception for LLM '{self.llm}': {e}"))
            return {
                "llm_codeblocks_pairs": [(self.llm, {})]
            }
