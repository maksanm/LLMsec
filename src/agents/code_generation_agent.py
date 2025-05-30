from chains.code_generation_chain import CodeGenerationChain, SupportedLLMs


class CodeGenerationAgent:

    def __init__(self, llm: SupportedLLMs):
        self.llm = llm
        self.stacks_identification_chain = CodeGenerationChain().create(llm)

    def invoke(self, state):
        try:
            code_blocks = self.stacks_identification_chain.invoke(state)["code_blocks"]
            return {
                "llm_codeblocks_pairs": [(self.llm, code_blocks)]
            }
        except Exception as e:
            print((f"Exception for LLM '{self.llm}': {e}"))
            return {
                "llm_codeblocks_pairs": [(self.llm, {})]
            }
