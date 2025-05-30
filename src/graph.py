from langgraph.graph import START, END, StateGraph
from langgraph.types import Send
from typing import Annotated, TypedDict, List
from operator import add

from agents.validation_agent import ValidationAgent
from agents.stacks_identification_agent import StacksIdentificationAgent
from agents.code_generation_agent import CodeGenerationAgent, SupportedLLMs


class TaskState(TypedDict):
    task_description: str
    tech_stacks: List[str]
    is_valid: bool
    llm_codeblocks_pairs: Annotated[List[tuple[str, dict]], add]


class GraphFactory:

    llm_agents = []

    def _validation_routing(self, state):
        if state["is_valid"]:
            return "stacks_identification_agent"
        else:
            return END

    def _code_generation_routing(self, state: TaskState):
        llm_agents_forward_instructions = []
        for tech_stack in state["tech_stacks"]:
            llm_agent_state = {
                "task_description": state["task_description"],
                "tech_stack": tech_stack
            }
            for llm_agent in self.llm_agents:
                llm_agents_forward_instructions.append(Send(llm_agent, llm_agent_state))
        return llm_agents_forward_instructions

    def create(self):
        graph = StateGraph(TaskState)

        graph.add_node("validation_agent", ValidationAgent(llm=SupportedLLMs.OPENAI).invoke)
        graph.add_node("stacks_identification_agent", StacksIdentificationAgent(llm=SupportedLLMs.OPENAI).invoke)
        for llm in SupportedLLMs:
            graph.add_node(f"{llm}_code_agent", CodeGenerationAgent(llm).invoke)
            self.llm_agents.append(f"{llm}_code_agent")

        graph.add_edge(START, "validation_agent")
        graph.add_conditional_edges("validation_agent", self._validation_routing)
        graph.add_conditional_edges("stacks_identification_agent", self._code_generation_routing)
        for llm_agent in self.llm_agents:
            graph.add_edge(llm_agent, END)

        return graph.compile()
