from langgraph.graph import START, END, StateGraph
from langgraph.types import Send
from typing import Annotated, TypedDict, List
from operator import add

from agents.validation_agent import ValidationAgent
from agents.stacks_identification_agent import StacksIdentificationAgent
from agents.code_generation_agent import CodeGenerationAgent, SupportedLLMs


class TaskState(TypedDict):
    task_description: str
    tech_stacks: Annotated[List[str], add]


class GraphFactory:

    def _route_to_code_generation_agents(self, state: TaskState):
        llm_agents_forward_instructions = []
        for tech_stack in state["tech_stacks"]:
            llm_agent_state = {
                "task_description": state["task_description"],
                "tech_stack": tech_stack
            }
            llm_agents_forward_instructions.append(Send("openai_code_generation_agent", llm_agent_state))
            llm_agents_forward_instructions.append(Send("deepseek_code_generation_agent", llm_agent_state))
        return llm_agents_forward_instructions

    def create(self):
        graph = StateGraph(TaskState)

        graph.add_node("validation_agent", ValidationAgent().invoke)
        graph.add_node("stacks_identification_agent", StacksIdentificationAgent().invoke)
        graph.add_node("openai_code_generation_agent", CodeGenerationAgent(llm=SupportedLLMs.OPENAI_41).invoke)
        graph.add_node("deepseek_code_generation_agent", CodeGenerationAgent(llm=SupportedLLMs.DEEPSEEK_V3).invoke)

        graph.add_edge(START, "validation_agent")
        graph.add_edge("validation_agent", "stacks_identification_agent")
        graph.add_conditional_edges("stacks_identification_agent", self._route_to_code_generation_agents)
        graph.add_edge("openai_code_generation_agent", END)
        graph.add_edge("deepseek_code_generation_agent", END)

        return graph.compile()
