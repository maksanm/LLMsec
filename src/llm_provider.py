import os
from enum import Enum

from langchain_openai import ChatOpenAI
from langchain_deepseek import ChatDeepSeek
from langchain_xai import ChatXAI


class SupportedLLMs(str, Enum):
    OPENAI = "gpt-4.1"
    DEEPSEEK = "deepseek-chat"
    GROK = "grok-3"

LOWCOST_MAP = {
    SupportedLLMs.OPENAI: "gpt-4.1-mini",
    SupportedLLMs.DEEPSEEK: "deepseek-chat",
    SupportedLLMs.GROK: "grok-3-mini"
}

def get_llm(llm: SupportedLLMs):
    match llm:
        case SupportedLLMs.OPENAI:
            return ChatOpenAI(model_name=llm, temperature=0.0)
        case SupportedLLMs.DEEPSEEK:
            return ChatDeepSeek(model_name=llm, temperature=0.0)
        case SupportedLLMs.GROK:
            return ChatXAI(model_name=llm, temperature=0.0)
        case _:
            raise ValueError(f"Unsupported LLM: {llm}")

def get_lowcost_llm(llm: SupportedLLMs):
    match llm:
        case SupportedLLMs.OPENAI:
            return ChatOpenAI(model_name=LOWCOST_MAP[llm], temperature=0.0)
        case SupportedLLMs.DEEPSEEK:
            return ChatDeepSeek(model_name=LOWCOST_MAP[llm], temperature=0.0)
        case SupportedLLMs.GROK:
            return ChatXAI(model_name=LOWCOST_MAP[llm], temperature=0.0)
        case _:
            raise ValueError(f"Unsupported LLM: {llm}")