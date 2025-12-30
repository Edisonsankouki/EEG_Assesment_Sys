import importlib.util
from dataclasses import dataclass


@dataclass
class LLMResponse:
    text: str


def has_transformers():
    return importlib.util.find_spec("transformers") is not None


def has_llama_cpp():
    return importlib.util.find_spec("llama_cpp") is not None


def run_transformers(prompt, model_name):
    if not has_transformers():
        raise RuntimeError("transformers not installed")
    from transformers import AutoModelForCausalLM, AutoTokenizer

    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModelForCausalLM.from_pretrained(model_name)
    inputs = tokenizer(prompt, return_tensors="pt")
    output = model.generate(**inputs, max_new_tokens=512)
    text = tokenizer.decode(output[0], skip_special_tokens=True)
    return LLMResponse(text=text)


def run_llama_cpp(prompt, model_path):
    if not has_llama_cpp():
        raise RuntimeError("llama_cpp not installed")
    from llama_cpp import Llama

    llm = Llama(model_path=model_path)
    out = llm(prompt, max_tokens=512)
    return LLMResponse(text=out["choices"][0]["text"])
