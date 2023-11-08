

def load_zephyr_7b_beta():

    import torch
    from transformers import BitsAndBytesConfig
    from llama_index.prompts import PromptTemplate
    from llama_index.llms import HuggingFaceLLM

    quantization_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_compute_dtype=torch.float16,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_use_double_quant=True,
    )


    def messages_to_prompt(messages):
        prompt = ""
        for message in messages:
            if message.role == 'system':
                prompt += f"<|system|>\n{message.content}</s>\n"
            elif message.role == 'user':
                prompt += f"<|user|>\n{message.content}</s>\n"
            elif message.role == 'assistant':
                prompt += f"<|assistant|>\n{message.content}</s>\n"

        # ensure we start with a system prompt, insert blank if needed
        if not prompt.startswith("<|system|>\n"):
            prompt = "<|system|>\n</s>\n" + prompt

        # add final assistant prompt
        prompt = prompt + "<|assistant|>\n"

        return prompt


    llm = HuggingFaceLLM(
        # model_name="HuggingFaceH4/zephyr-7b-beta",
        # tokenizer_name="HuggingFaceH4/zephyr-7b-beta",
        model_name="llms/zephyr-7b-beta",
        tokenizer_name="llms/zephyr-7b-beta",
        query_wrapper_prompt=PromptTemplate("<|system|>\n</s>\n<|user|>\n{query_str}</s>\n<|assistant|>\n"),
        context_window=4096,
        max_new_tokens=256,
        model_kwargs={"quantization_config": quantization_config},
        # tokenizer_kwargs={},
        # tokenizer_kwargs={"max_length": 2048}
        # generate_kwargs={"temperature": 0.25, "top_k": 50, "top_p": 0.95},
        generate_kwargs={"temperature": 0.25, "do_sample": False},
        messages_to_prompt=messages_to_prompt,
        device_map="auto",
    )

    return llm


def free_gpu_memory(llm_variable):

    import gc
    import torch

    if llm_variable in globals():
        llm = globals()[llm_variable]._model.cpu()
        del llm
        gc.collect()
        torch.cuda.empty_cache()

        