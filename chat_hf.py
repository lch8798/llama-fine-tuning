import sys
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM

if len(sys.argv) < 2:
    print("Run as: python3 chat_hf.py chavinlo/alpaca-native")
    sys.exit()

model_name = sys.argv[1]

PROMPT_DICT = {
    "prompt_input": (
        "Below is an instruction that describes a task, paired with an input that provides further context. "
        "Write a response that appropriately completes the request.\n\n"
        "### Instruction:\n{instruction}\n\n### Input:\n{input}\n\n### Response:"
    ),
    "prompt_no_input": (
        "Below is an instruction that describes a task. "
        "Write a response that appropriately completes the request.\n\n"
        "### Instruction:\n{instruction}\n\n### Response:"
    ),
}

tokenizer = AutoTokenizer.from_pretrained(model_name, cache_dir="./models/" + model_name)
model = AutoModelForCausalLM.from_pretrained(model_name, cache_dir="./models/" + model_name).to("mps")
    
def gen(prompt, user_input=None, max_new_tokens=128, temperature=0.5):
    if user_input:
        x = PROMPT_DICT['prompt_input'].format(instruction=prompt, input=user_input)
    else:
        x = PROMPT_DICT['prompt_no_input'].format(instruction=prompt)
    
    input_ids = tokenizer.encode(x, return_tensors="pt").to('mps')
    gen_tokens = model.generate(
        input_ids,
        max_new_tokens=max_new_tokens,
        num_return_sequences=1,
        temperature=temperature,
        no_repeat_ngram_size=6,
        do_sample=True,
    )
    gen_text = tokenizer.decode(gen_tokens[0], skip_special_tokens=True)
    
    return gen_text.replace(x, '')

def main():
    prompt = "Python으로 uptime을 찾는 코드"
    generated_text = gen(prompt)
    print(generated_text)
    
main()