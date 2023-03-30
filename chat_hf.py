import sys
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM

if len(sys.argv) < 2:
    print("Run as: python3 chat_hf.py chavinlo/alpaca-native")
    sys.exit()

model_name = sys.argv[1]

PROMPT_DICT = {
    "prompt_input": (
        "Below is an instruction that describes a task, paired with an input that provides further context.\n"
        "아래는 작업을 설명하는 명령어와 추가적 맥락을 제공하는 입력이 짝을 이루는 예제입니다.\n\n"
        "Write a response that appropriately completes the request.\n요청을 적절히 완료하는 응답을 작성하세요.\n\n"
        "### Instruction(명령어):\n{instruction}\n\n### Input(입력):\n{input}\n\n### Response(응답):"
    ),
    "prompt_no_input": (
        "Below is an instruction that describes a task.\n"
        "아래는 작업을 설명하는 명령어입니다.\n\n"
        "Write a response that appropriately completes the request.\n명령어에 따른 요청을 적절히 완료하는 응답을 작성하세요.\n\n"
        "### Instruction(명령어):\n{instruction}\n\n### Response(응답):"
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