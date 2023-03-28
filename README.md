# LLaMA Fine Tuning KO

**Install dependencies**

```bash
pip3 install virtualenv
python3 -m venv env
source env/bin/activate
pip3 install -r requirements.txt
pip3 install -e .
```

---

### LLaMA setup

for inference

**base model**  
https://huggingface.co/decapoda-research/llama-7b-hf

**install model**

```bash
brew install git-lfs
git lfs install
git clone https://huggingface.co/decapoda-research/llama-7b-hf
```

(move to /models/7b after install)

**example of execution**

```bash
python3 chat.py --ckpt_dir models/7b --tokenizer_path models/7b/tokenizer.model --max_batch_size 8 --max_seq_len 256
```

---

### Alpaca setup

for chat model like ChatGPT

**base model**

**install model**

```bash
python3 export_state_dict_checkpoint.py 7B
python3 clean_hf_cache.py
```
