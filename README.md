# LLaMA Fine Tuning KO

**Install dependencies**

```bash
pip3 install virtualenv
python3 -m venv env
source env/bin/activate
pip3 install -r requirements.txt
pip3 install -e .
```

**base model**  
https://huggingface.co/decapoda-research/llama-7b-hf

**install model**

```bash
brew install git-lfs
git lfs install
git clone https://huggingface.co/decapoda-research/llama-7b-hf
```

---

### Alpaca setup

for chat model like ChatGPT

```bash
python3 export_state_dict_checkpoint.py 7B
python3 clean_hf_cache.py
```
