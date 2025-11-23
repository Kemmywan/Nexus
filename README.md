# Nexus

An 2024-2025-Summer prp

---

Teacher: Gui Jiaping

Collaborators: Wu Jiahang

# Net Detector

## Implement

Detect net data anomalies. 

* test on optc

```
python train_fasttext.py --dataset optc_day23-flow

python preprocess.py --dataset optc_day23-flow

python train_vae.py --dataset optc_day23-flow

python eval.py --dataset optc_day23-flow
```

## Prerequisites

```
conda create --name provenance python=3.12
pip install pandas
pip install tqdm
pip install git+https://github.com/casics/nostril.git
pip install gensim
pip3 install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121
pip install matplotlib
pip install seaborn
pip install streamz
pip install schedule
pip install nearpy
pip install pydot
pip install graphviz
...
```
