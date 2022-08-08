Lifelong Online Product Quantization.
=======

## Dataset

### Text dataset
1. NQ-dataset Download
   - DPR official code : [LINK](https://github.com/facebookresearch/DPR)
   - Do below scripts in DPR official code.
   ```bash
   cd dpr
   python data/download_data.py --resource "data.wikipedia_split.psgs_w100"
   python data/download_data.py --resource "data.retriever.nq-dev"
   python data/download_data.py --resource "data.retriever.nq-train"
   python data/download_data.py --resource "data.retriever.nq-adv-hn-train"
   - ```
   - Then copy downloads folder in this folder


### Preprocessing text dataset.
1. NQ-dataset preprocessing
   - Fill region of `data/NQ_dataset/preprocessing/PREPROCESS_DATA/NQ_preprocessing.py`
   ```python
   # Fill here
   # Total Dataset Size: 65395
   # Data path
   NQ_DATA_PATH = "/Users/hyojungo/PycharmProjects/ai-cv-tmp/product_quantization/PQ_text_retrieval/downloads/data/retriever"
   # 1st pretraining dataset config
   PRE_TRAIN_SIZE = 60295
   PRE_TEST_SIZE = 100
   # Online Continual Dataset (T ROUND)
   NUM_ROUND = 5
   ROUND_TRAIN = 900
   ROUND_TEST = 100
   ```
   - run python NQ_preprocessing.py


## Pretraining
### Text pretraining
- `python pretrain_models.py`
## Online Continual training
- `python online_continual.py`
