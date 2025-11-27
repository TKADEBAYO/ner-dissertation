 # Named Entity Recognition Dissertation Project

  ## Overview

  This project is a comparative study of Named Entity Recognition (NER) approaches, focusing on evaluating off-the-shelf models against domain-aware fine-tuned models across multiple datasets. It involves downloading, preprocessing, training, fine-tuning, and evaluating models on four benchmark NER datasets: CoNLL-2003, WNUT-17, BC5CDR, and JNLPBA.

  The goal is to analyze performance differences between general-purpose models (spaCy, HuggingFace Transformers) and domain-specific fine-tuned BERT models, providing insight into the effectiveness of domain adaptation in NER tasks.

  ## Project Structure

ner-dissertation/
├── data/
│ ├── raw/ # Raw downloaded datasets
│ └── processed/ # Preprocessed datasets ready for training/evaluation
├── outputs/
│ ├── graphs/ # Evaluation plots and charts
│ └── metrics/ # Evaluation metrics and logs
├── src/ # Source code scripts
│ ├── download_datasets.py
│ ├── preprocess.py
│ ├── train_finetune.py
│ ├── evaluate_baseline.py
│ └── evaluate_models.py
├── requirements.txt # Python dependencies
├── README.md # This file


## Setup Instructions

1. **Create and activate a Python virtual environment** 

   python -m venv venv
   source venv/bin/activate  # Linux/macOS
   venv\Scripts\activate     # Windows PowerShell
   ```

2. **Install dependencies**:

   
   pip install -r requirements.txt
   ```

3. **Download datasets**:

   Run the dataset download script to fetch and save raw datasets:

   
   python src/download_datasets.py
   
4. **Preprocess datasets**:

   Convert raw datasets into standardized JSONL format:

   python src/preprocess.py
   

## Running Experiments

- **Fine-tune models** on specific datasets:

  python src/train_finetune.py --dataset conll2003
  python src/train_finetune.py --dataset wnut17
  python src/train_finetune.py --dataset bc5cdr
  python src/train_finetune.py --dataset jnlpba
  ```

- **Evaluate off-the-shelf models** (spaCy and HuggingFace):

  python src/evaluate_baseline.py
  ```

- **Compare all model evaluations and generate summary reports and plots**:

  ```
  python src/evaluate_models.py
  ```

## Outputs

- Evaluation metrics for each dataset and model are saved in `outputs/metrics/`.
- Comparison plots visualizing model performance across datasets are saved in `outputs/graphs/`.
- Fine-tuned models are saved in `models/` directory (if implemented).

## Notes

Model fine-tuning used pre-trained BERT base models.
Token alignment, subword masking, and class imbalance mitigation techniques were applied.
Metrics include entity-level precision, recall, and F1 across all datasets.
Due to large file sizes, full model weights are stored only on GitHub.

## Contact

For questions or issues, please contact the project author.

Joseph Adebayo
Email: aja23@st-andrews.ac.uk
Phone: 07401563425

