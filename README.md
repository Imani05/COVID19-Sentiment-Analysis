# COVID19-Sentiment-Analysis

This research explores the optimization of transfer learning strategies for sentiment analysis, focusing on COVID-19 related tweets. The study comparatively analyses the performance of BERT and RoBERTa models against a Naive Bayes baseline, addressing the unique challenges posed by pandemic-related social media content. The research aims to enhance our understanding of public sentiment during a global health crisis and improve the efficacy of sentiment analysis techniques in this context.

## Setup

### Prerequisites

- Python 3.9+
- pip

### Installation

```bash
# Clone the repository
git clone <repo-url>
cd COVID19-Sentiment-Analysis

# (Optional) Create a virtual environment
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt

# Download NLTK data (required for tokenization)
python -c "import nltk; nltk.download('stopwords'); nltk.download('punkt')"
```

### Running the Notebook

```bash
jupyter notebook "COVID19 Sentiment Analysis.ipynb"
```

Run cells sequentially from top to bottom. The notebook will automatically download pre-trained BERT and RoBERTa weights from Hugging Face on first run (requires an internet connection).

## Datasets

The study utilizes two datasets, `Corona_train.csv` and `Corona_test.csv`, comprising COVID-19 related tweets. A comprehensive preprocessing pipeline is implemented, including emoji removal, hashtag cleaning, and special character filtering.

The sentiment labels are consolidated into three classes:

| Original Label     | Mapped Class |
|--------------------|--------------|
| Extremely Negative | Negative (0) |
| Negative           | Negative (0) |
| Neutral            | Neutral (1)  |
| Positive           | Positive (2) |
| Extremely Positive | Positive (2) |

## Methodology

The preprocessing pipeline applies the following steps to each tweet:

1. Emoji removal
2. URL and mention removal
3. Non-ASCII character removal
4. Hashtag cleaning (trailing hashtags removed; mid-sentence `#` symbols stripped)
5. Special character filtering (`$`, `&`, punctuation)
6. Whitespace normalization

After cleaning, samples with fewer than 5 words or more than 80 tokens (after BERT tokenization) are removed to exclude malformed or non-English entries.

A `RandomOverSampler` is applied to the training set to address class imbalance before the train/validation split.

Models trained:

- **Naive Bayes** (TF-IDF baseline)
- **BERT** (`bert-base-uncased`, fine-tuned for 10 epochs)
- **RoBERTa** (`roberta-base`, fine-tuned for 10 epochs)

All transformer models use:
- Maximum sequence length: 128 tokens
- Batch size: 16
- Optimizer: Adam (lr = 3e-5)
- Early stopping (patience = 3) with best-weights restoration
- Learning rate reduction on plateau

## Results

Results demonstrate the superiority of transformer-based models over the Naive Bayes baseline. BERT and RoBERTa achieved overall accuracies of 90% and 90.36% respectively, significantly outperforming Naive Bayes (71%). Both models exhibited high performance in distinguishing between positive and negative sentiments, with F1-scores exceeding 0.91. However, the neutral class proved challenging for all models, highlighting the complexity of capturing neutrality in social media content.

## Model Convergence

The study reveals interesting differences in model convergence patterns, with BERT showing faster initial convergence but potential overfitting in later epochs, while RoBERTa demonstrated more gradual but steady improvement. These findings contribute to our understanding of the trade-offs between different transfer learning approaches in sentiment analysis.

## Broader Implications

The research also addresses broader implications, including the potential application of these models in crisis communication and public health monitoring. It highlights the importance of computational efficiency and ethical considerations in deploying such models in real-world scenarios.

## Limitations and Future Work

While the study provides valuable insights, it acknowledges limitations such as dataset constraints, language limitations, and the need for more fine-grained sentiment analysis. Future research directions include:

- Development of multilingual models
- Exploration of model interpretability techniques (e.g., LIME, SHAP)
- Investigation of real-time sentiment analysis systems for crisis management

## Conclusion

This study contributes to the growing body of research on advanced NLP techniques for sentiment analysis in the context of public health crises, offering insights that can inform both academic understanding and practical applications in crisis communication and sentiment monitoring.
