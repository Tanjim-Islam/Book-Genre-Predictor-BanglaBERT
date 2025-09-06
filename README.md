# Book Genre Predictor using BanglaBERT

A sophisticated machine learning project that predicts book genres from Bengali book summaries and reviews using state-of-the-art transformer models, specifically fine-tuned BanglaBERT.

## 🎯 Project Overview

This project implements an automated book genre classification system for Bengali literature. It analyzes Bengali book summaries, reviews, and descriptions to predict their genres with high accuracy. The system leverages the power of BanglaBERT (Bengali BERT) and traditional word embeddings to understand the semantic content of Bengali text.

### Key Features

- **Bengali Language Support**: Specifically designed for Bengali text processing and analysis
- **Multi-Genre Classification**: Classifies books into 7 distinct genres
- **High Accuracy**: Achieves 96.7% validation accuracy and F1-score
- **Dual Approach**: Implements both transformer-based (BanglaBERT) and traditional word embedding approaches
- **Comprehensive Preprocessing**: Advanced text cleaning and preprocessing for Bengali text

## 📚 Supported Genres

The model classifies books into the following 7 genres:

| Label | Genre | Bengali |
|-------|-------|---------|
| 0 | Fiction | কথাসাহিত্য |
| 1 | Thriller | থ্রিলার |
| 2 | Children's Book | শিশুতোষ |
| 3 | Political | রাজনৈতিক |
| 4 | Science Fiction | বৈজ্ঞানিক কল্পকাহিনী |
| 5 | War | যুদ্ধ |
| 6 | Motivational | অনুপ্রেরণামূলক |

## 🏗️ Technical Architecture

### Model Implementation

1. **BanglaBERT Fine-tuning**: 
   - Pre-trained model: `sagorsarker/bangla-bert-base`
   - Fine-tuned for sequence classification with 7 output classes
   - Maximum sequence length: 512 tokens
   - Optimized with AdamW optimizer (learning rate: 2e-5)

2. **Word2Vec Embeddings**:
   - Uses Bengali Word2Vec from BNLP library
   - Document-level embeddings through word vector averaging
   - 100-dimensional word vectors

3. **Text Preprocessing**:
   - Removal of metadata terms (নাম, লেখক, প্রকাশনী, etc.)
   - Bengali text normalization
   - Tokenization and padding for transformer input

### Dataset Structure

- **Training Set**: 3,886 labeled book summaries
- **Test Set**: 687 unlabeled summaries for prediction
- **Features**: Raw Bengali text summaries and cleaned versions
- **Labels**: Numerical labels (0-6) corresponding to genres

## 📊 Performance Metrics

The model demonstrates excellent performance across all genres:

```
              precision    recall  f1-score   support
           0       0.99      0.95      0.97       144
           1       0.98      0.96      0.97        54
           2       0.97      0.98      0.97        59
           3       0.84      1.00      0.91        41
           4       1.00      1.00      1.00        43
           5       0.97      0.91      0.94        34
           6       1.00      1.00      1.00        14
    accuracy                           0.97       389
   macro avg       0.96      0.97      0.97       389
weighted avg       0.97      0.97      0.97       389
```

- **Overall Accuracy**: 96.7%
- **Macro F1-Score**: 96.7%
- **Training Accuracy**: 93.0% (final epoch)

## 🚀 Getting Started

### Prerequisites

```bash
# Core ML libraries
torch>=1.9.0
transformers>=4.0.0
scikit-learn>=0.24.0

# Bengali NLP
bnlp>=1.0.0

# Data processing
pandas>=1.3.0
numpy>=1.21.0

# Visualization
matplotlib>=3.4.0

# Progress tracking
tqdm>=4.62.0
```

### Installation

1. **Clone the repository**:
```bash
git clone https://github.com/Tanjim-Islam/Book-Genre-Predictor-BanglaBERT.git
cd Book-Genre-Predictor-BanglaBERT
```

2. **Install dependencies**:
```bash
pip install torch transformers scikit-learn pandas numpy matplotlib tqdm bnlp
```

3. **Download the dataset**:
   - Training data: `Dataset/train.csv`
   - Test data: `Dataset/test.csv`
   - Pre-processed data: `Dataset/rab/`

### Usage

#### Training the Model

Open and run the Jupyter notebook `model.ipynb`:

```python
# Load and preprocess data
train_df = pd.read_csv('Dataset/rab/train_df.csv')
test_df = pd.read_csv('Dataset/rab/test_df.csv')

# Initialize BanglaBERT model
from transformers import AutoModelForSequenceClassification, AutoTokenizer

model_name = "sagorsarker/bangla-bert-base"
tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModelForSequenceClassification.from_pretrained(model_name, num_labels=7)

# Train the model (4 epochs recommended)
# ... (see notebook for complete training loop)
```

#### Making Predictions

```python
# Predict genre for a new Bengali book summary
def predict_genre(text, model, tokenizer):
    # Tokenize and predict
    inputs = tokenizer(text, padding=True, truncation=True, 
                      max_length=512, return_tensors="pt")

    with torch.no_grad():
        outputs = model(**inputs)
        prediction = torch.argmax(outputs.logits, dim=1)

    # Convert to genre name
    genres = ['Fiction', 'Thriller', 'Childrens-Book', 'Political', 
              'Science-Fiction', 'War', 'Motivational']
    return genres[prediction.item()]

# Example usage
summary = "একটি রোমাঞ্চকর গোয়েন্দা গল্প যেখানে একজন গোয়েন্দা একটি রহস্যজনক হত্যাকাণ্ডের সমাধান করার চেষ্টা করে।"
predicted_genre = predict_genre(summary, model, tokenizer)
print(f"Predicted Genre: {predicted_genre}")
```

## 📁 Repository Structure

```
Book-Genre-Predictor-BanglaBERT/
├── model.ipynb                 # Main implementation notebook
├── Dataset/
│   ├── train.csv              # Training dataset
│   ├── test.csv               # Test dataset
│   ├── bengali_word2vec.model # Pre-trained Word2Vec model
│   └── rab/
│       ├── train_df.csv       # Processed training data
│       └── test_df.csv        # Processed test data
├── Models/
│   ├── Model/
│   │   └── bnwiki_word2vec.model
│   └── BengaliWord2Vec Code/
└── README.md                  # Project documentation
```

## 🔬 Technical Implementation Details

### Data Preprocessing Pipeline

1. **Text Cleaning**:
   - Removal of book metadata terms (নাম, লেখক, প্রকাশনী, ধরণ, বই, মূল্য, পৃষ্টা, প্রকাশক, অনুবাদক, বইয়ের)
   - Handling missing values with Bengali placeholder text
   - Normalization of Bengali text

2. **Feature Engineering**:
   - Document-level embeddings using Bengali Word2Vec
   - BERT tokenization with attention masks
   - Sequence padding/truncation to 512 tokens

3. **Model Training**:
   - 90-10 train-validation split
   - Batch size: 16
   - Learning rate: 2e-5 with AdamW optimizer
   - 4 training epochs with validation monitoring

### Model Architecture

- **Base Model**: BanglaBERT (sagorsarker/bangla-bert-base)
- **Classification Head**: Linear layer with 7 output neurons
- **Loss Function**: CrossEntropyLoss
- **Evaluation Metrics**: Accuracy, F1-score (macro and weighted)

## 📈 Results and Analysis

### Training Progress

| Epoch | Training Loss | Training Accuracy | Validation Loss | Validation Accuracy | Validation F1 |
|-------|---------------|-------------------|-----------------|-------------------|---------------|
| 1     | 1.1369        | 58.75%           | 0.5307          | 84.58%           | 82.20%        |
| 2     | 0.4944        | 82.96%           | 0.2911          | 89.72%           | 88.42%        |
| 3     | 0.2929        | 90.14%           | 0.1403          | 95.89%           | 95.42%        |
| 4     | 0.1924        | 93.00%           | 0.0961          | 96.66%           | 96.70%        |

### Class Distribution

The dataset shows a balanced distribution across genres, with Fiction being the most common:

- Fiction: 1,345 samples (34.6%)
- Thriller: 702 samples (18.1%)
- Children's Book: 497 samples (12.8%)
- Political: 439 samples (11.3%)
- Science Fiction: 410 samples (10.5%)
- War: 297 samples (7.6%)
- Motivational: 196 samples (5.0%)

## 🌟 Applications and Use Cases

1. **Digital Libraries**: Automatic categorization of Bengali books
2. **Bookstores**: Enhanced search and recommendation systems
3. **Publishing Houses**: Content management and classification
4. **Educational Platforms**: Curriculum-based book organization
5. **Research**: Bengali literature analysis and trend identification

## 🤝 Contributing

Contributions are welcome! Please feel free to submit a Pull Request. For major changes, please open an issue first to discuss what you would like to change.

### Development Setup

1. Fork the repository
2. Create a feature branch (`git checkout -b feature/amazing-feature`)
3. Commit your changes (`git commit -m 'Add some amazing feature'`)
4. Push to the branch (`git push origin feature/amazing-feature`)
5. Open a Pull Request

## 📄 License

This project is open source and available under the [MIT License](LICENSE).

## 🙏 Acknowledgments

- **BanglaBERT**: [sagorsarker/bangla-bert-base](https://huggingface.co/sagorsarker/bangla-bert-base)
- **BNLP**: Bengali Natural Language Processing library
- **Transformers**: Hugging Face Transformers library
- **Dataset Contributors**: Thanks to all who contributed to the Bengali book dataset

## 📧 Contact

For questions, suggestions, or collaborations, please reach out:

- **Repository Owner**: [Tanjim-Islam](https://github.com/Tanjim-Islam)
- **Project Repository**: [Book-Genre-Predictor-BanglaBERT](https://github.com/Tanjim-Islam/Book-Genre-Predictor-BanglaBERT)

---

*This project demonstrates the power of transformer models in Bengali natural language processing and contributes to the advancement of Bengali computational linguistics.*
