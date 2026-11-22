# Emotion Classification with Attention (GoEmotions)

Multi-class emotion classifier identifying **6 core Ekman emotions** (`joy`, `sadness`, `anger`, `fear`, `surprise`, `disgust`) on Google's **GoEmotions** dataset.

## Models Compared
1. **LSTM** (with pre-trained GloVe `6B.100d` embeddings)
2. **GRU** (with pre-trained GloVe `6B.100d` embeddings)
3. **BiLSTM + Bahdanau Attention** (with per-token attention weight heatmaps)
4. **Fine-tuned DistilBERT** (HuggingFace `Trainer` API with weighted Cross-Entropy loss)

## Quick Start (Google Colab — T4 GPU)
1. Upload `emotion_classification_attention.ipynb` to Google Colab.
2. Select `Runtime -> Change runtime type -> T4 GPU`.
3. Click `Runtime -> Run all` (~10–15 minutes total).
