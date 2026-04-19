import torch
import pickle
from seqeval.metrics import classification_report
from transformers import AutoTokenizer, AutoModelForTokenClassification

# =========================
# LOAD PREPROCESSING
# =========================
with open("word2idx.pkl", "rb") as f:
    word2idx = pickle.load(f)
with open("idx2tag.pkl", "rb") as f:
    idx2tag = pickle.load(f)
with open("X_test.pkl", "rb") as f:
    X_test = pickle.load(f)
with open("y_test.pkl", "rb") as f:
    y_test = pickle.load(f)

# =========================
# LIMIT TEST DATA
# =========================
LIMIT = 300
X_test = X_test[:LIMIT]
y_test = y_test[:LIMIT]
print(f"Using {LIMIT} test samples for evaluation")

# =========================
# MODEL BiLSTM
# =========================
import torch.nn as nn

class BiLSTM_POS(nn.Module):
    def __init__(self, vocab_size, tagset_size):
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, 128)
        self.lstm = nn.LSTM(128, 128, bidirectional=True, batch_first=True)
        self.fc = nn.Linear(256, tagset_size)

    def forward(self, x):
        x = self.embedding(x)
        x, _ = self.lstm(x)
        x = self.fc(x)
        return x

# Load model
model = BiLSTM_POS(len(word2idx), len(idx2tag))
model.load_state_dict(torch.load("bilstm_pos_model.pth"))
model.eval()
print("BiLSTM model loaded!")

# Predict function
def predict(model, X_test):
    predictions = []
    with torch.no_grad():
        for sent in X_test:
            sent_tensor = torch.tensor(sent).unsqueeze(0)
            output = model(sent_tensor)
            pred = torch.argmax(output, dim=2).squeeze().tolist()
            predictions.append(pred)
    return predictions

y_pred_idx = predict(model, X_test)

y_pred = [[idx2tag[i] for i in pred_seq] for pred_seq in y_pred_idx]
y_true = [[idx2tag[i] for i in true_seq] for true_seq in y_test]

print("\n==============================")
print("Evaluation Result (BiLSTM)")
print("==============================\n")
print(classification_report(y_true, y_pred))

# =========================
# EVALUATE INDO-BERT
# =========================
tokenizer = AutoTokenizer.from_pretrained("indobert_pos_model")
model_bert = AutoModelForTokenClassification.from_pretrained("indobert_pos_model")
model_bert.eval()

print("\nEvaluating IndoBERT...")

idx2word = {v:k for k,v in word2idx.items()}

y_pred_bert = []
y_true_bert = []

for sent, true_tags in zip(X_test, y_test):
    words = [idx2word[i] for i in sent]

    tokens = tokenizer(words, is_split_into_words=True, return_tensors="pt", truncation=True)

    with torch.no_grad():
        outputs = model_bert(**tokens)

    predictions = torch.argmax(outputs.logits, dim=2)[0]
    word_ids = tokens.word_ids()

    pred_tags = []
    true_tags_text = []
    prev_word = None

    for token_idx, word_id in enumerate(word_ids):
        if word_id is None:
            continue
        if word_id != prev_word:
            pred_tags.append(idx2tag[predictions[token_idx].item()])
            true_tags_text.append(idx2tag[true_tags[word_id]])
        prev_word = word_id

    y_pred_bert.append(pred_tags)
    y_true_bert.append(true_tags_text)

print("\n==============================")
print("Evaluation Result (IndoBERT)")
print("==============================\n")
print(classification_report(y_true_bert, y_pred_bert))

print("\nIndoBERT model loaded!")

# =========================
# TEST UNTUK MENGUJI
# =========================

test_sentence = "Aku suka makan indomie dan ayam, apalagi ayam bakar yang di jual di daerah sana"

print("\n==============================")
print("Custom Sentence POS Tagging")
print("==============================")
print("Sentence:", test_sentence)

# split sederhana
words = test_sentence.replace(",", " ,").split()

# =========================
# BiLSTM PREDICTION
# =========================
encoded = [word2idx.get(w, 0) for w in words]  # unknown word -> 0
sent_tensor = torch.tensor(encoded).unsqueeze(0)

with torch.no_grad():
    output = model(sent_tensor)

pred = torch.argmax(output, dim=2).squeeze().tolist()
bilstm_tags = [idx2tag[i] for i in pred]

print("\nBiLSTM Prediction:")
for w, t in zip(words, bilstm_tags):
    print(f"{w:12} -> {t}")

# =========================
# INDO-BERT PREDICTION
# =========================
tokens = tokenizer(words, is_split_into_words=True, return_tensors="pt", truncation=True)

with torch.no_grad():
    outputs = model_bert(**tokens)

predictions = torch.argmax(outputs.logits, dim=2)[0]
word_ids = tokens.word_ids()

bert_tags = []
prev_word = None

for token_idx, word_id in enumerate(word_ids):
    if word_id is None:
        continue
    if word_id != prev_word:
        bert_tags.append(idx2tag[predictions[token_idx].item()])
    prev_word = word_id

print("\nIndoBERT Prediction:")
for w, t in zip(words, bert_tags):
    print(f"{w:12} -> {t}")