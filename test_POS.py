from conllu import parse_incr
import torch
import torch.nn as nn
import pickle
from collections import defaultdict
from transformers import AutoTokenizer, AutoModelForTokenClassification
from torch.optim import AdamW
from torch.utils.data import DataLoader, TensorDataset

# =========================
# LOAD DATASET
# =========================
def load_conllu(file_path):
    sentences, tags = [], []
    with open(file_path, "r", encoding="utf-8") as f:
        for tokenlist in parse_incr(f):
            words, pos = [], []
            for token in tokenlist:
                if isinstance(token["id"], int):
                    words.append(token["form"])
                    pos.append(token["upostag"])
            sentences.append(words)
            tags.append(pos)
    return sentences, tags

train_sentences, train_tags = load_conllu("UD_Indonesian-GSD-master/id_gsd-ud-train.conllu")
test_sentences, test_tags = load_conllu("UD_Indonesian-GSD-master/id_gsd-ud-test.conllu")
print("Dataset loaded:", len(train_sentences))

# =========================
# BUILD VOCAB
# =========================
word2idx = defaultdict(lambda: len(word2idx))
tag2idx = defaultdict(lambda: len(tag2idx))

for sent in train_sentences:
    for word in sent:
        word2idx[word]
for tag_seq in train_tags:
    for tag in tag_seq:
        tag2idx[tag]
idx2tag = {v:k for k,v in tag2idx.items()}

print("Vocab size:", len(word2idx))
print("Tag size:", len(tag2idx))

# =========================
# ENCODE DATA
# =========================
def encode(sentences, tags):
    X, y = [], []
    for sent, tag_seq in zip(sentences, tags):
        X.append([word2idx[w] for w in sent])
        y.append([tag2idx[t] for t in tag_seq])
    return X, y

X_train, y_train = encode(train_sentences[:1000], train_tags[:1000])  # BiLSTM & IndoBERT 1000 data
X_test, y_test = encode(test_sentences, test_tags)

# =========================
# SAVE PREPROCESSING
# =========================
with open("word2idx.pkl", "wb") as f:
    pickle.dump(dict(word2idx), f)
with open("idx2tag.pkl", "wb") as f:
    pickle.dump(idx2tag, f)
with open("X_test.pkl", "wb") as f:
    pickle.dump(X_test, f)
with open("y_test.pkl", "wb") as f:
    pickle.dump(y_test, f)
print("Preprocessing saved!")

# =========================
# MODEL BiLSTM
# =========================
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

model = BiLSTM_POS(len(word2idx), len(tag2idx))
loss_fn = nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(model.parameters(), lr=0.001)

for epoch in range(3):
    total_loss = 0
    for sent, tag in zip(X_train, y_train):
        sent = torch.tensor(sent).unsqueeze(0)
        tag = torch.tensor(tag)
        optimizer.zero_grad()
        output = model(sent)
        output = output.view(-1, len(tag2idx))
        loss = loss_fn(output, tag)
        loss.backward()
        optimizer.step()
        total_loss += loss.item()
    print("Epoch:", epoch, "Loss:", total_loss)

torch.save(model.state_dict(), "bilstm_pos_model.pth")
print("BiLSTM model saved!")

# =========================
# INDO-BERT (BATCH TRAINING)
# =========================
model_name = "indobenchmark/indobert-base-p1"
tokenizer = AutoTokenizer.from_pretrained(model_name)
model_bert = AutoModelForTokenClassification.from_pretrained(model_name, num_labels=len(tag2idx))
print("IndoBERT loaded!")

def tokenize_and_align(sentences, tags):
    # tokenize all sentences
    inputs = tokenizer(sentences, is_split_into_words=True, padding=True, truncation=True, return_tensors="pt")
    labels = []
    for i, tag_seq in enumerate(tags):
        word_ids = inputs.word_ids(batch_index=i)
        label_ids = []
        prev_word = None
        for word_id in word_ids:
            if word_id is None:
                label_ids.append(-100)
            elif word_id != prev_word:
                label_ids.append(tag2idx[tag_seq[word_id]])
            else:
                label_ids.append(tag2idx[tag_seq[word_id]])
            prev_word = word_id
        labels.append(label_ids)
    labels = torch.tensor(labels)
    return inputs, labels

train_inputs, train_labels = tokenize_and_align(train_sentences[:1000], train_tags[:1000])

# buat DataLoader untuk batching
dataset = TensorDataset(train_inputs["input_ids"], train_inputs["attention_mask"], train_labels)
batch_size = 16
loader = DataLoader(dataset, batch_size=batch_size, shuffle=True)

optimizer = AdamW(model_bert.parameters(), lr=5e-5)
model_bert.train()

for epoch in range(4):
    total_loss = 0
    for batch in loader:
        input_ids, attention_mask, labels = [b for b in batch]
        optimizer.zero_grad()
        outputs = model_bert(input_ids=input_ids, attention_mask=attention_mask, labels=labels)
        loss = outputs.loss
        loss.backward()
        optimizer.step()
        total_loss += loss.item()
    print(f"IndoBERT Epoch {epoch}, Loss: {total_loss:.4f}")
    torch.cuda.empty_cache()  # clear memory per epoch

# save IndoBERT
model_bert.save_pretrained("indobert_pos_model")
tokenizer.save_pretrained("indobert_pos_model")
print("IndoBERT model saved!")