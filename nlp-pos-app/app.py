import streamlit as st
import torch
import torch.nn as nn
import pickle
import pandas as pd
from transformers import AutoTokenizer, AutoModelForTokenClassification

# =========================
# KONFIGURASI HALAMAN
# =========================
st.set_page_config(page_title="POS Tagging: BiLSTM vs IndoBERT", layout="wide")


# =========================
# DEFINISI MODEL BiLSTM
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


# =========================
# CACHE & LOAD RESOURCES
# =========================
@st.cache_resource(show_spinner="Memuat model... (Hanya butuh waktu saat pertama kali dijalankan)")
def load_models():
    # Load Vocabularies
    with open("models/word2idx.pkl", "rb") as f:
        word2idx = pickle.load(f)
    with open("models/idx2tag.pkl", "rb") as f:
        idx2tag = pickle.load(f)

    # Load BiLSTM
    bilstm_model = BiLSTM_POS(len(word2idx), len(idx2tag))
    bilstm_model.load_state_dict(torch.load("models/bilstm_pos_model.pth", map_location=torch.device('cpu')))
    bilstm_model.eval()

    # Load IndoBERT
    tokenizer = AutoTokenizer.from_pretrained("models/indobert_pos_model")
    bert_model = AutoModelForTokenClassification.from_pretrained("models/indobert_pos_model")
    bert_model.eval()

    return word2idx, idx2tag, bilstm_model, tokenizer, bert_model


word2idx, idx2tag, bilstm_model, tokenizer, bert_model = load_models()


# =========================
# FUNGSI PREDIKSI
# =========================
def predict_bilstm(words, word2idx, idx2tag, model):
    encoded = [word2idx.get(w, 0) for w in words]  # 0 untuk unknown word
    sent_tensor = torch.tensor(encoded).unsqueeze(0)

    with torch.no_grad():
        output = model(sent_tensor)

    pred = torch.argmax(output, dim=2).squeeze().tolist()

    # Handle kasus jika cuma 1 kata (tolist() mengembalikan int, bukan list)
    if isinstance(pred, int):
        pred = [pred]

    return [idx2tag[i] for i in pred]


def predict_indobert(words, tokenizer, idx2tag, model):
    tokens = tokenizer(words, is_split_into_words=True, return_tensors="pt", truncation=True)

    with torch.no_grad():
        outputs = model(**tokens)

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

    return bert_tags


# =========================
# STREAMLIT UI
# =========================
st.title("🏷️ Part-of-Speech (POS) Tagger")
st.markdown("Bandingkan hasil prediksi kelas kata (POS Tagging) antara model **BiLSTM** dan **IndoBERT**.")

# Input Form
with st.form("pos_form"):
    test_sentence = st.text_area("Masukkan kalimat yang ingin dianalisis:",
                                 value="Aku suka makan indomie dan ayam, apalagi ayam bakar yang di jual di daerah sana",
                                 height=100)
    submitted = st.form_submit_button("Analisis Kalimat")

# Menampilkan Hasil
if submitted and test_sentence:
    # Preprocessing sederhana
    words = test_sentence.replace(",", " ,").replace(".", " .").split()

    if len(words) > 0:
        # Menjalankan Prediksi
        bilstm_tags = predict_bilstm(words, word2idx, idx2tag, bilstm_model)
        bert_tags = predict_indobert(words, tokenizer, idx2tag, bert_model)

        # Membuat DataFrame untuk tampilan tabel yang rapi
        df_results = pd.DataFrame({
            "Kata": words,
            "BiLSTM Tag": bilstm_tags,
            "IndoBERT Tag": bert_tags
        })

        st.markdown("### Hasil Prediksi:")

        # Membagi layar menjadi 2 kolom untuk tampilan perbandingan text code
        col1, col2 = st.columns(2)

        with col1:
            st.success("🤖 BiLSTM Prediction")
            # Membuat format output mirip seperti di terminal yang Anda minta
            bilstm_output = ""
            for w, t in zip(words, bilstm_tags):
                bilstm_output += f"{w:15} -> {t}\n"
            st.code(bilstm_output, language="text")

        with col2:
            st.info("🚀 IndoBERT Prediction")
            bert_output = ""
            for w, t in zip(words, bert_tags):
                bert_output += f"{w:15} -> {t}\n"
            st.code(bert_output, language="text")

        st.markdown("---")
        # Opsional: Menampilkan sebagai tabel data agar lebih modern UI-nya
        with st.expander("Lihat Hasil dalam Bentuk Tabel (Klik untuk membuka)"):
            st.dataframe(df_results, use_container_width=True)

    else:
        st.warning("Silakan masukkan kalimat terlebih dahulu.")