import streamlit as st
import torch
import torch.nn as nn
import pickle
import pandas as pd
from transformers import AutoTokenizer, AutoModelForTokenClassification
import os

# =========================
# KONFIGURASI HALAMAN
# =========================
st.set_page_config(page_title="POS Tagging: BiLSTM vs IndoBERT", layout="wide")


# =========================
# DATA PERFORMA MODEL
# (dari hasil evaluasi pada test set 300 sampel)
# Alasan kenapa tidak menggunakan X_test.pkl dan y_test.pkl untuk mengambil data performa? karena ini data fix, jadi seperti laporan
# yang tidak perlu di lakukan lagi, tidak perlu di run lagi (intinya sudah fix), dan lebih menghemat waktu
#
# =========================
MODEL_METRICS = {
    "BiLSTM": {
        "precision": 0.74,
        "recall": 0.74,
        "f1": 0.74,
        "per_class": {
            "ART": 0.91, "CONJ": 0.89, "DJ": 0.54, "DP": 0.90,
            "DV": 0.67, "ERB": 0.68, "ET": 0.89, "OUN": 0.57,
            "RON": 0.94, "ROPN": 0.48, "UM": 0.66, "UNCT": 0.99,
            "UX": 0.93, "YM": 0.50
        }
    },
    "IndoBERT": {
        "precision": 0.89,
        "recall": 0.89,
        "f1": 0.89,
        "per_class": {
            "ART": 0.95, "CONJ": 0.94, "DJ": 0.78, "DP": 0.93,
            "DV": 0.81, "ERB": 0.94, "ET": 0.94, "OUN": 0.82,
            "RON": 0.96, "ROPN": 0.79, "UM": 0.95, "UNCT": 0.99,
            "UX": 0.99, "YM": 1.00
        }
    }
}


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
    BASE_DIR = os.path.dirname(os.path.abspath(__file__))

    word2idx_path = os.path.join(BASE_DIR, "models", "word2idx.pkl")
    idx2tag_path  = os.path.join(BASE_DIR, "models", "idx2tag.pkl")
    bilstm_path   = os.path.join(BASE_DIR, "models", "bilstm_pos_model.pth")
    indobert_path = os.path.join(BASE_DIR, "models", "indobert_pos_model")

    with open(word2idx_path, "rb") as f:
        word2idx = pickle.load(f)
    with open(idx2tag_path, "rb") as f:
        idx2tag = pickle.load(f)

    bilstm_model = BiLSTM_POS(len(word2idx), len(idx2tag))
    bilstm_model.load_state_dict(torch.load(bilstm_path, map_location=torch.device("cpu")))
    bilstm_model.eval()

    tokenizer  = AutoTokenizer.from_pretrained(indobert_path)
    bert_model = AutoModelForTokenClassification.from_pretrained(indobert_path)
    bert_model.eval()

    return word2idx, idx2tag, bilstm_model, tokenizer, bert_model


word2idx, idx2tag, bilstm_model, tokenizer, bert_model = load_models()


# =========================
# FUNGSI PREDIKSI
# =========================
def predict_bilstm(words, word2idx, idx2tag, model):
    encoded = [word2idx.get(w, 0) for w in words]
    sent_tensor = torch.tensor(encoded).unsqueeze(0)

    with torch.no_grad():
        output = model(sent_tensor)

    pred = torch.argmax(output, dim=2).squeeze().tolist()
    if isinstance(pred, int):
        pred = [pred]

    return [idx2tag[i] for i in pred]


def predict_indobert(words, tokenizer, idx2tag, model):
    tokens   = tokenizer(words, is_split_into_words=True, return_tensors="pt", truncation=True)

    with torch.no_grad():
        outputs = model(**tokens)

    predictions = torch.argmax(outputs.logits, dim=2)[0]
    word_ids    = tokens.word_ids()

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
# SIDEBAR: MODEL PERFORMANCE
# =========================
with st.sidebar:
    st.header("📊 Model Performance")
    st.caption("Hasil evaluasi pada test set (300 sampel, dataset UD Indonesian-GSD)")

    st.divider()

    # --- Weighted metrics ---
    metrics = ["precision", "recall", "f1"]
    metric_labels = {"precision": "Precision", "recall": "Recall", "f1": "F1-Score"}

    for metric in metrics:
        st.markdown(f"**{metric_labels[metric]}**")
        col_b, col_i = st.columns(2)

        bilstm_val  = MODEL_METRICS["BiLSTM"][metric]
        indobert_val = MODEL_METRICS["IndoBERT"][metric]

        with col_b:
            st.metric(label="BiLSTM",   value=f"{bilstm_val:.2f}")
        with col_i:
            st.metric(label="IndoBERT", value=f"{indobert_val:.2f}",
                      delta=f"+{indobert_val - bilstm_val:.2f} vs BiLSTM")

    st.divider()

    # --- Per-class F1 bar chart ---
    st.markdown("**F1-Score per Kelas POS**")

    classes     = list(MODEL_METRICS["BiLSTM"]["per_class"].keys())
    bilstm_f1   = list(MODEL_METRICS["BiLSTM"]["per_class"].values())
    indobert_f1 = list(MODEL_METRICS["IndoBERT"]["per_class"].values())

    df_chart = pd.DataFrame({
        "Kelas": classes,
        "BiLSTM": bilstm_f1,
        "IndoBERT": indobert_f1
    }).set_index("Kelas")

    st.bar_chart(df_chart, color=["blue", "green"])

    st.caption("BiLSTM  IndoBERT")

    st.divider()
    st.info(
        "**IndoBERT** unggul signifikan terutama pada kelas **ROPN** (nama diri), "
        "**ERB** (kata berulang), dan **OUN** (kata benda) — kategori yang membutuhkan "
        "pemahaman konteks lebih luas."
    )


# =========================
# MAIN UI
# =========================
st.title("Part-of-Speech (POS) Tagger")
st.markdown(
    "Bandingkan hasil prediksi kelas kata (POS Tagging) antara model **BiLSTM** dan **IndoBERT**. "
    "Baris yang **disorot merah** menunjukkan kata di mana kedua model memberikan prediksi yang **berbeda**."
)

# Input Form
with st.form("pos_form"):
    test_sentence = st.text_area(
        "Masukkan kalimat yang ingin dianalisis:",
        value="Aku suka makan indomie dan ayam, apalagi ayam bakar yang di jual di daerah sana",
        height=100
    )
    submitted = st.form_submit_button("Analisis Kalimat")

# =========================
# HASIL PREDIKSI
# =========================
if submitted and test_sentence:
    words = test_sentence.replace(",", " ,").replace(".", " .").split()

    if len(words) == 0:
        st.warning("Silakan masukkan kalimat terlebih dahulu.")
    else:
        bilstm_tags = predict_bilstm(words, word2idx, idx2tag, bilstm_model)
        bert_tags   = predict_indobert(words, tokenizer, idx2tag, bert_model)

        # --- Tampilan kode berdampingan (format lama, tetap ada) ---
        st.markdown("### Hasil Prediksi")
        col1, col2 = st.columns(2)

        with col1:
            st.success("BiLSTM Prediction")
            bilstm_output = ""
            for w, t in zip(words, bilstm_tags):
                bilstm_output += f"{w:15} -> {t}\n"
            st.code(bilstm_output, language="text")

        with col2:
            st.info("IndoBERT Prediction")
            bert_output = ""
            for w, t in zip(words, bert_tags):
                bert_output += f"{w:15} -> {t}\n"
            st.code(bert_output, language="text")

        st.divider()

        # --- Tabel perbandingan dengan highlight perbedaan ---
        st.markdown("### Tabel Perbandingan Detail")

        different_count = sum(1 for b, i in zip(bilstm_tags, bert_tags) if b != i)
        total_words     = len(words)
        agree_pct       = (total_words - different_count) / total_words * 100

        m1, m2, m3 = st.columns(3)
        m1.metric("Total Kata",       total_words)
        m2.metric("Prediksi Berbeda", different_count,
                  delta=f"{different_count} kata", delta_color="inverse")
        m3.metric("Tingkat Kesepakatan", f"{agree_pct:.0f}%")

        st.caption("Baris merah = kedua model berbeda prediksi")

        df_results = pd.DataFrame({
            "Kata":        words,
            "BiLSTM Tag":  bilstm_tags,
            "IndoBERT Tag": bert_tags,
            "Sama?":       ["✅" if b == i else "❌" for b, i in zip(bilstm_tags, bert_tags)]
        })

        def highlight_diff(row):
            """Warnai baris jika BiLSTM dan IndoBERT berbeda."""
            if row["Sama?"] == "❌":
                return ["background-color: #ffe0e0; color: #8b0000"] * len(row)
            return [""] * len(row)

        styled_df = df_results.style.apply(highlight_diff, axis=1)
        st.dataframe(styled_df, use_container_width=True, hide_index=True)

        # --- Penjelasan perbedaan ---
        diff_rows = [(w, b, i) for w, b, i in zip(words, bilstm_tags, bert_tags) if b != i]
        if diff_rows:
            with st.expander(f"📌 Analisis {len(diff_rows)} Perbedaan Prediksi"):
                for word, b_tag, i_tag in diff_rows:
                    b_f1 = MODEL_METRICS["BiLSTM"]["per_class"].get(b_tag, None)
                    i_f1 = MODEL_METRICS["IndoBERT"]["per_class"].get(i_tag, None)

                    b_f1_str = f"(F1 kelas {b_tag}: {b_f1:.2f})" if b_f1 else ""
                    i_f1_str = f"(F1 kelas {i_tag}: {i_f1:.2f})" if i_f1 else ""

                    st.markdown(
                        f"- **`{word}`** → BiLSTM: `{b_tag}` {b_f1_str} &nbsp;|&nbsp; "
                        f"IndoBERT: `{i_tag}` {i_f1_str}"
                    )
        else:
            st.success("Kedua model sepakat pada semua kata!")