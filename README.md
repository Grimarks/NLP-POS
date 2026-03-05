```markdown
# NLP POS Tagging

## Deskripsi Proyek
Proyek ini merupakan implementasi Part-of-Speech (POS) Tagging untuk bahasa Indonesia menggunakan dua pendekatan utama:

1. BiLSTM (Bidirectional LSTM) – model neural network sederhana untuk sequence labeling.
2. IndoBERT – model pre-trained berbasis BERT yang disesuaikan untuk token classification bahasa Indonesia.

Dataset yang digunakan berasal dari UD Indonesian-GSD, dengan preprocessing khusus untuk kebutuhan training dan evaluasi.

---

## Struktur Kode

### 1. test_POS.py
Kode ini digunakan untuk training model. Tahapan utama meliputi:

- Load dataset `.conllu` dari UD Indonesian-GSD.
- Build vocabulary kata (word2idx) dan tag (tag2idx).
- Encoding dataset menjadi format numerik untuk model.
- Training BiLSTM:
  - Menggunakan embedding size 128, hidden size 128, dan bidirectional LSTM.
  - Dataset dibatasi 1000 data training dari total ~4482 kalimat untuk mempercepat training dan mengurangi penggunaan memori, sambil tetap memberikan performa yang representatif.
- Training IndoBERT:
  - Menggunakan pre-trained indobenchmark/indobert-base-p1.
  - Dataset juga dibatasi 1000 kalimat untuk efisiensi.
  - Batch size 16, 4 epoch training.

Setelah training, model disimpan dalam folder:

- bilstm_pos_model.pth (BiLSTM)
- indobert_pos_model/ (IndoBERT)

---

### 2. POS.py
Kode ini digunakan untuk evaluasi model pada data uji (X_test.pkl dan y_test.pkl):

1. Memuat preprocessing yang telah disimpan (word2idx, idx2tag, dan data uji).
2. Membatasi data uji 300 sampel untuk evaluasi cepat.
3. Memuat BiLSTM dan melakukan prediksi.
4. Memuat IndoBERT dan melakukan prediksi token-level menggunakan tokenizer.
5. Menampilkan classification report untuk setiap model menggunakan library seqeval, termasuk precision, recall, dan F1-score.

---

## Hasil Evaluasi

### BiLSTM
- Micro F1: 0.74
- Macro F1: 0.70
- Weighted F1: 0.74

Beberapa kategori POS memiliki performa rendah (misal: DJ dan ROPN) karena keterbatasan data training dan kompleksitas bahasa.

### IndoBERT
- Micro F1: 0.89
- Macro F1: 0.87
- Weighted F1: 0.89

IndoBERT menunjukkan performa lebih baik secara keseluruhan karena memanfaatkan representasi pre-trained contextual embeddings.

---

## Alasan Pemilihan Subset Data
Dataset asli memiliki 4482 kalimat. Training seluruh data membutuhkan sumber daya yang besar dan lama. Oleh karena itu:

- Training: 1000 kalimat cukup untuk demonstrasi dan eksperimen awal.
- Testing: 300 sampel digunakan untuk evaluasi cepat dan reproducibility.

---

## Struktur Folder
```

.
├── test_POS.py          # Kode untuk training BiLSTM dan IndoBERT
├── POS.py               # Kode untuk evaluasi model
├── bilstm_pos_model.pth # Model BiLSTM hasil training
├── indobert_pos_model/  # Model IndoBERT hasil training
├── word2idx.pkl         # Preprocessing kata
├── idx2tag.pkl          # Preprocessing tag
├── X_test.pkl           # Data uji
├── y_test.pkl           # Label data uji
└── .gitignore           # File yang diabaikan git

```

---

## Catatan
- Semua file besar, seperti model IndoBERT (model.safetensors), tidak disertakan di repo untuk menghindari batasan GitHub (100MB).
- Gunakan .gitignore untuk mencegah commit file sementara atau virtual environment (venv/).
- Disarankan menggunakan GPU untuk training IndoBERT agar lebih cepat.

---

## Dependensi
- Python 3.8+
- PyTorch
- Transformers (`pip install transformers`)
- conllu (`pip install conllu`)
- seqeval (`pip install seqeval`)
```

## Pipeline POS Tagging

### 1. Load Dataset
- Dataset: UD Indonesian-GSD (`.conllu`).
- Parsing untuk mendapatkan kata (`words`) dan tag POS (`upostag`).

### 2. Build Vocabulary
- `word2idx`: mapping kata → indeks.
- `tag2idx`: mapping tag → indeks.
- `idx2tag`: mapping indeks → tag.

### 3. Encoding Data
- Kalimat diubah menjadi deretan indeks (`X`).
- Tag diubah menjadi indeks (`y`).

### 4. Save Preprocessing
File preprocessing disimpan:
- `word2idx.pkl`
- `idx2tag.pkl`
- `X_test.pkl`
- `y_test.pkl`

### 5. Training BiLSTM
- Arsitektur: embedding 128, hidden 128, LSTM bidirectional.
- Dataset training: 1000 kalimat (subset dari 4482).
- 3 epoch training, loss function: CrossEntropyLoss.
- Model disimpan sebagai `bilstm_pos_model.pth`.

### 6. Training IndoBERT
- Pre-trained model: `indobenchmark/indobert-base-p1`.
- Dataset training: 1000 kalimat untuk efisiensi memori dan waktu training.
- Tokenisasi per kata, label disesuaikan.
- DataLoader dengan batch size 16.
- Optimizer: AdamW, 4 epoch.
- Model disimpan di folder `indobert_pos_model/`.

### 7. Evaluasi
- Load preprocessing dan test data (`X_test.pkl`, `y_test.pkl`), batasi 300 sampel untuk evaluasi cepat.
- **BiLSTM**: prediksi dan hitung precision, recall, F1-score.
- **IndoBERT**: prediksi per token, hitung precision, recall, F1-score.

### 8. Hasil Perbandingan
- **BiLSTM**: weighted F1-score ~0.74.
- **IndoBERT**: weighted F1-score ~0.89.
- IndoBERT unggul pada hampir semua kategori POS.
