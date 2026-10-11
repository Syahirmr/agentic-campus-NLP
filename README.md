# Agentic Campus — Autonomous Research Topic Routing & Thesis Advisory Agent

> Proyek Tugas Mata Kuliah **Natural Language Processing (NLP)** — pemetaan otomatis judul & abstrak skripsi mahasiswa Teknik Informatika ke dalam empat domain riset utama menggunakan pendekatan **Machine Learning** dan **Deep Learning**.

**Agentic Campus** adalah sebuah *routing agent* akademik yang bertugas membaca judul dan abstrak skripsi, memahami topiknya, lalu mengarahkannya ke bidang konsentrasi dosen pembimbing yang paling relevan. Sistem ini dibangun di atas tiga modul pemrosesan berurutan: **preprocessing**, **topic modeling**, dan **klasifikasi teks**.

---

## 1. Deskripsi Proyek

Inti permasalahan yang diselesaikan: proses penentuan dosen pembimbing dan pengelompokan topik skripsi masih dilakukan secara manual, padahal jumlah dokumen skripsi terus bertambah. Proyek ini menjawabnya dengan cara membangun pipeline NLP end-to-end yang mampu:

1. **Membersihkan** teks abstrak berbahasa Indonesia yang sering kali masih mengandung *typo*, singkatan, dan *slang*.
2. **Menemukan tren makro** penelitian secara *unsupervised* melalui *topic modeling*.
3. **Mengklasifikasikan** dokumen ke dalam empat bidang konsentrasi menggunakan representasi teks klasik (TF-IDF) maupun *word embedding* (Word2Vec) yang digabungkan dengan arsitektur *deep learning* (BiLSTM).

Tujuan akhirnya: setiap skripsi — termasuk yang belum berlabel — mendapatkan prediksi bidang riset beserta skor kepercayaannya secara otomatis.

---

## 2. Informasi Dataset

| Aspek | Keterangan |
|---|---|
| **Sumber** | Kaggle — *Dataset Judul dan Abstrak Skripsi / Penelitian* |
| **Total dokumen** | **1.136** entri (kolom `Judul` + `Abstrak`) |
| **Dokumen berlabel** | **200** dokumen |
| **Dokumen tanpa label** | **936** dokumen |
| **Jumlah kelas** | **4** bidang konsentrasi riset |
| **Format** | Teks berbahasa Indonesia |

### Distribusi Kelas (Subset Berlabel, 200 Dokumen)

| Kelas | Bidang Konsentrasi Riset | Jumlah Dokumen |
|:---:|---|---:|
| 1 | Sistem Cerdas & Data Mining | 97 |
| 2 | Keamanan Informasi & Kriptografi | 14 |
| 3 | NLP & Text Mining | 46 |
| 4 | Computer Vision & Sistem Informasi | 43 |
| | **Total** | **200** |

> **Catatan:** distribusi kelas tidak seimbang (*imbalanced*) — Kelas 2 hanya diwakili 14 dokumen. Kondisi ini ditangani dengan `class_weight='balanced'` pada model Deep Learning dan LinearSVC. Sebanyak **936 dokumen tanpa label** tetap dimanfaatkan untuk melatih Word2Vec (*unsupervised*) dan sebagai target inferensi *pseudo-label routing* seluruh korpus.

---

## 3. Struktur Repositori

```text
Agentic Campus/
├── .gitignore                 # aturan pengecualian Git (contoh/, checkpoints, cache)
├── README.md                  # dokumentasi proyek (file ini)
├── requirements.txt           # daftar dependensi Python
│
├── data/
│   ├── raw/
│   │   └── data_gabungan.csv                      # korpus mentah (judul + abstrak)
│   ├── processed/
│   │   └── data_clean.csv                         # korpus hasil preprocessing
│   └── outputs/
│       ├── df_with_topics.csv                     # hasil topic modeling (LDA)
│       └── df_with_classification_predictions.csv # hasil inferensi seluruh korpus
│
├── models/
│   └── word2vec_campus.model                      # model Word2Vec hasil pelatihan
│
├── notebooks/
│   ├── 01_preprocessing.ipynb                     # Modul 01 — preprocessing & EDA
│   ├── 02_topic_modeling.ipynb                    # Modul 02 — LDA topic modeling
│   └── 03_classification_ml_dl.ipynb              # Modul 03 — klasifikasi ML & DL
│
└── contoh/                     # folder referensi dari dosen (diabaikan oleh .gitignore)
```

> Folder `contoh/` berisi notebook referensi pembelajaran dan **tidak di-push** ke GitHub (sudah tercantum di `.gitignore`).

---

## 4. Pipeline & Metodologi

Proyek dijalankan melalui **tiga modul berurutan** yang saling terhubung: output setiap modul menjadi input modul berikutnya.

### Modul 01 — Preprocessing, EDA & Subword Tokenization (`01_preprocessing.ipynb`)

Tahap ini menyiapkan korpus agar layak diproses model NLP.

- **Normalisasi typo & slang** — kamus pemetaan istilah akademik dan singkatan tidak baku (mis. *"nggak"* → *"tidak"*, *"gak"* → *"tidak"*) agar kata kunci penentu topik tidak hilang.
- **Pembersihan teks & tokenisasi formal** — *lowercasing*, penghapusan URL, angka, dan tanda baca, dilanjutkan **tokenisasi formal berbasis NLTK**.
- **Stopword removal dengan Sastrawi** — pembuangan kata umum bahasa Indonesia agar fitur lebih informatif.
- **Exploratory Data Analysis (EDA)** — visualisasi **WordCloud** dan statistik korpus untuk memahami sebaran kosakata serta kata paling dominan.
- **WordPiece Subword Tokenization** — eksplorasi tokenisasi subword berbasis BERT (`BertTokenizer`) untuk mengatasi masalah *Out-of-Vocabulary* (OOV), dilengkapi perhitungan **Compression Ratio**.

**Luaran:** `data/processed/data_clean.csv`

### Modul 02 — Topic Modeling dengan LDA (`02_topic_modeling.ipynb`)

Tahap *unsupervised learning* untuk menemukan tren makro penelitian tanpa menggunakan label.

- Pembentukan **Document-Term Matrix (DTM)** dari korpus bersih.
- Pelatihan **Latent Dirichlet Allocation (LDA)** dengan **6 topik** untuk menggali tema penelitian yang muncul secara implisit di korpus.
- Ekstraksi **10 kata kunci teratas** per topik sebagai representasi tema.
- Penentuan **topik dominan per dokumen** untuk melihat distribusi tema secara keseluruhan.

**Luaran:** `data/outputs/df_with_topics.csv`

### Modul 03 — Word Embedding & Klasifikasi (`03_classification_ml_dl.ipynb`)

Inti proyek: dua pendekatan representasi teks dibandingkan secara langsung.

**A. Representasi & Validasi Data**

- **Pembagian data ketat 80 : 10 : 10** (Train — Validasi — Test) dengan **stratifikasi kelas** dua tahap, sehingga proporsi setiap kelas terjaga di seluruh subset dan tidak terjadi *data leakage*.
- TF-IDF **hanya di-*fit* pada data train**, lalu ditransformasikan ke validasi dan test.

**B. TF-IDF + Machine Learning Klasik**

Tiga algoritma dibandingkan:

| Algoritma | Konfigurasi |
|---|---|
| **LinearSVC** | `C=0.8`, `max_iter=2000` |
| **Logistic Regression** | `C=5.0`, `solver='lbfgs'` |
| **Random Forest** | `n_estimators=200` |

Konfigurasi TF-IDF: `ngram_range=(1,2)`, `sublinear_tf=True`, `max_features=8000`, `max_df=0.90`.

**C. Word2Vec + BiLSTM (Deep Learning)**

- **Word2Vec dilatih secara *unsupervised* pada seluruh 1.136 dokumen** (termasuk yang tanpa label) dengan `vector_size=100`, `window=5`, `min_count=2`, `epochs=30`.
- Bobot Word2Vec dipakai sebagai **inisialisasi layer Embedding** (bukan inisialisasi acak) pada arsitektur **Bidirectional LSTM (BiLSTM)**, yang membaca urutan kata dari dua arah sehingga konteks kalimat akademik yang kompleks lebih tertangkap.
- Arsitektur: `Embedding → SpatialDropout1D → BiLSTM → Dense(ReLU) → Dropout → Dense(Softmax)`.
- **Early stopping** memantau `val_accuracy` untuk mencegah overfitting.

**Luaran:** `data/outputs/df_with_classification_predictions.csv` (inferensi seluruh 1.136 dokumen) dan `models/word2vec_campus.model`.

---

## 5. Performa Model (Test Set — 10%)

Evaluasi akhir dilakukan pada **test set independen** yang tidak pernah dilihat model selama pelatihan. **Macro-F1** dijadikan metrik utama karena lebih adil untuk dataset dengan distribusi kelas yang tidak seimbang.

| Model | Representasi | Test Accuracy | Macro-F1 |
|---|---|:---:|:---:|
| **LinearSVC** | TF-IDF | 0.8500 | 0.8535 |
| **Logistic Regression** | TF-IDF | 0.8500 | 0.8535 |
| **Random Forest** | TF-IDF | 0.8500 | 0.8535 |
| **Word2Vec + BiLSTM** | Word2Vec (Embedding) | 0.8000 | 0.7997 |

> **Catatan analisis:** pada test set yang kecil dan sangat *separable* ini, ketiga model TF-IDF mencapai akurasi pelatihan 1.0000 dan menghasilkan prediksi yang identik pada 20 dokumen uji — sehingga metrik mereka sama persis. Model **Word2Vec + BiLSTM** memberikan performa sedikit lebih rendah, hal yang wajar untuk dataset berlabel yang terbatas (200 dokumen), namun unggul dalam menangkap **konteks urutan kata** dan mampu menggeneralisasi ke judul skripsi baru yang belum pernah dilihat.

---

## 6. Cara Menjalankan (Quick Start)

### Prasyarat

- Python **>= 3.10**
- `pip` dan kemampuan membuat *virtual environment*

### Langkah 1 — Clone & Siapkan Environment

```bash
git clone <URL_REPOSITORI_ANDA>
cd "Agentic Campus"

# Buat virtual environment
python -m venv venv

# Aktivasi (Windows)
venv\Scripts\activate
# Aktivasi (Linux / macOS)
source venv/bin/activate
```

### Langkah 2 — Install Dependensi

```bash
pip install -r requirements.txt
```

### Langkah 3 — Jalankan Notebook Secara Berurutan

Jalankan **dari dalam folder `notebooks/`** (notebook menggunakan path relatif seperti `../data/...`):

```bash
cd notebooks
jupyter notebook
```

Lalu eksekusi **secara berurutan** (urutan ini wajib, karena setiap modul bergantung pada output modul sebelumnya):

| Urutan | Notebook | Menghasilkan |
|:---:|---|---|
| 1 | `01_preprocessing.ipynb` | `data/processed/data_clean.csv` |
| 2 | `02_topic_modeling.ipynb` | `data/outputs/df_with_topics.csv` |
| 3 | `03_classification_ml_dl.ipynb` | `models/word2vec_campus.model` + `data/outputs/df_with_classification_predictions.csv` |

> **Tips:** Notebook 01 otomatis mengunduh resource NLTK yang dibutuhkan (`nltk.download`). Proses pelatihan Word2Vec dan BiLSTM pada Modul 03 adalah tahap paling berat; gunakan environment dengan RAM yang memadai. Seluruh eksperimen dikunci dengan `SEED = 42` agar hasil dapat direproduksi.

---

## Lisensi & Atribusi

Proyek ini dibuat untuk keperluan akademik (tugas mata kuliah Natural Language Processing). Dataset bersumber dari Kaggle.
