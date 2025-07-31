import streamlit as st
import string
import os
import re
import PyPDF2
import nltk
from nltk.tokenize import sent_tokenize, word_tokenize
from Sastrawi.Stemmer.StemmerFactory import StemmerFactory
from nltk.corpus import stopwords
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import networkx as nx
import numpy as np
import matplotlib.pyplot as plt
from rouge_score import rouge_scorer
import pandas as pd
import seaborn as sns

# =======================
# FUNGSI SETUP NLTK DATA 
# =======================
@st.cache_resource
def setup_nltk_data():
    try:
        # Cek ketersediaan data NLTK
        nltk.data.find('tokenizers/punkt')
        nltk.data.find('corpora/stopwords')
        st.info("NLTK data sudah terunduh.")
    except nltk.downloader.DownloadError:
        with st.spinner("Mengunduh NLTK data..."):
            # Unduh
            nltk.download('punkt', quiet=True)
            nltk.download('stopwords', quiet=True)
            st.success("NLTK data berhasil diunduh!")
    return True

setup_nltk_data()

# Konfigurasi halaman Streamlit
st.set_page_config(
    page_title="Peringkasan Teks Otomatis dengan TextRank",
    layout="wide"
)

st.title("Peringkasan Teks Otomatis Putusan Pengadilan")
st.markdown("Aplikasi demo untuk meringkas dokumen PDF berbahasa Indonesia menggunakan algoritma TextRank.")

# =========================================================
# FUNGSI-FUNGSI UTAMA (DI-CACHE UNTUK EFISIENSI)
# =========================================================

@st.cache_data
def read_pdf(uploaded_file):
    text = ""
    try:
        with uploaded_file as file:
            pdf_reader = PyPDF2.PdfReader(file)
            for page in pdf_reader.pages:
                text += page.extract_text() + "\n"
    except Exception as e:
        st.error(f"Error membaca PDF: {e}")
    return text

@st.cache_data
def read_txt(uploaded_file):
    try:
        return uploaded_file.read().decode("utf-8")
    except Exception as e:
        st.error(f"Error membaca file TXT: {e}")
        return ""

@st.cache_data
def normalize_abbreviations(text):
    text = re.sub(r'([a-zA-Z])\.', r'\1<DOT>', text)
    return text

@st.cache_data
def clean_text_and_segment(text):
    watermark_phrases = [
        r'Mahkamah\s+Agung\s+Republik\s+Indonesia',
        r'Mahkamah\s+Agung',
        r'putusan\s+mahkamahlegung\.go',
        r'SINTECH Journal \| \d+',
        r'Direktori Putusan Mahkamah Agung Republik Indonesia',
        r'putusan\s+mahkamah\s+agung\.go\.id',
        r'demi keadilan berdasarkan ketuhanan yang maha esa',
        r'-{2,}',
        r'\_+',
        r'\n'
    ]
    for phrase in watermark_phrases:
        text = re.sub(phrase, ' ', text, flags=re.IGNORECASE)

    text = normalize_abbreviations(text)
    
    # PERBAIKAN: Gunakan model bahasa 'indonesian' untuk tokenisasi
    sentences = sent_tokenize(text, language='indonesian')
    
    sentences = [re.sub(r'<DOT>', '.', s) for s in sentences]
    sentences = [re.sub(r'\s+', ' ', s).strip() for s in sentences]
    sentences = [s for s in sentences if len(s.split()) > 3]

    return sentences

@st.cache_data
def process_sentences(sentences):
    stemmer = StemmerFactory().create_stemmer()
    indo_stopwords = set(stopwords.words('indonesian'))
    
    processed_sentences = []
    
    for sent in sentences:
        tokens = word_tokenize(sent.lower())
        
        stemmed_tokens = [
            stemmer.stem(token) for token in tokens 
            if token not in indo_stopwords and token not in string.punctuation and len(token) > 2
        ]
        
        processed_sentences.append(" ".join(stemmed_tokens))
    
    return processed_sentences

@st.cache_data
def calculate_tfidf(processed_sentences):
    tfidf_vectorizer = TfidfVectorizer(ngram_range=(1, 2))
    tfidf_matrix = tfidf_vectorizer.fit_transform(processed_sentences)
    feature_names = tfidf_vectorizer.get_feature_names_out()
    return tfidf_matrix, feature_names

@st.cache_data
def textrank_algorithm(similarity_matrix):
    n_sentences = similarity_matrix.shape[0]
    graph = nx.Graph()
    graph.add_nodes_from(range(n_sentences))
    
    non_diag_sim = similarity_matrix[~np.eye(n_sentences, dtype=bool)]
    
    if non_diag_sim.size > 0:
        threshold = np.median(non_diag_sim)
    else:
        threshold = 0.0

    for i in range(n_sentences):
        for j in range(i + 1, n_sentences):
            if similarity_matrix[i][j] > threshold:
                graph.add_edge(i, j, weight=similarity_matrix[i][j])
    
    try:
        if graph.number_of_nodes() == 0:
            pagerank_scores = {}
        else:
            pagerank_scores = nx.pagerank(graph, alpha=0.85, max_iter=100, tol=1e-6)
    except nx.exceptions.PowerIterationFailedConvergence:
        st.warning("Pagerank tidak konvergen. Menggunakan bobot yang sama.")
        pagerank_scores = {i: 1.0/n_sentences for i in range(n_sentences)}
    except Exception as e:
        st.error(f"Error saat menjalankan TextRank: {e}. Menggunakan bobot yang sama.")
        pagerank_scores = {i: 1.0/n_sentences for i in range(n_sentences)}

    return pagerank_scores, graph

@st.cache_data
def calculate_rouge_scores(summary, reference_summary):
    scorer = rouge_scorer.RougeScorer(['rouge1', 'rouge2', 'rougeL'], use_stemmer=True)

    def normalize_text(text):
        text = re.sub(r'\s+', ' ', text.lower().strip())
        return text

    if not summary or not reference_summary:
        return None

    normalized_summary = normalize_text(summary)
    normalized_ref = normalize_text(reference_summary)
    
    scores = scorer.score(normalized_ref, normalized_summary)
    
    return scores

# =========================================================
# ANTARMUKA PENGGUNA (UI)
# =========================================================

with st.sidebar:
    st.header("Pengaturan Ringkasan")
    uploaded_pdf = st.file_uploader(
        "Unggah dokumen PDF putusan pengadilan",
        type="pdf",
        help="Hanya file PDF yang diizinkan."
    )
    
    compression_rate_percent = st.slider(
        "Tingkat Kompresi (%)",
        min_value=10, max_value=90, value=75, step=5
    )
    
    uploaded_reference_txt = st.file_uploader(
        "Unggah file .txt Ringkasan Referensi (Opsional)",
        type="txt",
        help="Unggah file teks (.txt) yang berisi ringkasan manual untuk evaluasi ROUGE."
    )
    
    summarize_button = st.button("Ringkas Teks", use_container_width=True)

if uploaded_pdf and summarize_button:
    with st.spinner("Memproses dokumen..."):
        original_text = read_pdf(uploaded_pdf)
        
        if not original_text.strip():
            st.error("Dokumen PDF kosong atau tidak bisa diekstrak.")
            st.stop()

        sentences = clean_text_and_segment(original_text)
        processed_sentences = process_sentences(sentences)
        
        if not processed_sentences:
            st.error("Tidak ada kalimat yang valid setelah preprocessing. Coba dokumen lain atau sesuaikan filter.")
            st.stop()

        tfidf_matrix, feature_names = calculate_tfidf(processed_sentences)
        
        if tfidf_matrix.shape[0] == 0 or tfidf_matrix.shape[1] == 0:
            st.error("Tidak ada fitur TF-IDF yang dihasilkan. Coba dokumen lain atau sesuaikan preprocessing.")
            st.stop()

        similarity_matrix = cosine_similarity(tfidf_matrix)

        pagerank_scores, graph = textrank_algorithm(similarity_matrix)
        
        if not pagerank_scores:
            st.error("TextRank tidak dapat menghitung skor. Coba dokumen lain.")
            st.stop()

        ranked_sentences = sorted(pagerank_scores.items(), key=lambda x: x[1], reverse=True)

        summary_ratio = (100 - compression_rate_percent) / 100
        n_summary_sentences = max(1, int(len(sentences) * summary_ratio))
        
        top_sentences_idx = [idx for idx, _ in ranked_sentences[:n_summary_sentences]]
        top_sentences_idx.sort()
        
        summary_sentences = [sentences[idx] for idx in top_sentences_idx]
        summary = ' '.join(summary_sentences)
        
    st.success("Ringkasan berhasil dibuat!")

    st.subheader("📊 Metrik Ringkasan")
    col1, col2, col3 = st.columns(3)
    with col1:
        st.metric("Jumlah Kalimat Asli", len(sentences))
    with col2:
        st.metric("Kalimat dalam Ringkasan", n_summary_sentences)
    with col3:
        st.metric("Tingkat Kompresi", f"{compression_rate_percent}%")

    st.subheader("📄 Hasil Ringkasan")
    st.text_area(
        "Ringkasan Teks Otomatis",
        summary,
        height=300
    )
    
    st.subheader("📝 Teks Asli")
    st.text_area(
        "Teks Dokumen Asli",
        original_text,
        height=300
    )

    if uploaded_reference_txt:
        reference_summary_text = read_txt(uploaded_reference_txt)
        if reference_summary_text.strip() and summary.strip():
            st.subheader("🎯 Hasil Evaluasi ROUGE")
            with st.spinner("Menghitung skor ROUGE..."):
                rouge_results = calculate_rouge_scores(summary, reference_summary_text)

            if rouge_results and 'rouge1' in rouge_results:
                rouge_df = pd.DataFrame({
                    'Metric': ['ROUGE-1', 'ROUGE-2', 'ROUGE-L'],
                    'Precision': [rouge_results['rouge1'].precision, rouge_results['rouge2'].precision, rouge_results['rougeL'].precision],
                    'Recall': [rouge_results['rouge1'].recall, rouge_results['rouge2'].recall, rouge_results['rougeL'].recall],
                    'F1-Score': [rouge_results['rouge1'].fmeasure, rouge_results['rouge2'].fmeasure, rouge_results['rougeL'].fmeasure]
                })
                st.dataframe(rouge_df.set_index('Metric'), use_container_width=True)

                metrics_list = ['rouge1', 'rouge2', 'rougeL']
                precision_scores = [rouge_results[m].precision for m in metrics_list]
                recall_scores = [rouge_results[m].recall for m in metrics_list]
                f1_scores = [rouge_results[m].fmeasure for m in metrics_list]
                
                fig, ax = plt.subplots(figsize=(10, 5))
                x = np.arange(len(metrics_list))
                width = 0.25

                ax.bar(x - width, precision_scores, width, label='Precision', alpha=0.8)
                ax.bar(x, recall_scores, width, label='Recall', alpha=0.8)
                ax.bar(x + width, f1_scores, width, label='F1-Score', alpha=0.8)

                ax.set_xlabel('ROUGE Metrics')
                ax.set_ylabel('Score')
                ax.set_title('ROUGE Evaluation Scores')
                ax.set_xticks(x)
                ax.set_xticklabels([m.upper() for m in metrics_list])
                ax.legend()
                ax.grid(True, alpha=0.3)
                plt.tight_layout()
                st.pyplot(fig)
            else:
                st.warning("Tidak dapat menghitung skor ROUGE. Pastikan ringkasan otomatis dan referensi tidak kosong.")
        else:
            st.info("Silakan unggah file .txt Ringkasan Referensi di sidebar untuk melihat hasil evaluasi ROUGE.")


    with st.expander("Lihat Detail Dokumen & Proses"):
        st.subheader("Visualisasi Kata Kunci (TF-IDF)")
        if len(feature_names) > 0:
            feature_scores = np.asarray(tfidf_matrix.sum(axis=0)).flatten()
            top_features_idx = feature_scores.argsort()[-20:][::-1]
            top_features_scores = feature_scores[top_features_idx]
            top_features_names = [feature_names[idx] for idx in top_features_idx]
            
            fig_tfidf, ax_tfidf = plt.subplots(figsize=(10, 8))
            ax_tfidf.barh(range(len(top_features_names)), top_features_scores, color='skyblue')
            ax_tfidf.set_yticks(range(len(top_features_names)))
            ax_tfidf.set_yticklabels(top_features_names)
            ax_tfidf.set_xlabel("Skor TF-IDF")
            ax_tfidf.set_title("20 Kata Kunci Teratas Berdasarkan TF-IDF")
            ax_tfidf.invert_yaxis()
            fig_tfidf.tight_layout()
            st.pyplot(fig_tfidf)
        else:
            st.info("Tidak ada fitur TF-IDF untuk divisualisasikan.")
        
        st.subheader("Visualisasi Matriks Kemiripan (Cosine Similarity)")
        if similarity_matrix.size > 0 and len(sentences) > 1:
            fig_sim, ax_sim = plt.subplots(figsize=(10, 8))
            sns.heatmap(similarity_matrix, cmap='viridis', ax=ax_sim)
            ax_sim.set_title("Heatmap Matriks Kemiripan Antar Kalimat")
            ax_sim.set_xlabel("Indeks Kalimat")
            ax_sim.set_ylabel("Indeks Kalimat")
            st.pyplot(fig_sim)
        else:
            st.info("Tidak ada matriks kemiripan untuk divisualisasikan.")

        st.subheader("Visualisasi Graf TextRank")
        if graph.number_of_nodes() > 0 and graph.number_of_edges() > 0:
            fig_graph, ax_graph = plt.subplots(figsize=(12, 10))
            node_sizes = [v * 50000 for v in pagerank_scores.values()]
            pos = nx.spring_layout(graph, k=0.5, iterations=50)
            nx.draw_networkx_nodes(graph, pos, node_size=node_sizes, node_color='skyblue', ax=ax_graph)
            nx.draw_networkx_edges(graph, pos, width=1.0, alpha=0.5, ax=ax_graph)
            nx.draw_networkx_labels(graph, pos, font_size=8, font_color='black', ax=ax_graph)
            ax_graph.set_title("Graf Kalimat Berdasarkan Skor TextRank")
            ax_graph.axis('off')
            st.pyplot(fig_graph)
        else:
            st.info("Graf tidak dapat divisualisasikan. Mungkin dokumen terlalu pendek.")

        st.subheader("Urutan Kalimat Setelah Preprocessing")
        for i, s in enumerate(sentences):
            st.text(f"{i+1}. {s}")
        
        st.subheader("Kalimat Setelah Tokenisasi, Stemming & Stopwords Removal")
        for i, s in enumerate(processed_sentences):
            st.text(f"{i+1}. {s}")

if not uploaded_pdf:
    st.info("Silakan unggah file PDF untuk memulai proses peringkasan.")
