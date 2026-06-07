import streamlit as st
import pandas as pd
import numpy as np
import tensorflow as tf
from tensorflow.keras import layers
import joblib
import plotly.graph_objects as go
import plotly.express as px
from sklearn.metrics import classification_report, confusion_matrix
import time

# ============================================================
# KONFIGURASI HALAMAN
# ============================================================
st.set_page_config(
    page_title="NeuroMarketing EEG Analyzer",
    page_icon="🧠",
    layout="wide",
    initial_sidebar_state="expanded"
)

# ============================================================
# CUSTOM CSS
# ============================================================
st.markdown("""
<style>
    [data-testid="stHeader"] { display: none !important; }
    header[data-testid="stHeader"] { display: none !important; }
    #MainMenu { display: none !important; }
    footer { display: none !important; }
    [data-testid="stToolbar"] { display: none !important; }
    .stDeployButton { display: none !important; }
    [data-testid="stAppViewContainer"] { background-color: #0d0d1a; padding-top: 0 !important; }
    [data-testid="stSidebar"] { background-color: #12122a; border-right: 1px solid #2a2a5a; }

    .main-title {
        font-size: 2.5rem; font-weight: 800; text-align: center;
        background: linear-gradient(135deg, #00d4ff, #7b2ff7);
        -webkit-background-clip: text; -webkit-text-fill-color: transparent;
        padding: 1rem 0;
    }

    /* Metric cards */
    .metric-card {
        background: linear-gradient(135deg, #1a1a3e, #2a1a5e);
        border: 1px solid #3a3a7a; border-radius: 12px;
        padding: 1.2rem; text-align: center; margin: 0.5rem 0;
    }
    .metric-value { font-size: 2rem; font-weight: 800; color: #00d4ff; }
    .metric-label { font-size: 0.85rem; color: #b0b0cc; margin-top: 0.3rem; }
    .attention-high { color: #00ff88; font-weight: bold; }
    .attention-low  { color: #ff4466; font-weight: bold; }

    /* Insight cards — tinggi seragam */
    .insight-card {
        background: linear-gradient(135deg, #1a1a3e, #2a1a5e);
        border: 1px solid #3a3a7a; border-radius: 12px;
        padding: 1.2rem; text-align: center;
        margin: 0.5rem 0;
        min-height: 110px;
        display: flex; flex-direction: column;
        justify-content: center; align-items: center;
    }
    .insight-icon { font-size: 1.5rem; color: #00d4ff; font-weight: 800; margin-bottom: 0.4rem; }
    .insight-label { font-size: 0.85rem; color: #b0b0cc; line-height: 1.5; }

    /* Tombol — teks kontras */
    .stButton > button {
        background: linear-gradient(135deg, #7b2ff7, #00d4ff);
        color: #ffffff !important;
        border: none; border-radius: 8px;
        padding: 0.6rem 1.5rem; font-weight: 700;
        font-size: 1rem; width: 100%;
        text-shadow: 0 1px 2px rgba(0,0,0,0.5);
    }
    .stButton > button:hover {
        opacity: 0.9;
    }

    /* Download button */
    [data-testid="stDownloadButton"] > button {
        background: linear-gradient(135deg, #7b2ff7, #00d4ff);
        color: #ffffff !important;
        border: none; border-radius: 8px;
        padding: 0.6rem 1.5rem; font-weight: 700;
        font-size: 1rem; width: 100%;
        text-shadow: 0 1px 2px rgba(0,0,0,0.5);
    }
    [data-testid="stDownloadButton"] > button:hover {
        opacity: 0.9;
    }

    .info-box {
        background: #1a1a3e; border-left: 4px solid #7b2ff7;
        border-radius: 0 8px 8px 0; padding: 1rem; margin: 0.5rem 0;
        color: #ccccee; font-size: 0.9rem;
    }

    /* Heading lebih cerah */
    h1, h2, h3 { color: #ffffff !important; }
    h4 { color: #e8e8ff !important; }
    p, li { color: #c0c0dd; }

    /* Section heading khusus untuk konten utama */
    .section-heading {
        color: #ffffff !important;
        font-size: 1.2rem;
        font-weight: 700;
        margin: 1.2rem 0 0.6rem 0;
        padding-bottom: 0.3rem;
        border-bottom: 1px solid #3a3a7a;
    }

    .stTabs [data-baseweb="tab"] { color: #aaaacc; }
    .stTabs [aria-selected="true"] { color: #00d4ff !important; }

    /* Teks di dalam expander */
    [data-testid="stExpander"] p,
    [data-testid="stExpander"] li { color: #c0c0dd; }
</style>
""", unsafe_allow_html=True)

# ============================================================
# DEFINISI FUZZY LAYER
# ============================================================
class FuzzyLayer(layers.Layer):
    def __init__(self, num_rules, **kwargs):
        super().__init__(**kwargs)
        self.num_rules = num_rules

    def build(self, input_shape):
        self.mu = self.add_weight(
            name='mu', shape=(input_shape[-1], self.num_rules),
            initializer='random_normal', trainable=True
        )
        self.sigma = self.add_weight(
            name='sigma', shape=(input_shape[-1], self.num_rules),
            initializer=tf.keras.initializers.RandomUniform(minval=0.5, maxval=1.5),
            constraint=tf.keras.constraints.NonNeg(), trainable=True
        )

    def call(self, inputs):
        x = tf.expand_dims(inputs, axis=-1)
        membership = tf.exp(-tf.square((x - self.mu) / (self.sigma ** 2 + 1e-8)))
        firing_strength = tf.reduce_mean(membership, axis=1)
        return firing_strength / (tf.reduce_sum(firing_strength, axis=1, keepdims=True) + 1e-8)

    def get_config(self):
        config = super().get_config()
        config.update({'num_rules': self.num_rules})
        return config

# ============================================================
# FUNGSI HELPER
# ============================================================
TIME_STEPS = 10
EEG_CHANNELS = ['AF3', 'AF4', 'F3', 'F4', 'F7', 'F8', 'FC5', 'FC6',
                 'T7', 'T8', 'P7', 'P8', 'O1', 'O2']

@st.cache_resource
def load_model_and_scaler():
    try:
        model = tf.keras.models.load_model(
            'model_eeg.h5',
            custom_objects={'FuzzyLayer': FuzzyLayer}
        )
        scaler = joblib.load('scaler_eeg.pkl')
        return model, scaler, True
    except Exception as e:
        return None, None, False

def create_sequences_predict(data_X, time_steps=10):
    Xs = []
    for i in range(len(data_X) - time_steps):
        Xs.append(data_X[i:(i + time_steps)])
    return np.array(Xs)

def hitung_attention_per_segmen(y_pred_proba, y_pred_label, fps=128, segmen_detik=5):
    sampel_per_segmen = fps * segmen_detik
    hasil = []
    for i in range(0, len(y_pred_proba), sampel_per_segmen):
        chunk_proba = y_pred_proba[i:i + sampel_per_segmen]
        chunk_label = y_pred_label[i:i + sampel_per_segmen]
        if len(chunk_proba) == 0:
            continue
        eye_open_ratio = np.mean(chunk_label == 0)
        segmen_ke = len(hasil) + 1
        waktu_mulai = (i / fps)
        hasil.append({
            'Segmen': f'Seg {segmen_ke}',
            'Waktu Mulai (s)': round(waktu_mulai, 1),
            'Eye-Open Ratio (%)': round(eye_open_ratio * 100, 1),
            'Attention Score': round(eye_open_ratio * 100, 1),
            'Kategori': 'Tinggi' if eye_open_ratio >= 0.6 else ('Sedang' if eye_open_ratio >= 0.4 else 'Rendah')
        })
    return pd.DataFrame(hasil)

# ============================================================
# LOAD MODEL
# ============================================================
model, scaler, model_loaded = load_model_and_scaler()

# ============================================================
# SIDEBAR
# ============================================================
with st.sidebar:
    st.markdown("## NeuroMarketing EEG")
    st.markdown("---")

    status_color = "✔" if model_loaded else "X"
    status_text  = "Model Siap" if model_loaded else "Model Tidak Ditemukan"
    st.markdown(f"**Status Model:** {status_color} {status_text}")

    if not model_loaded:
        st.error("Letakkan file berikut di folder yang sama dengan app.py:\n- `model_eeg.h5`\n- `scaler_eeg.pkl`")

    st.markdown("---")
    st.markdown("### Pengaturan Analisis")
    fps = st.selectbox("Sampling Rate (Hz)", [128, 256, 512], index=0,
                       help="Frekuensi sampling perangkat EEG kamu")
    segmen_detik = st.slider("Durasi Segmen (detik)", 2, 30, 5,
                             help="Seberapa panjang tiap jendela analisis")
    threshold = st.slider("Threshold Prediksi", 0.3, 0.7, 0.5, 0.05,
                          help="Nilai di atas threshold = mata tertutup")

    st.markdown("---")
    st.markdown("### Panduan Singkat")
    st.markdown("""
    1. Upload file CSV EEG
    2. Pastikan kolom sesuai
    3. Klik **Mulai Analisis EEG**
    4. Lihat hasil di tab yang tersedia
    """)

    st.markdown("---")
    st.markdown("### Penjelasan Parameter")
    with st.expander("Durasi Segmen"):
        st.markdown("""
        Menentukan **seberapa panjang jendela waktu** untuk menghitung satu Attention Score.

        Contoh: jika diset **5 detik** @ 128 Hz, maka setiap 640 sampel EEG digabung menjadi satu segmen dan dihitung rata-rata perhatiannya.

        - **Nilai kecil (2-5 detik)** - segmen lebih banyak, analisis lebih detail per momen
        - **Nilai besar (10-30 detik)** - segmen lebih sedikit, gambaran perhatian lebih umum
        """)
    with st.expander("Threshold Prediksi"):
        st.markdown("""
        Batas keputusan model untuk mengklasifikasikan **mata terbuka atau tertutup**.

        - Nilai **di atas threshold** - mata **tertutup** (perhatian pasif)
        - Nilai **di bawah threshold** - mata **terbuka** (perhatian aktif)

        - **Threshold rendah (0.3-0.4)** - attention score cenderung lebih tinggi
        - **Threshold tinggi (0.6-0.7)** - attention score cenderung lebih rendah
        """)

    st.markdown("---")
    st.markdown("### Kelompok Neuromarketing")
    st.markdown("""
    <div style="background:#1a1a3e; border-radius:10px; padding:1rem; border:1px solid #3a3a7a;">
    <p style="color:#ccccee; font-size:0.85rem; margin:0.2rem 0;">
        - <b>Senia Nur Hasanah</b><br>
        <span style="color:#8888aa; font-size:0.8rem;">140810230021</span>
    </p>
    <p style="color:#ccccee; font-size:0.85rem; margin:0.2rem 0;">
        - <b>Audrey Shaina Tjandra</b><br>
        <span style="color:#8888aa; font-size:0.8rem;">140810230026</span>
    </p>
    <p style="color:#ccccee; font-size:0.85rem; margin:0.2rem 0;">
        - <b>Siti Nailah Eko Putri Alawiyah</b><br>
        <span style="color:#8888aa; font-size:0.8rem;">140810230059</span>
    </p>
    </div>
    """, unsafe_allow_html=True)

# ============================================================
# KONTEN UTAMA
# ============================================================
st.markdown('<div class="main-title">NeuroMarketing EEG Analyzer</div>', unsafe_allow_html=True)
st.markdown('<p style="text-align:center; color:#aaaacc;">Analisis perhatian responden berbasis sinyal EEG menggunakan CNN-LSTM-ANFIS</p>', unsafe_allow_html=True)
st.markdown("---")

tab1, tab2, tab3, tab4 = st.tabs(["Upload & Prediksi", "Dashboard Attention", "Performa Model", "Tentang Model"])

# ============================================================
# TAB 1: UPLOAD & PREDIKSI
# ============================================================
with tab1:
    st.markdown('<div class="section-heading">Upload Data EEG Responden</div>', unsafe_allow_html=True)

    col_info, col_upload = st.columns([1, 1])

    with col_info:
        st.markdown("""
        <div class="info-box">
        <b>Format CSV yang dibutuhkan:</b><br>
        - 14 kolom channel EEG: AF3, AF4, F3, F4, F7, F8, FC5, FC6, T7, T8, P7, P8, O1, O2<br>
        - Kolom terakhir bisa berisi label (opsional) atau dikosongkan<br>
        - Tidak perlu header khusus, langsung data numerik<br><br>
        <b>Contoh perangkat yang kompatibel:</b><br>
        - Emotiv EPOC / EPOC+ / EPOC X<br>
        - Dataset UCI EEG Eye State
        </div>
        """, unsafe_allow_html=True)

    with col_upload:
        uploaded_file = st.file_uploader(
            "Pilih file CSV EEG",
            type=['csv'],
            help="Upload file CSV berisi rekaman EEG responden"
        )

    if uploaded_file is not None:
        try:
            df_input = pd.read_csv(uploaded_file)

            kolom_tersedia = [c for c in EEG_CHANNELS if c in df_input.columns]

            if len(kolom_tersedia) < 14:
                if df_input.shape[1] >= 14:
                    st.warning("Nama kolom tidak cocok. Menggunakan 14 kolom pertama sebagai channel EEG.")
                    X_input = df_input.iloc[:, :14].values
                    y_true = df_input.iloc[:, 14].values if df_input.shape[1] > 14 else None
                else:
                    st.error("File CSV harus memiliki minimal 14 kolom channel EEG!")
                    st.stop()
            else:
                X_input = df_input[EEG_CHANNELS].values
                last_col = df_input.columns[-1]
                y_true = df_input[last_col].values if last_col not in EEG_CHANNELS else None

            st.success(f"Data berhasil dimuat: **{len(df_input):,} baris** x {X_input.shape[1]} channel")

            with st.expander("Preview Data (5 baris pertama)"):
                st.dataframe(df_input.head(), use_container_width=True)

            if not model_loaded:
                st.error("Model belum di-load. Cek sidebar untuk instruksi.")
            else:
                if st.button("Mulai Analisis EEG", key="analyze_btn"):
                    with st.spinner("Sedang memproses sinyal EEG..."):
                        progress_bar = st.progress(0)

                        progress_bar.progress(25, text="Menstandarisasi data...")
                        X_scaled = scaler.transform(X_input)

                        progress_bar.progress(50, text="Membuat sequences temporal...")
                        X_seq = create_sequences_predict(X_scaled, TIME_STEPS)

                        progress_bar.progress(75, text="Model sedang memprediksi...")
                        y_proba = model.predict(X_seq, verbose=0).flatten()
                        y_pred  = (y_proba > threshold).astype(int)

                        progress_bar.progress(100, text="Selesai!")
                        time.sleep(0.5)
                        progress_bar.empty()

                    st.session_state['y_proba'] = y_proba
                    st.session_state['y_pred']  = y_pred
                    st.session_state['y_true']  = y_true
                    st.session_state['n_total'] = len(y_pred)

                    st.markdown('<div class="section-heading">Ringkasan Hasil Prediksi</div>', unsafe_allow_html=True)

                    eye_open_pct  = (y_pred == 0).mean() * 100
                    eye_close_pct = (y_pred == 1).mean() * 100
                    avg_attention = eye_open_pct

                    c1, c2, c3, c4 = st.columns(4)
                    with c1:
                        st.markdown(f"""<div class="metric-card">
                            <div class="metric-value">{len(y_pred):,}</div>
                            <div class="metric-label">Total Sampel Diprediksi</div>
                        </div>""", unsafe_allow_html=True)
                    with c2:
                        st.markdown(f"""<div class="metric-card">
                            <div class="metric-value attention-high">{eye_open_pct:.1f}%</div>
                            <div class="metric-label">Mata Terbuka (Perhatian Aktif)</div>
                        </div>""", unsafe_allow_html=True)
                    with c3:
                        st.markdown(f"""<div class="metric-card">
                            <div class="metric-value attention-low">{eye_close_pct:.1f}%</div>
                            <div class="metric-label">Mata Tertutup (Perhatian Pasif)</div>
                        </div>""", unsafe_allow_html=True)
                    with c4:
                        if avg_attention >= 60:
                            kategori = "TINGGI"
                            kat_color = "#00ff88"
                        elif avg_attention >= 40:
                            kategori = "SEDANG"
                            kat_color = "#ffaa00"
                        else:
                            kategori = "RENDAH"
                            kat_color = "#ff4466"
                        st.markdown(f"""<div class="metric-card">
                            <div class="metric-value" style="color:{kat_color};">{kategori}</div>
                            <div class="metric-label">Tingkat Atensi Keseluruhan</div>
                        </div>""", unsafe_allow_html=True)

                    st.markdown('<div class="section-heading">Probabilitas Mata Tertutup (Per Timestep)</div>', unsafe_allow_html=True)

                    n_show = min(len(y_proba), 1000)
                    fig_pred = go.Figure()
                    fig_pred.add_trace(go.Scatter(
                        y=y_proba[:n_show],
                        mode='lines',
                        name='P(Mata Tertutup)',
                        line=dict(color='#00d4ff', width=1),
                        fill='tozeroy',
                        fillcolor='rgba(0,212,255,0.1)'
                    ))
                    fig_pred.add_hline(y=threshold, line_dash="dash", line_color="#ff4466",
                                      annotation_text=f"Threshold ({threshold})")
                    fig_pred.update_layout(
                        template='plotly_dark',
                        paper_bgcolor='rgba(0,0,0,0)',
                        plot_bgcolor='rgba(15,15,40,0.8)',
                        height=300,
                        margin=dict(l=0, r=0, t=30, b=0),
                        xaxis_title="Timestep",
                        yaxis_title="Probabilitas",
                        yaxis=dict(range=[0, 1])
                    )
                    st.plotly_chart(fig_pred, use_container_width=True)

                    if n_show < len(y_proba):
                        st.caption(f"Menampilkan {n_show} dari {len(y_proba):,} prediksi untuk performa rendering.")

                    st.info("Analisis selesai. Lihat tab Dashboard Attention untuk analisis neuromarketing lengkap.")

        except Exception as e:
            st.error(f"Terjadi error saat membaca file: {e}")

# ============================================================
# TAB 2: DASHBOARD ATTENTION
# ============================================================
with tab2:
    st.markdown('<div class="section-heading">Dashboard Attention Neuromarketing</div>', unsafe_allow_html=True)

    if 'y_pred' not in st.session_state:
        st.markdown("""
        <div class="info-box" style="text-align:center; padding:2rem;">
        Silakan upload data EEG dan jalankan analisis di tab <b>Upload & Prediksi</b> terlebih dahulu.
        </div>
        """, unsafe_allow_html=True)
    else:
        y_proba = st.session_state['y_proba']
        y_pred  = st.session_state['y_pred']

        df_segmen = hitung_attention_per_segmen(y_proba, y_pred, fps=fps, segmen_detik=segmen_detik)

        st.markdown(f"<p>Analisis dibagi menjadi segmen <b>{segmen_detik} detik</b> @ <b>{fps} Hz</b></p>", unsafe_allow_html=True)

        col_bar, col_pie = st.columns([2, 1])

        with col_bar:
            st.markdown('<div class="section-heading">Attention Score per Segmen Waktu</div>', unsafe_allow_html=True)
            warna_bar = ['#00ff88' if k == 'Tinggi' else ('#ffaa00' if k == 'Sedang' else '#ff4466')
                         for k in df_segmen['Kategori']]

            fig_bar = go.Figure(go.Bar(
                x=df_segmen['Segmen'],
                y=df_segmen['Attention Score'],
                marker_color=warna_bar,
                text=df_segmen['Attention Score'].astype(str) + '%',
                textposition='outside',
                textfont=dict(color='#ffffff', size=13)
            ))
            fig_bar.add_hline(y=60, line_dash="dot", line_color="#00ff88",
                              annotation_text="Batas Atensi Tinggi (60%)",
                              annotation_font_color="#00ff88")
            fig_bar.add_hline(y=40, line_dash="dot", line_color="#ff4466",
                              annotation_text="Batas Atensi Rendah (40%)",
                              annotation_font_color="#ff4466")
            fig_bar.update_layout(
                template='plotly_dark',
                paper_bgcolor='rgba(0,0,0,0)',
                plot_bgcolor='rgba(15,15,40,0.8)',
                height=350, margin=dict(l=0, r=0, t=20, b=0),
                yaxis=dict(range=[0, 110], title='Attention Score (%)', color='#ffffff'),
                xaxis=dict(title='Segmen Waktu', color='#ffffff'),
                font=dict(color='#ffffff')
            )
            st.plotly_chart(fig_bar, use_container_width=True)

        with col_pie:
            st.markdown('<div class="section-heading">Distribusi Kategori Atensi</div>', unsafe_allow_html=True)
            dist = df_segmen['Kategori'].value_counts()
            fig_pie = go.Figure(go.Pie(
                labels=dist.index,
                values=dist.values,
                marker_colors=['#00ff88', '#ffaa00', '#ff4466'],
                hole=0.4,
                textfont=dict(color='#ffffff', size=14)
            ))
            fig_pie.update_layout(
                template='plotly_dark',
                paper_bgcolor='rgba(0,0,0,0)',
                height=350, margin=dict(l=0, r=0, t=20, b=0),
                showlegend=True,
                legend=dict(font=dict(color='#ffffff'))
            )
            st.plotly_chart(fig_pie, use_container_width=True)

        st.markdown('<div class="section-heading">Heatmap Atensi (Eye-Open Ratio per Segmen)</div>', unsafe_allow_html=True)
        matrix_data = df_segmen['Attention Score'].values.reshape(1, -1)

        fig_heat = go.Figure(go.Heatmap(
            z=matrix_data,
            x=df_segmen['Segmen'],
            colorscale=[[0, '#ff4466'], [0.5, '#ffaa00'], [1, '#00ff88']],
            zmin=0, zmax=100,
            colorbar=dict(
                title=dict(text='Score (%)', font=dict(color='#ffffff')),
                tickfont=dict(color='#ffffff')
            ),
            texttemplate="%{z}%",
            textfont=dict(color='#ffffff', size=13)
        ))
        fig_heat.update_layout(
            template='plotly_dark',
            paper_bgcolor='rgba(0,0,0,0)',
            plot_bgcolor='rgba(15,15,40,0.8)',
            height=180, margin=dict(l=0, r=0, t=20, b=0),
            xaxis=dict(color='#ffffff'),
            font=dict(color='#ffffff')
        )
        st.plotly_chart(fig_heat, use_container_width=True)

        st.markdown('<div class="section-heading">Insight Neuromarketing Otomatis</div>', unsafe_allow_html=True)

        seg_terbaik  = df_segmen.loc[df_segmen['Attention Score'].idxmax()]
        seg_terburuk = df_segmen.loc[df_segmen['Attention Score'].idxmin()]
        pct_tinggi   = (df_segmen['Kategori'] == 'Tinggi').mean() * 100

        col_i1, col_i2, col_i3 = st.columns(3)
        with col_i1:
            st.markdown(f"""<div class="insight-card">
                <div class="insight-icon">Terbaik</div>
                <div class="insight-label">Segmen: <b>{seg_terbaik['Segmen']}</b><br>
                Waktu: {seg_terbaik['Waktu Mulai (s)']}s<br>
                Score: {seg_terbaik['Attention Score']}%</div>
            </div>""", unsafe_allow_html=True)
        with col_i2:
            st.markdown(f"""<div class="insight-card">
                <div class="insight-icon">Terendah</div>
                <div class="insight-label">Segmen: <b>{seg_terburuk['Segmen']}</b><br>
                Waktu: {seg_terburuk['Waktu Mulai (s)']}s<br>
                Score: {seg_terburuk['Attention Score']}%</div>
            </div>""", unsafe_allow_html=True)
        with col_i3:
            st.markdown(f"""<div class="insight-card">
                <div class="insight-icon" style="color:#00ff88;">{pct_tinggi:.0f}%</div>
                <div class="insight-label">Proporsi Segmen<br>Atensi Tinggi</div>
            </div>""", unsafe_allow_html=True)

        with st.expander("Lihat Tabel Detail Semua Segmen"):
            st.dataframe(df_segmen, use_container_width=True, hide_index=True)

        csv_export = df_segmen.to_csv(index=False).encode('utf-8')
        st.download_button(
            label="Download Hasil Analisis (CSV)",
            data=csv_export,
            file_name="hasil_attention_eeg.csv",
            mime='text/csv'
        )

# ============================================================
# TAB 3: PERFORMA MODEL
# ============================================================
with tab3:
    st.markdown('<div class="section-heading">Evaluasi Performa Model</div>', unsafe_allow_html=True)

    if 'y_pred' not in st.session_state or st.session_state['y_true'] is None:
        st.markdown("""
        <div class="info-box">
        Tab ini membutuhkan data dengan <b>label ground truth</b> (kolom terakhir berisi 0/1).<br>
        Upload file EEG yang sudah memiliki label (seperti dataset training), lalu jalankan analisis.
        </div>
        """, unsafe_allow_html=True)
    else:
        y_pred = st.session_state['y_pred']
        y_true_raw = st.session_state['y_true']

        y_true = y_true_raw[TIME_STEPS - 1: TIME_STEPS - 1 + len(y_pred)]

        if len(y_true) == len(y_pred):
            report = classification_report(y_true, y_pred,
                                           target_names=['Mata Terbuka (0)', 'Mata Tertutup (1)'],
                                           output_dict=True)
            df_report = pd.DataFrame(report).transpose().round(3)

            acc  = report['accuracy']
            prec = report['weighted avg']['precision']
            rec  = report['weighted avg']['recall']
            f1   = report['weighted avg']['f1-score']

            col_acc, col_prec, col_rec, col_f1 = st.columns(4)
            with col_acc:
                st.markdown(f"""<div class="metric-card">
                    <div class="metric-value">{acc:.1%}</div>
                    <div class="metric-label">Accuracy</div>
                </div>""", unsafe_allow_html=True)
            with col_prec:
                st.markdown(f"""<div class="metric-card">
                    <div class="metric-value">{prec:.1%}</div>
                    <div class="metric-label">Precision</div>
                </div>""", unsafe_allow_html=True)
            with col_rec:
                st.markdown(f"""<div class="metric-card">
                    <div class="metric-value">{rec:.1%}</div>
                    <div class="metric-label">Recall</div>
                </div>""", unsafe_allow_html=True)
            with col_f1:
                st.markdown(f"""<div class="metric-card">
                    <div class="metric-value">{f1:.1%}</div>
                    <div class="metric-label">F1-Score</div>
                </div>""", unsafe_allow_html=True)

            st.markdown('<div class="section-heading">Confusion Matrix</div>', unsafe_allow_html=True)
            cm = confusion_matrix(y_true, y_pred)
            fig_cm = go.Figure(go.Heatmap(
                z=cm,
                x=['Pred: Terbuka', 'Pred: Tertutup'],
                y=['True: Tertutup', 'True: Terbuka'],
                colorscale='Blues',
                text=cm, texttemplate="%{text}",
                textfont=dict(size=18, color='white')
            ))
            fig_cm.update_layout(
                template='plotly_dark',
                paper_bgcolor='rgba(0,0,0,0)',
                height=350, margin=dict(l=0, r=0, t=20, b=0),
                xaxis=dict(color='#ffffff'),
                yaxis=dict(color='#ffffff'),
                font=dict(color='#ffffff')
            )
            st.plotly_chart(fig_cm, use_container_width=True)

            with st.expander("Classification Report"):
                st.dataframe(df_report, use_container_width=True)
        else:
            st.warning("Panjang prediksi dan label tidak cocok. Pastikan data dan model sesuai.")

# ============================================================
# TAB 4: TENTANG MODEL
# ============================================================
with tab4:
    st.markdown('<div class="section-heading">Tentang Arsitektur CNN-LSTM-ANFIS</div>', unsafe_allow_html=True)

    col_arch, col_flow = st.columns([1, 1])

    with col_arch:
        st.markdown("""
        #### Komponen Model

        **CNN Block (Convolutional)**

        Mengekstrak pola spasial dari sinyal EEG antar-channel. Dua lapis Conv1D 64-filter mendeteksi pola lokal seperti spike dan gelombang.

        **LSTM Block (Long Short-Term Memory)**

        Membaca dinamika temporal — bagaimana pola EEG berubah seiring waktu. Ini penting karena sinyal otak bersifat sekuensial.

        **Projection Layer**

        Dense(32) memproyeksikan representasi LSTM ke ruang fitur yang lebih interpretable sebelum masuk ANFIS.

        **ANFIS Block (Fuzzy Inference)**

        Menggunakan 8 fuzzy rules dengan fungsi keanggotaan Gaussian untuk menangkap ketidakpastian dalam interpretasi sinyal EEG.

        **Output**

        Sigmoid menghasilkan probabilitas 0-1: nilai mendekati 1 berarti mata tertutup (perhatian pasif).
        """)

    with col_flow:
        st.markdown("""
        #### Relevansi untuk Neuromarketing

        | Aspek | Penjelasan |
        |-------|-----------|
        | **Perhatian (Attention)** | Mata terbuka = responden aktif memperhatikan stimulus |
        | **Engagement** | Proporsi eye-open selama iklan/konten ditampilkan |
        | **Segmen Kritis** | Momen dengan attention score rendah = konten perlu diperbaiki |
        | **Perbandingan** | Bandingkan attention score antar-stimulus atau antar-responden |

        #### Parameter Teknis

        | Parameter | Nilai |
        |-----------|-------|
        | Time Steps | 10 |
        | CNN Filters | 64 |
        | LSTM Units | 64 |
        | Fuzzy Rules | 8 |
        | Optimizer | Adam (lr=0.0005) |
        | Loss | Binary Crossentropy |
        | Dataset | UCI EEG Eye State (14.980 sampel) |
        """)

    st.markdown("---")
    st.markdown("""
    <div class="info-box">
    <b>Referensi Dataset:</b> Roesler, O. (2013). EEG Eye State Dataset. UCI Machine Learning Repository.<br>
    <b>Framework:</b> TensorFlow / Keras + Streamlit<br>
    <b>Channels EEG:</b> AF3, AF4, F3, F4, F7, F8, FC5, FC6, T7, T8, P7, P8, O1, O2 (Emotiv EPOC)
    </div>
    """, unsafe_allow_html=True)
