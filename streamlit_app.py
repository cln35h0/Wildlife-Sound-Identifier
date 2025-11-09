import streamlit as st
import numpy as np, pandas as pd, io, json, hashlib
import librosa, matplotlib.pyplot as plt
from pathlib import Path
try:
    from joblib import load
except Exception:
    load = None

SR = 32000
TARGET_SR = 16000
WIN_SEC = 5
N_MELS = 128
HOP_LENGTH = 512
N_FFT = 1024
MAX_SECONDS = 60 

try:
    import resampy
    HAS_RESAMPY = True
except Exception:
    HAS_RESAMPY = False

try:
    from scipy.signal import resample_poly  
    HAS_SCIPY = True
except Exception:
    HAS_SCIPY = False

def safe_resample(y, orig_sr, target_sr):
    """Resample y to target_sr using whichever backend is available."""
    if orig_sr == target_sr:
        return y, orig_sr
    if HAS_RESAMPY:
        y2 = librosa.resample(y, orig_sr=orig_sr, target_sr=target_sr)
        return y2, target_sr
    if HAS_SCIPY:
        y2 = resample_poly(y, target_sr, orig_sr)
        return y2, target_sr
    ratio = target_sr / float(orig_sr)
    idx = (np.arange(int(len(y) * ratio)) / ratio)
    i0 = np.clip(idx.astype(int), 0, len(y) - 2)
    frac = idx - i0
    y2 = (1.0 - frac) * y[i0] + frac * y[i0 + 1]
    return y2.astype(np.float32), target_sr

DEMO_SPECIES = [
    "great_tinamou","red_snouted_tree_frog","leafcutter_ant","howler_monkey",
    "rufous_motmot","spectacled_caiman","black_tamarin","glass_frog",
    "nightjar_sp","tanager_sp"
]

def detect_model_mode():
    here = Path(".")
    return (here/"model.joblib").exists() and (here/"scaler.joblib").exists() and (here/"label_list.json").exists()

def maybe_load_taxonomy():
    """Load species code → English name mapping from taxonomy.csv (BirdCLEF format)."""
    tax = Path("taxonomy.csv")
    if not tax.exists():
        return {}

    try:
        df = pd.read_csv(tax)

        possible_code_cols = ["species_id", "primary_label", "code", "ebird_code"]
        possible_name_cols = ["english_name", "common_name", "species", "scientific_name"]

        code_col = next((c for c in possible_code_cols if c in df.columns), None)
        name_col = next((c for c in possible_name_cols if c in df.columns), None)

        if code_col and name_col:
            return dict(zip(df[code_col].astype(str), df[name_col].astype(str)))

        st.warning("taxonomy.csv found but columns not recognized.")
        return {}

    except Exception as e:
        st.error(f"Failed to load taxonomy.csv: {e}")
        return {}


def pretty_label(lbl, mapping):
    return f"{mapping.get(lbl, lbl)} ({lbl})" if mapping else lbl

def to_logmel(y, sr):
    S = librosa.feature.melspectrogram(
        y=y, sr=sr, n_mels=N_MELS, n_fft=N_FFT, hop_length=HOP_LENGTH, power=2.0
    )
    S = librosa.power_to_db(S + 1e-10)
    return S

def chunk_windows(y, sr, win_sec=WIN_SEC):
    step = int(win_sec * sr)
    chunks = []
    for start in range(0, len(y), step):
        seg = y[start:start + step]
        if len(seg) < step:
            seg = np.pad(seg, (0, step - len(seg)))
        chunks.append(seg)
    return chunks

def stable_seed_from_audio(seg: np.ndarray) -> int:
    """Deterministic seed per segment for demo mode."""
    h = hashlib.sha1(seg.tobytes()).hexdigest()
    return int(h[:8], 16)

def pseudo_probs_from_features(y, sr, label_list):
    chunks = chunk_windows(y, sr, WIN_SEC)
    out = []
    for seg in chunks:
        sc  = librosa.feature.spectral_centroid(y=seg, sr=sr).mean()
        zcr = librosa.feature.zero_crossing_rate(seg).mean()
        rmse= librosa.feature.rms(y=seg).mean()
        rng = np.random.default_rng(stable_seed_from_audio(seg))
        probs = rng.random(len(label_list)).astype(np.float32)
        bias = np.array([
            (sc/5000.0),(zcr*2.0),(rmse*5.0),(sc/4000.0 + zcr),(rmse*4.0),
            (zcr*1.5),(sc/4500.0),(rmse*3.5),(zcr*1.2),(sc/4800.0 + rmse)
        ], dtype=np.float32)[:len(label_list)]
        probs = np.maximum(probs + 0.2*bias, 0)
        probs = probs / (probs.sum() + 1e-8)
        out.append(probs)
    return np.vstack(out)

def model_predict_windows(model, scaler, labels, y, sr):
    chunks = chunk_windows(y, sr, WIN_SEC)
    all_probs = []
    for seg in chunks:
        S = to_logmel(seg, sr)
        v = np.concatenate([S.mean(axis=1), S.std(axis=1)]).reshape(1, -1)
        v = scaler.transform(v)
        probs = model.predict_proba(v)[0]
        all_probs.append(probs)
    return np.vstack(all_probs)

st.title("Wildlife Sound Identifier")
st.caption("Upload rainforest audio (.ogg/.wav). The app predicts per-5s species probabilities.")

MODEL_MODE = detect_model_mode()
labels = None
name_map = maybe_load_taxonomy()

if MODEL_MODE:
    st.success("MODEL MODE: Found model.joblib, scaler.joblib, label_list.json — real predictions will be used.")
    try:
        model = load("model.joblib")
        scaler = load("scaler.joblib")
        with open("label_list.json","r") as f:
            labels = json.load(f)
    except Exception as e:
        st.error("Error loading model artifacts; falling back to DEMO MODE.")
        st.code(str(e))
        MODEL_MODE = False
else:
    st.info("DEMO MODE: No trained model found. Using deterministic pseudo-probabilities (works for viva).")

uploaded = st.file_uploader("Upload audio (.ogg or .wav)", type=["ogg", "wav"], accept_multiple_files=False)

use_sample = st.button("Use sample audio instead")
if use_sample:
    try:
        with open("sample_audio.wav", "rb") as fh:
            uploaded = io.BytesIO(fh.read())
            uploaded.name = "sample_audio.wav"
        st.success("Loaded sample_audio.wav")
    except Exception as e:
        st.error("sample_audio.wav not found. Run create_sample_audio.py once.")

top_k = st.slider("Top-K per window", 1, 10, 5)

if uploaded:
    try:
        data = uploaded.read() if hasattr(uploaded, "read") else uploaded.getvalue()

        st.audio(data, format="audio/wav")

        with st.spinner("Loading & preprocessing audio..."):
            y, sr = librosa.load(io.BytesIO(data), sr=None, mono=True)
            if not librosa.util.valid_audio(y, mono=True):
                st.error("Uploaded audio appears invalid or silent.")
                st.stop()

            if MAX_SECONDS is not None and len(y)/sr > MAX_SECONDS:
                y = y[: int(MAX_SECONDS * sr)]
                st.warning(f"Analysis capped to first {MAX_SECONDS} seconds for speed.")

            y, sr = safe_resample(y, orig_sr=sr, target_sr=TARGET_SR)

        st.write(f"Audio length: {len(y)/sr:.2f} sec  |  Sample rate: {sr} Hz  |  Window: {WIN_SEC}s")

        with st.spinner("Computing spectrogram..."):
            seg0 = y[: min(len(y), WIN_SEC*sr)]
            S0 = to_logmel(seg0, sr)
            fig = plt.figure()
            plt.imshow(S0, origin="lower", aspect="auto")
            plt.title("Log-Mel Spectrogram (first window)")
            plt.xlabel("Frames")
            plt.ylabel("Mel bins")
            st.pyplot(fig)
            plt.close(fig)

        with st.spinner("Running predictions..."):
            if MODEL_MODE:
                probs_win = model_predict_windows(model, scaler, labels, y, sr)
                label_list = labels
            else:
                probs_win = pseudo_probs_from_features(y, sr, DEMO_SPECIES)
                label_list = DEMO_SPECIES

        rows = []
        for i, p in enumerate(probs_win, start=1):
            idx = np.argsort(p)[::-1][:top_k]
            row = {"window": f"{(i-1)*WIN_SEC:>4.0f}-{i*WIN_SEC:>4.0f}s"}
            for j, k in enumerate(idx, start=1):
                row[f"#{j} species"] = pretty_label(label_list[k], name_map)
                row[f"#{j} prob"] = float(np.round(p[k], 6))
            rows.append(row)
        df = pd.DataFrame(rows)
        st.dataframe(df, use_container_width=True)

        mean_probs = probs_win.mean(axis=0)
        top_idx = np.argsort(mean_probs)[::-1][:min(top_k*2, len(label_list))]
        fig2 = plt.figure()
        plt.bar([pretty_label(label_list[i], name_map) for i in top_idx], mean_probs[top_idx])
        plt.xticks(rotation=45, ha="right")
        plt.title("Mean probability across windows (Top species)")
        st.pyplot(fig2)
        plt.close(fig2)

        csv = df.to_csv(index=False).encode("utf-8")
        st.download_button("Download per-window predictions (CSV)", data=csv, file_name="predictions.csv", mime="text/csv")

    except Exception as e:
        st.error("Failed to process audio. Details below:")
        st.code(str(e))