"""RACL demo front end (Streamlit).   Run:  streamlit run app.py

Shows what the code does so it can be presented in a browser. It does NOT show real results:
- the degradation page applies the real degradation functions to a clip you upload (or a built-in test clip);
- the pilot page trains the five variants on SYNTHETIC features (made-up numbers), only to prove the pipeline runs;
- real results need FakeAVCeleb, real encoders (D-02, D-15) and the supervisor decisions in Guide section 16.
"""
import subprocess
import sys
import tempfile
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import streamlit as st
import yaml

ROOT = Path(__file__).parent
sys.path.insert(0, str(ROOT))
from views import degrade as D, desync as DS
from views.codecs import SR

VARIANTS = ["C", "D0", "D1a", "D1b", "D1"]
st.set_page_config(page_title="RACL demo", layout="wide")


def banner(text="SYNTHETIC / DEMO ONLY: this page shows that the pipeline runs. It is not a research result."):
    st.warning(text)


# ------------------------------------------------------------------ page 1
def page_overview():
    st.title("RACL: robustness of audio-visual deepfake detection")
    st.write("A windowed cross-modal detector on frozen encoder features. The aim is to keep the false-positive rate on "
             "**real** clips low when audio or video is degraded (DR-FPR, lower is better).")
    st.subheader("Pipeline")
    st.graphviz_chart("""digraph { rankdir=TB; node [shape=box, style=rounded];
      "FakeAVCeleb\\nvideos" -> "split by identity" -> "2 s clips" -> "degraded / desync views" -> "frozen encoders\\n(cached features)"
      -> "RACL model\\n(windowed cross-attention)" -> "threshold on clean val" -> "evaluate: clean + 3 held-out conditions"
      -> "bootstrap over videos" -> "tables + plots" }""")
    st.subheader("The five variants (they differ only in three flags)")
    rows = []
    for v in VARIANTS:
        f = yaml.safe_load(open(ROOT / "configs" / f"{v}.yaml"))["flags"]; rows.append({"variant": v, **f})
    st.dataframe(pd.DataFrame(rows), hide_index=True)
    st.subheader("Status")
    st.table(pd.DataFrame([
        ["Code for data split, views, model, losses, evaluation, report", "written and tested on synthetic data"],
        ["Real data (FakeAVCeleb)", "access requested, not received yet"],
        ["Real encoders (D-02, D-15) and margins (D-06)", "waiting for supervisor decisions"],
        ["Real results", "none yet"]], columns=["item", "status"]))


# ------------------------------------------------------------------ page 2
def test_clip(seconds=2):
    """Built-in test clip: moving pattern + tone (frames [50,H,W,3] uint8, 16 kHz audio)."""
    rng = np.random.default_rng(0); n = 50; h = w = 128
    yy, xx = np.mgrid[0:h, 0:w]
    frames = np.stack([np.stack([(128 + 100 * np.sin((xx + 4 * t) / 12)) , (128 + 100 * np.sin((yy - 3 * t) / 10)),
                                 np.full((h, w), 100 + 2 * t)], -1) for t in range(n)]).clip(0, 255).astype(np.uint8)
    t = np.arange(SR * seconds) / SR
    wav = 0.3 * np.sin(2 * np.pi * 220 * t) * (1 + np.sin(2 * np.pi * 3 * t)) / 2 + 0.01 * rng.standard_normal(len(t))
    return frames, wav.astype(np.float32)


def load_clip(up):
    if up is None:
        return test_clip(), "built-in test clip"
    from data import make_clips as MC
    with tempfile.NamedTemporaryFile(suffix=Path(up.name).suffix, delete=False) as f:
        f.write(up.read()); path = f.name
    frames, wav = MC.read_frames(path), MC.read_audio(path)
    if len(frames) < 50 or len(wav) < 2 * SR:
        st.error("The video needs at least 2 seconds of picture and sound."); st.stop()
    # centre crop = placeholder for the face / mouth crop (D-02)
    return (MC.center_crop(frames[:50], 128), wav[:2 * SR]), "your upload (centre crop is a placeholder for the face crop, D-02)"


def wave_plot(series: dict):
    fig, ax = plt.subplots(len(series), 1, figsize=(7, 1.4 * len(series)), sharex=True)
    ax = np.atleast_1d(ax)
    for a, (k, x) in zip(ax, series.items()):
        a.plot(np.arange(len(x)) / SR, x, lw=0.5); a.set_ylabel(k, rotation=0, ha="right", fontsize=8); a.set_yticks([])
    ax[-1].set_xlabel("seconds"); fig.tight_layout(); return fig


def page_views():
    st.title("What a degradation does to a clip")
    st.write("These use the real functions from `views/degrade.py` and `views/desync.py`. Strength values here are only "
             "for looking at them; the training ranges are still subject to approval (D-14).")
    up = st.file_uploader("Upload a short video (mp4, at least 2 s) or leave empty to use the built-in test clip", type=["mp4", "mov", "avi", "mkv"])
    (frames, wav), src = load_clip(up); st.caption(f"Using: {src}")
    rng = np.random.default_rng(0)
    left, right = st.columns(2)
    with left:
        st.subheader("Video")
        kind = st.selectbox("Video degradation", ["none", "gaussian blur", "motion blur"])
        strength = st.slider("blur strength (sigma / length)", 0.5, 8.0, 2.0, key="vs")
        vd = frames if kind == "none" else (D.gaussian_blur(frames, strength) if kind == "gaussian blur"
                                           else D.motion_blur(frames, int(max(2, strength * 2)), 0))
        c1, c2 = st.columns(2); c1.image(frames[25], caption="clean"); c2.image(vd[25], caption=kind)
    with right:
        st.subheader("Audio")
        kind_a = st.selectbox("Audio degradation", ["none", "noise (SNR)", "reverb", "desync: shift audio"])
        wd = wav
        if kind_a == "noise (SNR)":
            wd = D.add_noise_snr(wav, st.slider("SNR in dB (lower = noisier)", 0, 30, 10), rng)
        elif kind_a == "reverb":
            wd = D.reverb(wav, D.synthetic_rir(SR, st.slider("reverb time RT60 (s)", 0.1, 1.0, 0.4), rng))
        elif kind_a == "desync: shift audio":
            ms = st.slider("shift in ms (the supervisor must still fix the range, D-13)", 40, 500, 200)
            shift = int(SR * ms / 1000); wd = np.roll(wav, shift)
        st.write("clean"); st.audio(wav, sample_rate=SR)
        st.write(kind_a); st.audio(np.clip(wd, -1, 1).astype(np.float32), sample_rate=SR)
        st.pyplot(wave_plot({"clean": wav, kind_a: wd}))
    st.info("A detector should keep calling a REAL clip real even after these changes; that false alarm rate is DR-FPR.")


# ------------------------------------------------------------------ page 3
@st.cache_data(show_spinner=False)
def run_synthetic_pilot(epochs: int):
    import pilot_synthetic as P
    import make_report as R
    out = Path(tempfile.mkdtemp(prefix="racl_demo_"))
    clips, cache = P.build_world(out, n_identities=30, min_real=3)
    for v in VARIANTS:
        P.run_variant(v, clips, cache, out, epochs=epochs, seed=0)
    ci, verdict = R.make_report(out / "results", out / "summary", VARIANTS, (0,), n_boot=200)
    return str(out / "summary"), verdict


def page_pilot():
    st.title("Synthetic pilot: does the whole pipeline run?")
    banner("The features here are RANDOM NUMBERS made up by the code. The numbers and plots below are NOT results.")
    epochs = st.slider("training epochs per variant", 1, 5, 2)
    if st.button("Run the pilot (5 variants, about 1 minute)"):
        with st.spinner("training 5 variants, evaluating, bootstrapping, plotting..."):
            st.session_state["pilot"] = run_synthetic_pilot(epochs)
    if "pilot" not in st.session_state:
        st.stop()
    summ, verdict = st.session_state["pilot"]; summ = Path(summ)
    st.success("Every step ran: training, threshold, evaluation on 4 conditions, bootstrap intervals, tables and plots.")
    st.write("Files written:", ", ".join(sorted(p.name for p in summ.iterdir())))
    t1, t2, t3 = st.tabs(["metrics", "bootstrap intervals", "plots"])
    with t1:
        st.dataframe(pd.read_csv(summ / "metrics_mean_std.csv"))
    with t2:
        st.caption("Margins are unset (D-06), so labels read 'unset (D-06)'.")
        st.dataframe(pd.read_csv(summ / "bootstrap_ci.csv"))
        st.json(verdict)
    with t3:
        for p in sorted(summ.glob("*.png")):
            st.image(str(p), caption=p.name, width=520)


# ------------------------------------------------------------------ page 5
def list_runs(results_dir):
    """Finished runs (best.pt and threshold.json present), e.g. results/D1/seed0."""
    return sorted(p.parent for p in Path(results_dir).glob("*/seed*/threshold.json") if (p.parent / "best.pt").exists())


def demo_videos(manifest_csv, root):
    """{label: (path, split, truth)} for the videos listed in videos_split.csv; empty if the file is missing."""
    f = Path(manifest_csv)
    if not f.exists():
        return {}
    m = pd.read_csv(f); out = {}
    for r in m.itertuples():
        out[f"{r.split}: {Path(r.path).parent.name}/{Path(r.path).name}  (truth: {'fake' if r.label else 'real'})"] = (
            Path(root) / r.path, r.split, "fake" if r.label else "real")
    return dict(sorted(out.items()))


@st.cache_resource(show_spinner="loading encoders (first time downloads Wav2Vec 2.0 and ResNet-18)...")
def get_encoders():
    import os
    if os.environ.get("RACL_ENCODERS") == "standin":                       # used only by the automatic tests
        from features import standins
        return standins.video_encoder, standins.audio_encoder
    from features.real_encoders import load_audio_encoder, load_video_encoder
    return load_video_encoder(), load_audio_encoder()


def page_predict():
    import os
    from predict_video import predict_video
    st.title("Predict: is this video fake?")
    banner("PILOT MODEL trained on a handful of videos, with a placeholder video encoder (ResNet-18, not a lip encoder) and a "
           "placeholder face crop. Its answers can be wrong and are NOT a measured accuracy.")
    runs = list_runs(os.environ.get("RACL_RESULTS", ROOT / "results"))
    if not runs:
        st.info("No trained run found. Run `python run_all.py --max_shift_ms 200 --batch_size 8 --seeds 0` first."); st.stop()
    names = {f"{r.parent.name} / {r.name}": r for r in runs}
    pick = st.selectbox("Trained run", list(names), index=list(names).index("D1 / seed0") if "D1 / seed0" in names else 0)
    run = names[pick]
    demos = demo_videos(ROOT / "videos_split.csv", ROOT)
    src = st.radio("Video", ["a demo video from demo_videos", "upload my own"], horizontal=True)
    truth = split = None
    if src.startswith("a demo") and demos:
        label = st.selectbox("Which video", list(demos)); path, split, truth = demos[label]
    else:
        up = st.file_uploader("Upload a video (at least 2 s, with sound)", type=["mp4", "mov", "avi", "mkv"])
        if up is None:
            st.stop()
        with tempfile.NamedTemporaryFile(suffix=Path(up.name).suffix, delete=False) as f:
            f.write(up.read()); path = Path(f.name)
    st.video(str(path))
    if split in ("train", "val"):
        st.error(f"This video was in the {split.upper()} split: the model has already seen it, so a correct answer does NOT show it can "
                 "detect new fakes. Use a TEST video for that.")
    elif split == "test":
        st.success("This video was in the TEST split: the model never saw it in training. (Still only a few videos.)")
    if st.button("Predict"):
        venc, aenc = get_encoders()
        with st.spinner("cutting clips, extracting features, scoring..."):
            r = predict_video(path, run, venc, aenc)
        (st.error if r["verdict"] == "FAKE" else st.success)(f"Model says: {r['verdict']}   (mean p_fake {r['mean_p_fake']:.2f}, threshold {r['threshold']:.2f})")
        if truth:
            ok = truth.upper() == r["verdict"]
            st.write(f"True label: **{truth}**, so the model is **{'right' if ok else 'wrong'}** on this video.")
        st.bar_chart(r["clips"].set_index("start_s")["p_fake"])
        st.dataframe(r["clips"], hide_index=True)
        st.caption("Verdict rule for this demo [Proposed]: mean p_fake over the 2 s clips against the saved threshold (set once on validation, never changed).")


# ------------------------------------------------------------------ page 6
def page_results():
    import json
    import os
    st.title("Pilot results (real features, tiny data)")
    banner("PILOT on a handful of SWAN-DF demo videos with a placeholder video encoder. These numbers are NOT evidence for or against "
           "any hypothesis: the test set has very few real videos, and the numbers change a lot from seed to seed.")
    summ = Path(os.environ.get("RACL_RESULTS", ROOT / "results")) / "summary"
    if not (summ / "metrics_all.csv").exists():
        st.info("No summary yet. Run `python run_all.py --max_shift_ms 200 --batch_size 8 --seeds 0 1 2`."); st.stop()
    v = json.load(open(summ / "h2_verdict.json"))
    st.write(f"Decision rule outcome: **{v['outcome']}**")
    if "warning" in v:
        st.error(v["warning"])
    allm = pd.read_csv(summ / "metrics_all.csv")
    st.write(f"{allm.variant.nunique()} variants x {allm.seed.nunique() if 'seed' in allm else '?'} seeds. "
             "Margins are not set (D-06), so no variant is declared better or equivalent.")
    t1, t2, t3, t4 = st.tabs(["mean and std over seeds", "every run", "bootstrap intervals", "plots"])
    with t1:
        st.dataframe(pd.read_csv(summ / "metrics_mean_std.csv"))
    with t2:
        st.dataframe(allm)
    with t3:
        st.caption("Intervals over videos. With 2 real test videos they are very wide or undefined.")
        st.dataframe(pd.read_csv(summ / "bootstrap_ci.csv"))
    with t4:
        for f in sorted(summ.glob("*.png")):
            st.image(str(f), caption=f.name, width=520)


# ------------------------------------------------------------------ page 4
def page_tests():
    st.title("Automatic checks")
    st.write("Each test checks one rule of the method, for example: no identity in two splits; variant C never loads degraded "
             "data; L_pair gives no gradient to the clean branch; the threshold is saved once and never changed.")
    if st.button("Run all tests (about 1 minute)"):
        with st.spinner("running pytest..."):
            r = subprocess.run([sys.executable, "-m", "pytest", "tests", "-q"], cwd=ROOT, capture_output=True, text=True)
        (st.success if r.returncode == 0 else st.error)(r.stdout.strip().splitlines()[-1] if r.stdout.strip() else "no output")
        st.code(r.stdout[-3000:])


PAGES = {"1. Overview": page_overview, "2. Degradations": page_views, "3. Synthetic pilot": page_pilot, "4. Tests": page_tests, "5. Predict": page_predict, "6. Pilot results": page_results}
choice = st.sidebar.radio("Page", list(PAGES))
st.sidebar.caption("No real data and no real results yet.")
PAGES[choice]()