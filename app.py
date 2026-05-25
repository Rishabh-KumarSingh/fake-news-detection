import streamlit as st
import joblib
import re
import string
import os

# ── Page config ───────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="Fake News Detector",
    page_icon="🔍",
    layout="wide",
    initial_sidebar_state="expanded"
)

# ── Custom CSS ────────────────────────────────────────────────────────────────
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Space+Grotesk:wght@400;600;700&family=JetBrains+Mono&display=swap');

html, body, [class*="css"] {
    font-family: 'Space Grotesk', sans-serif;
    background-color: #0f1117;
    color: #e0e0e0;
}
.main { background-color: #0f1117; }

.stTextArea textarea {
    background-color: #1a1d27 !important;
    color: #e0e0e0 !important;
    border: 1px solid #2ecc71 !important;
    border-radius: 8px !important;
    font-family: 'JetBrains Mono', monospace !important;
    font-size: 14px !important;
}

.predict-btn button {
    background: linear-gradient(135deg, #2ecc71, #27ae60) !important;
    color: white !important;
    font-weight: 700 !important;
    border-radius: 8px !important;
    border: none !important;
    padding: 0.6rem 2rem !important;
    font-size: 16px !important;
    width: 100% !important;
}

.result-real {
    background: linear-gradient(135deg, #1a3a2a, #1e4d35);
    border-left: 5px solid #2ecc71;
    border-radius: 12px;
    padding: 20px 24px;
    margin: 16px 0;
}
.result-fake {
    background: linear-gradient(135deg, #3a1a1a, #4d1e1e);
    border-left: 5px solid #e74c3c;
    border-radius: 12px;
    padding: 20px 24px;
    margin: 16px 0;
}
.metric-card {
    background: #1a1d27;
    border: 1px solid #2c3350;
    border-radius: 10px;
    padding: 16px;
    text-align: center;
    margin: 6px 0;
}
.metric-card .value {
    font-size: 2rem;
    font-weight: 700;
    color: #2ecc71;
}
.metric-card .label {
    font-size: 0.8rem;
    color: #888;
    text-transform: uppercase;
    letter-spacing: 1px;
}
.confidence-bar-outer {
    background: #2c3350;
    border-radius: 20px;
    height: 14px;
    width: 100%;
    margin: 8px 0;
    overflow: hidden;
}
.confidence-bar-inner-real {
    background: linear-gradient(90deg, #2ecc71, #27ae60);
    height: 100%;
    border-radius: 20px;
    transition: width 0.5s ease;
}
.confidence-bar-inner-fake {
    background: linear-gradient(90deg, #e74c3c, #c0392b);
    height: 100%;
    border-radius: 20px;
    transition: width 0.5s ease;
}
h1, h2, h3 { color: #ffffff !important; }
.sidebar .sidebar-content { background-color: #1a1d27; }
</style>
""", unsafe_allow_html=True)

# ── Load model ────────────────────────────────────────────────────────────────
STOP_WORDS = set([
    'i','me','my','myself','we','our','ours','ourselves','you','your','yours',
    'yourself','he','him','his','himself','she','her','hers','herself','it','its',
    'itself','they','them','their','theirs','themselves','what','which','who','whom',
    'this','that','these','those','am','is','are','was','were','be','been','being',
    'have','has','had','having','do','does','did','doing','a','an','the','and','but',
    'if','or','because','as','until','while','of','at','by','for','with','about',
    'against','between','into','through','during','before','after','above','below',
    'to','from','up','down','in','out','on','off','over','under','again','further',
    'then','once','here','there','when','where','why','how','all','both','each',
    'few','more','most','other','some','such','no','nor','not','only','own','same',
    'so','than','too','very','s','t','can','will','just','don','should','now','d',
    'll','m','o','re','ve','y','ain','aren','couldn','didn','doesn','hadn','hasn',
    'haven','isn','ma','mightn','mustn','needn','shan','shouldn','wasn','weren',
    'won','wouldn'
])

def preprocess_text(text):
    text = str(text).lower()
    text = re.sub(r'http\S+|www\S+', '', text)
    text = re.sub(r'\d+', '', text)
    text = text.translate(str.maketrans('', '', string.punctuation))
    tokens = [w for w in text.split() if w not in STOP_WORDS and len(w) > 2]
    return " ".join(tokens)

@st.cache_resource
def load_model():
    model_path = r"C:\\Users\\91830\\Downloads\\fake_news_detection_project\\fake_news_detection\\models\\best_model.pkl"
    return joblib.load(model_path)

model = load_model()

# ── Sidebar ───────────────────────────────────────────────────────────────────
with st.sidebar:
    st.markdown("## 🔍 About")
    st.markdown("""
    **Fake News Detector** uses NLP + Machine Learning to classify news articles.
    
    **How it works:**
    1. Text is cleaned & preprocessed
    2. TF-IDF converts text to features
    3. Logistic Regression predicts the label
    
    **Signals of Fake News:**
    - Excessive exclamation marks !!!
    - ALL CAPS words
    - Emotional language
    - Unverified claims
    - Sensational headlines
    """)
    st.markdown("---")
    st.markdown("### 📊 Try Sample News")
    
    sample_real = st.button("📰 Load Real News Sample")
    sample_fake = st.button("🚨 Load Fake News Sample")

# ── Main UI ───────────────────────────────────────────────────────────────────
st.markdown("""
<div style='text-align:center; padding: 10px 0 20px 0;'>
  <h1 style='font-size:2.6rem; font-weight:700; margin-bottom:4px;'>
    🔍 Fake News Detector
  </h1>
  <p style='color:#888; font-size:1rem;'>
    Powered by NLP · TF-IDF · Logistic Regression &nbsp;|&nbsp; 
    
  </p>
</div>
""", unsafe_allow_html=True)

col_left, col_right = st.columns([3, 2], gap="large")

with col_left:
    st.markdown("### 📝 Enter News Article")

    real_sample = """Scientists Discover New Species of Deep-Sea Fish in Pacific Ocean

Marine biologists from the University of Hawaii have identified a previously unknown species of fish living at depths exceeding 3,000 meters. The discovery was made during a joint expedition funded by NOAA and published in the journal Nature. Sources have been verified."""

    fake_sample = """SHOCKING: Government Puts Mind-Control Chips in COVID Vaccines, Whistleblower Reveals!!!

A former government employee has come forward with EXPLOSIVE evidence that the COVID-19 vaccines contain MICROSCOPIC CHIPS designed to monitor and control the thoughts of citizens. The mainstream media is REFUSING to cover this story to protect their globalist masters. Share this before it gets DELETED! Wake up sheeple!!!"""

    default_text = ""
    if sample_real:
        default_text = real_sample
    elif sample_fake:
        default_text = fake_sample

    headline = st.text_input("Headline (optional)", placeholder="Enter the news headline …",
                              value="")
    body_text = st.text_area("Article Body", value=default_text,
                              height=220,
                              placeholder="Paste the full article or news snippet here …")

    st.markdown('<div class="predict-btn">', unsafe_allow_html=True)
    predict_clicked = st.button("🔍 Analyze Article", use_container_width=True)
    st.markdown('</div>', unsafe_allow_html=True)

with col_right:
    st.markdown("### 📈 Prediction Result")

    if predict_clicked and body_text.strip():
        full_text = (headline + " " + body_text).strip()
        clean     = preprocess_text(full_text)
        prediction  = model.predict([clean])[0]          # 0=REAL, 1=FAKE
        proba       = model.predict_proba([clean])[0]
        real_conf   = proba[0] * 100
        fake_conf   = proba[1] * 100
        label       = "FAKE" if prediction == 1 else "REAL"
        confidence  = fake_conf if label == "FAKE" else real_conf

        if label == "REAL":
            st.markdown(f"""
            <div class="result-real">
              <div style='font-size:2.2rem;'>✅</div>
              <div style='font-size:1.6rem; font-weight:700; color:#2ecc71;'>REAL NEWS</div>
              <div style='color:#aaa; margin-top:4px;'>This article appears to be credible</div>
            </div>
            """, unsafe_allow_html=True)
        else:
            st.markdown(f"""
            <div class="result-fake">
              <div style='font-size:2.2rem;'>🚨</div>
              <div style='font-size:1.6rem; font-weight:700; color:#e74c3c;'>FAKE NEWS</div>
              <div style='color:#aaa; margin-top:4px;'>This article shows signs of misinformation</div>
            </div>
            """, unsafe_allow_html=True)

        bar_color_class = "confidence-bar-inner-real" if label == "REAL" else "confidence-bar-inner-fake"
        st.markdown(f"""
        <div style='margin-top:12px;'>
          <div style='display:flex; justify-content:space-between;'>
            <span style='font-size:0.9rem; color:#aaa;'>Confidence</span>
            <span style='font-weight:700; color:{"#2ecc71" if label=="REAL" else "#e74c3c"};'>{confidence:.1f}%</span>
          </div>
          <div class="confidence-bar-outer">
            <div class="{bar_color_class}" style="width:{confidence}%;"></div>
          </div>
        </div>
        """, unsafe_allow_html=True)

        st.markdown("#### Probability Breakdown")
        c1, c2 = st.columns(2)
        with c1:
            st.markdown(f"""
            <div class="metric-card">
              <div class="value" style="color:#2ecc71;">{real_conf:.1f}%</div>
              <div class="label">Real Probability</div>
            </div>""", unsafe_allow_html=True)
        with c2:
            st.markdown(f"""
            <div class="metric-card">
              <div class="value" style="color:#e74c3c;">{fake_conf:.1f}%</div>
              <div class="label">Fake Probability</div>
            </div>""", unsafe_allow_html=True)

        # Signals detected
        st.markdown("#### 🔎 Signals Detected")
        excl_count = full_text.count('!')
        caps_count = sum(1 for w in full_text.split() if w.isupper() and len(w) > 2)
        word_count = len(full_text.split())
        signals = []
        if excl_count > 2:
            signals.append(f"⚠️ {excl_count} exclamation marks (sensationalism)")
        if caps_count > 3:
            signals.append(f"⚠️ {caps_count} ALL CAPS words (emotional tone)")
        trigger_words = ['shocking','breaking','exposed','bombshell','urgent',
                         'secret','leaked','whistleblower','globalist','wake up',
                         'sheeple','cover-up','they dont want','deep state']
        found = [w for w in trigger_words if w in full_text.lower()]
        if found:
            signals.append(f"⚠️ Trigger words: {', '.join(found[:4])}")
        if not signals:
            signals.append("✅ No major red flags detected")
        for s in signals:
            st.markdown(f"- {s}")

    elif predict_clicked:
        st.warning("Please enter some article text to analyze.")
    else:
        st.markdown("""
        <div style='text-align:center; padding:40px 20px; color:#555;'>
          <div style='font-size:3rem;'>📰</div>
          <div style='margin-top:12px; font-size:1rem;'>
            Enter a news article on the left and click <strong>Analyze</strong>
          </div>
          <div style='margin-top:8px; font-size:0.85rem;'>
            Or use the sample buttons in the sidebar
          </div>
        </div>
        """, unsafe_allow_html=True)

# ── Footer ────────────────────────────────────────────────────────────────────
st.markdown("---")
st.markdown("""
<div style='text-align:center; color:#555; font-size:0.8rem; padding:10px 0;'>
  Fake News Detection · Built with Python, scikit-learn & Streamlit
</div>
""", unsafe_allow_html=True)
