import streamlit as st
from inference import (
    load_tokenizer, load_model,
    predict_next_words, build_token_sequence
)

# ---- Page config ----
st.set_page_config(
    page_title="Next Word Predictor",
    page_icon="✍️",
    layout="centered"
)

# ---- Custom CSS ----
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Playfair+Display:ital,wght@0,500;1,400&family=DM+Mono:wght@400;500&display=swap');

html, body, [class*="css"] {
    font-family: 'DM Mono', monospace;
}
h1, h2, h3 {
    font-family: 'Playfair Display', serif !important;
}
.style-badge {
    display: inline-block;
    padding: 4px 14px;
    border-radius: 20px;
    font-size: 12px;
    font-weight: 500;
    margin-bottom: 1rem;
}
.badge-shakespeare {
    background: #2C2C2A;
    color: #FAC775;
}
.badge-normal {
    background: #E1F5EE;
    color: #085041;
}
.sentence-box {
    background: #F8F7F4;
    border-left: 3px solid #888780;
    padding: 12px 16px;
    border-radius: 0 8px 8px 0;
    font-size: 15px;
    color: #2C2C2A;
    margin: 1rem 0;
    min-height: 48px;
    font-family: 'Playfair Display', serif;
    font-style: italic;
}
.suggestion-btn {
    margin: 4px 4px 4px 0;
}
.context-note {
    font-size: 11px;
    color: #888780;
    margin-top: 0.5rem;
}
</style>
""", unsafe_allow_html=True)


# ---- Load model + tokenizer (cached) ----
@st.cache_resource
def load_resources():
    tokenizer = load_tokenizer("vocab.pkl")
    lstm_model = load_model("lstm_model_best.pth")
    return tokenizer, lstm_model

tokenizer, model = load_resources()

STYLE_TOKENS = {"Normal": 2, "Shakespearean": 1}


# ---- Session state init ----
if "words" not in st.session_state:
    st.session_state.words = []          # list of word strings typed so far
if "suggestions" not in st.session_state:
    st.session_state.suggestions = []
if "style" not in st.session_state:
    st.session_state.style = "Normal"


# ---- Header ----
st.title("Next Word Predictor")
st.caption("LSTM · trained on 1.1M sequences · 62k vocab · Normal & Shakespearean styles")


# ---- Style selector ----
col1, col2 = st.columns([2, 3])
with col1:
    style = st.radio("Prediction style", ["Normal", "Shakespearean"],
                     horizontal=True, key="style_radio")
    if style != st.session_state.style:
        # Reset on style change
        st.session_state.style = style
        st.session_state.words = []
        st.session_state.suggestions = []

style_token = STYLE_TOKENS[st.session_state.style]

badge_cls = "badge-shakespeare" if style == "Shakespearean" else "badge-normal"
badge_label = "✦ Shakespearean mode" if style == "Shakespearean" else "◎ Normal mode"
st.markdown(f'<div class="style-badge {badge_cls}">{badge_label}</div>',
            unsafe_allow_html=True)


# ---- Current sentence display ----
sentence_display = " ".join(st.session_state.words) if st.session_state.words else "Start typing below..."
st.markdown(f'<div class="sentence-box">{sentence_display}</div>', unsafe_allow_html=True)

# Context window note
if len(st.session_state.words) >= 18:
    st.markdown(f'<p class="context-note">⚠ Context window: using last 19 words of {len(st.session_state.words)} total.</p>',
                unsafe_allow_html=True)


# ---- Word input ----
with st.form("word_form", clear_on_submit=True):
    user_word = st.text_input("Type a word and press Enter", placeholder="e.g. hello")
    submitted = st.form_submit_button("Add word")

if submitted and user_word.strip():
    word = user_word.strip().lower()
    st.session_state.words.append(word)

    token_seq = build_token_sequence(
        style_token,
        st.session_state.words,
        tokenizer
    )
    st.session_state.suggestions = predict_next_words(
        token_seq, model, tokenizer,
        top_k=6, max_len=20, style_token=style_token
    )
    st.rerun()


# ---- Suggestions ----
if st.session_state.suggestions:
    st.markdown("**Suggested next words** — click to add:")
    cols = st.columns(len(st.session_state.suggestions))
    for i, (word, prob) in enumerate(st.session_state.suggestions):
        with cols[i]:
            if st.button(f"{word}\n{prob}%", key=f"sug_{i}"):
                st.session_state.words.append(word)
                token_seq = build_token_sequence(
                    style_token,
                    st.session_state.words,
                    tokenizer
                )
                st.session_state.suggestions = predict_next_words(
                    token_seq, model, tokenizer,
                    top_k=6, max_len=20, style_token=style_token
                )
                st.rerun()


# ---- Controls ----
st.divider()
col_a, col_b, col_c = st.columns(3)

with col_a:
    if st.button("⌫ Undo last word") and st.session_state.words:
        st.session_state.words.pop()
        if st.session_state.words:
            token_seq = build_token_sequence(
                style_token, st.session_state.words, tokenizer
            )
            st.session_state.suggestions = predict_next_words(
                token_seq, model, tokenizer,
                top_k=6, max_len=20, style_token=style_token
            )
        else:
            st.session_state.suggestions = []
        st.rerun()

with col_b:
    if st.button("↺ Reset sentence"):
        st.session_state.words = []
        st.session_state.suggestions = []
        st.rerun()

with col_c:
    full_sentence = " ".join(st.session_state.words)
    st.download_button("⬇ Save sentence", full_sentence,
                       file_name="sentence.txt", mime="text/plain",
                       disabled=not st.session_state.words)


# ---- Stats sidebar ----
with st.sidebar:
    st.markdown("### Session stats")
    st.metric("Words typed", len(st.session_state.words))
    st.metric("Style", st.session_state.style)
    st.metric("Context fed to model",
              f"{min(len(st.session_state.words)+1, 20)} tokens")

    st.divider()
    st.markdown("### About")
    st.markdown("""
- **Architecture**: 2-layer LSTM  
- **Vocab**: 62,549 tokens  
- **Training data**: 1.1M sequences  
- **Styles**: Normal + Shakespearean  
- **Best test PPL**: 925.6  
    """)

    st.divider()
    st.markdown("### How it works")
    st.markdown("""
1. Choose a style
2. Type your first word
3. Pick from suggestions or type your own
4. The model sees: `<style_token> word1 word2 ...`
5. Context capped at 20 tokens
    """)