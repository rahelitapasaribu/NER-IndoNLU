
import streamlit as st
import pandas as pd
import re
import html
from collections import Counter

# ==========================================
# PAGE CONFIG
# ==========================================

st.set_page_config(
    page_title="EntityLens | NER System",
    page_icon="◈",
    layout="wide",
    initial_sidebar_state="expanded"
)

# ==========================================
# SESSION STATE
# ==========================================

if "news_text" not in st.session_state:
    st.session_state.news_text = ""

if "entities" not in st.session_state:
    st.session_state.entities = None

if "analyzed_text" not in st.session_state:
    st.session_state.analyzed_text = ""

# ==========================================
# ENTITY COLORS
# ==========================================

ENTITY_COLORS = {
    "Person": ("#E8EFFB", "#365C96"),
    "Location": ("#E5F2EC", "#39765A"),
    "Product/Brand": ("#F0EAF8", "#7955A0"),
    "Event": ("#FBF0E2", "#A36D32"),
    "Food/Beverage": ("#F8E9ED", "#A25470")
}

# ==========================================
# CUSTOM CSS
# ==========================================

st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=DM+Sans:wght@400;500;600;700&display=swap');

html, body, [class*="css"] {
    font-family: 'DM Sans', sans-serif;
}

.stApp {
    background-color: #F7F8FA;
}

.block-container {
    max-width: 1350px;
    padding: 2rem 3rem 3rem 3rem;
}

[data-testid="stSidebar"] {
    background-color: #FFFFFF;
    border-right: 1px solid #EAECF0;
}

[data-testid="stSidebar"] .block-container {
    padding: 2rem 1.2rem;
}

#MainMenu, footer, header {
    visibility: hidden;
}

/* Header */

.brand {
    font-size: 13px;
    font-weight: 700;
    letter-spacing: 2px;
    color: #52739A;
    margin-bottom: 12px;
}

.page-title {
    font-size: 31px;
    font-weight: 700;
    color: #202C3B;
    letter-spacing: -1px;
    margin-bottom: 5px;
}

.page-subtitle {
    color: #8792A1;
    font-size: 14px;
    margin-bottom: 28px;
}

/* Section */

.section-heading {
    color: #263548;
    font-size: 16px;
    font-weight: 700;
    margin-bottom: 5px;
}

.section-description {
    color: #929BA8;
    font-size: 12px;
    margin-bottom: 15px;
}

/* Input */

.stTextArea textarea {
    background: #FFFFFF !important;
    border: 1px solid #E3E7ED !important;
    border-radius: 10px !important;
    color: #344054 !important;
    font-size: 14px !important;
    line-height: 1.8 !important;
    padding: 16px !important;
}

.stTextArea textarea:focus {
    border-color: #8CA5C3 !important;
    box-shadow: 0 0 0 1px #8CA5C3 !important;
}

/* Buttons */

.stButton > button {
    background: #304B68;
    color: white;
    border: none;
    border-radius: 8px;
    height: 42px;
    font-size: 13px;
    font-weight: 600;
    transition: 0.2s;
}

.stButton > button:hover {
    background: #20364D;
    color: white;
}

/* Cards */

.metric-card {
    background: #FFFFFF;
    border: 1px solid #EAECF0;
    border-radius: 11px;
    padding: 18px 20px;
}

.metric-label {
    color: #8A94A3;
    font-size: 12px;
    font-weight: 500;
}

.metric-value {
    color: #263548;
    font-size: 25px;
    font-weight: 700;
    margin-top: 5px;
}

/* Result panel */

.result-panel {
    background: #FFFFFF;
    border: 1px solid #EAECF0;
    border-radius: 11px;
    padding: 22px;
    min-height: 180px;
}

.article-text {
    color: #475467;
    font-size: 14px;
    line-height: 2.2;
    white-space: pre-wrap;
}

/* Table */

[data-testid="stDataFrame"] {
    border: 1px solid #EAECF0;
    border-radius: 9px;
    overflow: hidden;
}

/* Divider */

hr {
    border-color: #EAECF0 !important;
}

/* Footer */

.footer {
    color: #A0A8B3;
    text-align: center;
    font-size: 11px;
    padding-top: 35px;
}
</style>
""", unsafe_allow_html=True)


# ==========================================
# SIDEBAR
# ==========================================

with st.sidebar:

    st.markdown("""
    <div style="padding: 5px 0 25px 0;">
        <div style="font-size:25px; font-weight:700; color:#304B68;">
            ◈ EntityLens
        </div>
        <div style="font-size:11px; color:#98A2B3; margin-top:4px;">
            INDONESIAN NER SYSTEM
        </div>
    </div>
    """, unsafe_allow_html=True)

    st.markdown("##### CONFIGURATION")

    model_choice = st.selectbox(
        "Select model",
        ["CRF", "Transformer"]
    )

    st.caption(
        "Choose the model used for entity extraction."
    )

    st.divider()

    st.markdown("##### ENTITY CATEGORIES")

    for name, (bg, color) in ENTITY_COLORS.items():

        st.markdown(
            f"""
            <div style="
                display:flex;
                align-items:center;
                gap:10px;
                margin:13px 0;
                font-size:12px;
                color:#475467;
            ">
                <div style="
                    width:10px;
                    height:10px;
                    border-radius:3px;
                    background:{color};
                "></div>
                {name}
            </div>
            """,
            unsafe_allow_html=True
        )

    st.divider()

    st.markdown("""
    <div style="
        font-size:11px;
        color:#98A2B3;
        line-height:1.8;
    ">
        <b style="color:#667085;">PROJECT INFORMATION</b><br>
        Named Entity Recognition<br>
        Dataset: IndoNLU NERP<br>
        Language: Indonesian
    </div>
    """, unsafe_allow_html=True)


# ==========================================
# HEADER
# ==========================================

st.markdown(
    '<div class="brand">TEXT ANALYSIS / NER</div>',
    unsafe_allow_html=True
)

st.markdown(
    '<div class="page-title">Named Entity Recognition</div>',
    unsafe_allow_html=True
)

st.markdown(
    '<div class="page-subtitle">'
    'Identify and extract named entities from Indonesian news articles.'
    '</div>',
    unsafe_allow_html=True
)


# ==========================================
# INPUT SECTION
# ==========================================

st.markdown(
    '<div class="section-heading">News Article</div>',
    unsafe_allow_html=True
)

st.markdown(
    '<div class="section-description">'
    'Paste the news content you want to analyze.'
    '</div>',
    unsafe_allow_html=True
)

examples = {
    "Example 1": (
        "Presiden Joko Widodo menghadiri Festival Budaya "
        "di Jakarta bersama sejumlah pejabat pemerintah."
    ),
    "Example 2": (
        "Festival Kuliner Nusantara diselenggarakan di "
        "Bali. Pengunjung dapat menikmati nasi goreng "
        "dan berbagai makanan tradisional."
    )
}

col_example, col_empty = st.columns([1, 2])

with col_example:

    example = st.selectbox(
        "Sample article",
        ["Choose an example"] + list(examples.keys())
    )

with col_empty:
    st.write("")

if example != "Choose an example":

    if st.button("Load sample article"):

        st.session_state.news_text = examples[example]
        st.session_state.entities = None
        st.rerun()

news_text = st.text_area(
    "Article content",
    key="news_text",
    height=190,
    placeholder="Enter Indonesian news article here...",
    label_visibility="collapsed"
)

col_info, col_button = st.columns([3, 1])

with col_info:
    st.caption(
        f"{len(news_text.split())} words · {len(news_text)} characters"
    )

with col_button:

    analyze = st.button(
        "Analyze Article",
        use_container_width=True
    )


# ==========================================
# DUMMY INFERENCE
# ==========================================

if analyze:

    if not news_text.strip():

        st.warning("Please enter a news article.")

    else:

        # DEMO ONLY
        # Replace this section with actual model inference later.

        demo_entities = [
            {"entity": "Joko Widodo", "type": "Person"},
            {"entity": "Festival Budaya", "type": "Event"},
            {"entity": "Jakarta", "type": "Location"}
        ]

        # Keep only entities that actually appear in the input.
        detected = []

        for item in demo_entities:

            if re.search(
                re.escape(item["entity"]),
                news_text,
                flags=re.IGNORECASE
            ):
                detected.append(item)

        st.session_state.entities = detected
        st.session_state.analyzed_text = news_text


# ==========================================
# HIGHLIGHT FUNCTION
# ==========================================

def render_highlight(text, entities):

    matches = []

    for item in entities:

        pattern = re.compile(
            re.escape(item["entity"]),
            flags=re.IGNORECASE
        )

        for match in pattern.finditer(text):

            matches.append({
                "start": match.start(),
                "end": match.end(),
                "type": item["type"],
                "entity": match.group()
            })

    # Avoid overlapping spans
    matches.sort(key=lambda x: (x["start"], -(x["end"] - x["start"])))

    filtered = []
    last_end = -1

    for match in matches:

        if match["start"] >= last_end:
            filtered.append(match)
            last_end = match["end"]

    output = []
    current = 0

    for match in filtered:

        output.append(html.escape(text[current:match["start"]]))

        bg, color = ENTITY_COLORS[match["type"]]

        output.append(
            f'<span style="background:{bg};color:{color};'
            f'padding:3px 6px;border-radius:4px;font-weight:600;">'
            f'{html.escape(match["entity"])}</span>'
        )

        current = match["end"]

    output.append(html.escape(text[current:]))

    return "".join(output)


# ==========================================
# RESULTS
# ==========================================

if st.session_state.entities is not None:

    entities = st.session_state.entities

    st.divider()

    st.markdown(
        '<div class="section-heading">Analysis Results</div>',
        unsafe_allow_html=True
    )

    st.markdown(
        '<div class="section-description">'
        'Overview of the entities identified in the article.'
        '</div>',
        unsafe_allow_html=True
    )

    st.info(
        "Demo mode: displayed results are illustrative, "
        "not predictions from the trained model."
    )

    # METRICS

    counts = Counter(
        item["type"] for item in entities
    )

    metric_cols = st.columns(3)

    metrics = [
        ("Total Entities", len(entities)),
        ("Entity Categories", len(counts)),
        ("Model Selected", model_choice)
    ]

    for col, (label, value) in zip(metric_cols, metrics):

        with col:

            st.markdown(
                f"""
                <div class="metric-card">
                    <div class="metric-label">{label}</div>
                    <div class="metric-value">{value}</div>
                </div>
                """,
                unsafe_allow_html=True
            )

    st.write("")

    # TWO COLUMN RESULT

    left, right = st.columns([1.35, 1])

    with left:

        st.markdown("##### Highlighted Article")

        highlighted = render_highlight(
            st.session_state.analyzed_text,
            entities
        )

        st.markdown(
            f"""
            <div class="result-panel">
                <div class="article-text">{highlighted}</div>
            </div>
            """,
            unsafe_allow_html=True
        )

    with right:

        st.markdown("##### Extracted Entities")

        if entities:

            df = pd.DataFrame(entities)

            df.columns = ["Entity Text", "Entity Type"]

            df.insert(
                0,
                "No.",
                range(1, len(df) + 1)
            )

            st.dataframe(
                df,
                use_container_width=True,
                hide_index=True
            )

        else:

            st.warning(
                "No sample entities found in this article."
            )

    # CATEGORY DISTRIBUTION

    st.write("")
    st.markdown("##### Entity Distribution")

    distribution_cols = st.columns(5)

    for col, (name, (bg, color)) in zip(
        distribution_cols,
        ENTITY_COLORS.items()
    ):

        with col:

            st.markdown(
                f"""
                <div style="
                    background:white;
                    border:1px solid #EAECF0;
                    border-radius:9px;
                    padding:14px;
                ">
                    <div style="
                        font-size:11px;
                        color:#98A2B3;
                    ">{name}</div>
                    <div style="
                        font-size:22px;
                        font-weight:700;
                        color:{color};
                        margin-top:5px;
                    ">{counts.get(name, 0)}</div>
                </div>
                """,
                unsafe_allow_html=True
            )


# ==========================================
# FOOTER
# ==========================================

st.markdown("""
<div class="footer">
    EntityLens · Indonesian Named Entity Recognition
    <br>
    G1 Project · NLP
</div>
""", unsafe_allow_html=True)
