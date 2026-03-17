import streamlit as st
import numpy as np
import pickle
import re
import nltk
from nltk.stem import WordNetLemmatizer
from nltk.corpus import stopwords
import unicodedata

# ── Page config ───────────────────────────────────────────────
st.set_page_config(
    page_title="Contract Analyzer",
    page_icon="",
    layout="wide"
)

# ── Simple clean styling ──────────────────────────────────────
st.markdown("""
<style>
    .main { background-color: #f8f9fa; }

    /* Risk cards */
    .risk-high   { background:#ffe0e0; border-left:5px solid #ff4444;
                   padding:10px; border-radius:5px; margin:5px 0; }
    .risk-medium { background:#fff3cd; border-left:5px solid #ffa500;
                   padding:10px; border-radius:5px; margin:5px 0; }
    .risk-low    { background:#d4edda; border-left:5px solid #28a745;
                   padding:10px; border-radius:5px; margin:5px 0; }

    /* Metric box */
    .metric-box  { background:#ffffff; border:1px solid #dee2e6;
                   padding:15px; border-radius:8px;
                   text-align:center; }

    /* Clause section headers */
    .clause-section-header {
        font-weight: 700;
        font-size: 0.85rem;
        letter-spacing: 0.06em;
        text-transform: uppercase;
        padding: 6px 12px;
        border-radius: 4px 4px 0 0;
        margin-bottom: 0;
    }

    /* Clause value box — sits below its header */
    .clause-value {
        padding: 10px 14px;
        border-radius: 0 0 6px 6px;
        font-size: 0.92rem;
        line-height: 1.55;
        margin-bottom: 10px;
    }

    /* Not-found box */
    .clause-missing {
        background: #f8f9fa;
        border: 1px dashed #adb5bd;
        border-left: 4px solid #adb5bd;
        padding: 8px 12px;
        border-radius: 5px;
        margin-bottom: 8px;
        color: #6c757d;
        font-size: 0.88rem;
    }

    /* Per-clause-type color themes */
    .ch-parties   .clause-section-header { background:#dbeafe; color:#1e40af; }
    .ch-parties   .clause-value          { background:#eff6ff; border:1px solid #bfdbfe; border-top:none; }

    .ch-date      .clause-section-header { background:#dcfce7; color:#166534; }
    .ch-date      .clause-value          { background:#f0fdf4; border:1px solid #bbf7d0; border-top:none; }

    .ch-payment   .clause-section-header { background:#fef9c3; color:#854d0e; }
    .ch-payment   .clause-value          { background:#fefce8; border:1px solid #fde68a; border-top:none; }

    .ch-term      .clause-section-header { background:#fee2e2; color:#991b1b; }
    .ch-term      .clause-value          { background:#fff5f5; border:1px solid #fecaca; border-top:none; }

    .ch-liability .clause-section-header { background:#ede9fe; color:#5b21b6; }
    .ch-liability .clause-value          { background:#f5f3ff; border:1px solid #ddd6fe; border-top:none; }

    .ch-juris     .clause-section-header { background:#ffedd5; color:#9a3412; }
    .ch-juris     .clause-value          { background:#fff7ed; border:1px solid #fed7aa; border-top:none; }

    .ch-renewal   .clause-section-header { background:#cffafe; color:#164e63; }
    .ch-renewal   .clause-value          { background:#ecfeff; border:1px solid #a5f3fc; border-top:none; }
</style>
""", unsafe_allow_html=True)


# ══════════════════════════════════════════════════════════════
# LOAD MODELS — cached so they only load once
# ══════════════════════════════════════════════════════════════
@st.cache_resource
def load_all_models():
    import onnxruntime as ort

    # Load Word2Vec vectors (no gensim needed)
    vectors = np.load("w2v_vectors.npy")
    with open("w2v_vocab.txt") as f:
        vocab = f.read().splitlines()
    w2v = dict(zip(vocab, vectors))

    # Load ONNX model
    sess_options = ort.SessionOptions()
    sess_options.log_severity_level = 3
    lstm_session = ort.InferenceSession(
        "lstm_model.onnx",
        sess_options=sess_options
    )

    # Load label encoder
    with open("label_encoder.pkl", "rb") as f:
        le = pickle.load(f)

    # Download NLTK data
    nltk.download("punkt",        quiet=True)
    nltk.download("punkt_tab",    quiet=True)
    nltk.download("stopwords",    quiet=True)
    nltk.download("wordnet",      quiet=True)
    nltk.download("averaged_perceptron_tagger", quiet=True)
    nltk.download("maxent_ne_chunker", quiet=True)
    nltk.download("words",        quiet=True)

    return w2v, lstm_session, le

# ══════════════════════════════════════════════════════════════
# PREPROCESSING
# ══════════════════════════════════════════════════════════════
LEGAL_PRESERVE = {
    "shall", "must", "may", "will", "should",
    "not", "no", "never", "neither", "nor",
    "if", "unless", "until", "except", "provided",
    "without", "including", "notwithstanding",
    "party", "parties", "each", "both", "either",
    "during", "before", "after", "within", "upon",
    "whereas", "hereby", "thereof", "herein"
}

def preprocess_text(text):
    # Normalize unicode
    text = unicodedata.normalize("NFKD", text)
    text = text.encode("ascii", "ignore").decode("ascii")
    text = text.lower()

    # Remove noise
    text = re.sub(r'-\s*\d+\s*-', '', text)
    text = re.sub(r'page\s+\d+\s+of\s+\d+', '', text,
                  flags=re.IGNORECASE)
    text = re.sub(r'\s+', ' ', text)

    # Tokenize
    tokens = nltk.word_tokenize(text)

    # Stopword removal — preserve legal terms
    stop_words = set(stopwords.words("english")) - LEGAL_PRESERVE
    tokens = [t for t in tokens if t not in stop_words]

    # Lemmatize
    lemmatizer = WordNetLemmatizer()
    tokens = [lemmatizer.lemmatize(t) for t in tokens]

    return tokens


def tokens_to_vector(tokens, w2v_lookup, max_len=512, vector_size=200):
    """Convert tokens to padded sequence for LSTM"""
    sequence = [
        w2v_lookup.get(t, np.zeros(vector_size))
        for t in tokens[:max_len]
    ]
    while len(sequence) < max_len:
        sequence.append(np.zeros(vector_size))
    return np.array(sequence)


# ══════════════════════════════════════════════════════════════
# CLAUSE EXTRACTION
# ══════════════════════════════════════════════════════════════
PATTERNS = {
    "Effective Date": [
        r"(?:effective|entered|dated?|made)\s+(?:as\s+of\s+)?"
        r"([A-Z][a-z]+\s+\d{1,2},?\s+\d{4})",
        r"as\s+of\s+the\s+(\d+(?:st|nd|rd|th)?\s+day\s+of\s+"
        r"[A-Z][a-z]+,?\s+\d{4})",
        r"this\s+(\d+(?:st|nd|rd|th)?\s+day\s+of\s+"
        r"[A-Z][a-z]+,?\s+\d{4})",
        r"(?:effective|dated?)\s+(?:as\s+of\s+)?"
        r"(\d{1,2}[/-]\d{1,2}[/-]\d{2,4})",
    ],
    "Payment Terms": [
        r"(net\s+\d+\s*days?)",
        r"(\d+(?:\.\d+)?%)\s+of\s+(?:net\s+|gross\s+)?"
        r"(?:revenue|sales|receipts)",
        r"royalt(?:y|ies)\s+(?:of\s+|equal\s+to\s+)?"
        r"(\d+(?:\.\d+)?%)",
        r"(?:fee|compensation)\s+of\s+(\$[\d,]+(?:\.\d{2})?)",
        r"(\$[\d,]+(?:\.\d{2})?)\s+per\s+"
        r"(?:month|year|quarter)",
        r"(?:compensation|payment)\s+(?:as\s+)?set\s+forth\s+in\s+"
        r"(?:Exhibit|Schedule|Section)\s+([A-Z0-9]+)",
    ],
    "Termination": [
        r"(?:either|any)\s+party\s+may\s+terminate[^\.]{0,150}",
        r"terminat(?:e|ion)\s+(?:for\s+convenience|without\s+cause)"
        r"[^\.]{0,100}",
        r"(\d+)\s+days?\s+(?:prior\s+|advance\s+|written\s+)?"
        r"notice\s+(?:of\s+)?(?:termination|cancellation)",
        r"(?:agreement|term)\s+shall\s+(?:expire|terminate)"
        r"[^\.]{0,80}",
    ],
    "Liability Limit": [
        r"(?:liability|damages)\s+(?:shall\s+)?(?:not\s+exceed|"
        r"be\s+limited\s+to)\s+([^\.]{5,100})",
        r"in\s+no\s+event\s+shall[^\.]{0,200}",
        r"(?:indirect|consequential|punitive|special)\s+"
        r"damages[^\.]{0,100}",
        r"(?:aggregate|maximum)\s+liability[^\.]{0,100}",
    ],
    "Jurisdiction": [
        r"governed\s+by\s+(?:and\s+construed\s+)?(?:in\s+accordance"
        r"\s+with\s+)?(?:the\s+)?laws?\s+of\s+(?:the\s+)?"
        r"(?:State\s+of\s+)?([A-Z][a-zA-Z\s]{2,25})",
        r"(?:State|Commonwealth)\s+of\s+([A-Z][a-zA-Z\s]{2,20})",
        r"courts?\s+of\s+(?:the\s+)?(?:State\s+of\s+)?"
        r"([A-Z][a-zA-Z\s]{2,25})",
        r"(?:New\s+York|California|Delaware|Texas|Florida|"
        r"Illinois|Massachusetts)[^\.]{0,50}"
        r"(?:law|jurisdiction|court|govern)",
    ],
    "Renewal": [
        r"automatically?\s+(?:renew|extend|be\s+extended)"
        r"[^\.]{0,100}",
        r"successive\s+(?:one[\-\s]year|annual|monthly)"
        r"\s+(?:renewal\s+)?terms?[^\.]{0,80}",
        r"(?:continue|renew)[^\.]{0,60}unless[^\.]{0,60}",
    ]
}


def extract_parties(text):
    """
    Extract contracting parties using NLTK NER
    No spaCy required
    """
    header  = text[:1500]
    parties = []

    # Method 1: NLTK NER chunker
    try:
        tokens   = nltk.word_tokenize(header)
        pos_tags = nltk.pos_tag(tokens)
        chunks   = nltk.ne_chunk(pos_tags, binary=False)

        for chunk in chunks:
            if hasattr(chunk, "label"):
                if chunk.label() in ["ORGANIZATION", "PERSON"]:
                    name = " ".join(
                        c[0] for c in chunk
                    ).strip()
                    if (2 < len(name) < 80 and
                            name not in parties):
                        parties.append(name)
    except Exception:
        pass

    # Method 2: Regex for explicit party definitions
    party_pattern = (
        r'([A-Z][A-Za-z\s,\.]+(?:Inc|LLC|Ltd|Corp|Co|LP)'
        r'?\.?)\s*\("([^"]{2,30})"\)'
    )
    for full_name, _ in re.findall(party_pattern, header):
        full_name = full_name.strip()
        if (full_name not in parties and
                3 < len(full_name) < 100):
            parties.append(full_name)

    # Method 3: All-caps company names
    caps_pattern = r'\b([A-Z]{2,}(?:\s+[A-Z]{2,})+)\b'
    caps_matches = re.findall(caps_pattern, header)
    for match in caps_matches[:3]:
        if (match not in parties and
                len(match) > 4 and
                match not in ["THIS AGREEMENT",
                              "THE PARTIES"]):
            parties.append(match)

    return parties[:5]


def extract_clauses(text):
    results = {"Contracting Parties": extract_parties(text)}

    for clause_name, patterns in PATTERNS.items():
        found = []
        for pattern in patterns:
            try:
                matches = re.findall(
                    pattern, text, re.IGNORECASE | re.DOTALL
                )
                for match in matches:
                    if isinstance(match, tuple):
                        match = " ".join(
                            m for m in match if m
                        ).strip()
                    match = re.sub(r'\s+', ' ', match).strip()
                    if (3 < len(match) < 300 and
                            match.lower() not in
                            [f.lower() for f in found]):
                        found.append(match)
            except re.error:
                continue
        results[clause_name] = found[:3]

    return results


# ══════════════════════════════════════════════════════════════
# RISK ASSESSMENT
# ══════════════════════════════════════════════════════════════
CRITICAL_CLAUSES = {
    "Alliance & Cooperation":       ["parties", "termination",
                                     "jurisdiction"],
    "Distribution & Sales":         ["parties", "termination",
                                     "jurisdiction"],
    "License & IP":                 ["parties", "termination",
                                     "liability_limit",
                                     "jurisdiction"],
    "Services & Outsourcing":       ["parties", "termination",
                                     "liability_limit",
                                     "jurisdiction"],
    "Development & Manufacturing":  ["parties", "termination",
                                     "liability_limit",
                                     "jurisdiction"],
}

RISK_FLAGS = {
    "Unlimited Liability": {
        "severity": "HIGH",
        "description": "No liability cap — potential unlimited exposure",
        "patterns": [
            r"liability\s+shall\s+(?:not\s+be\s+)?limited",
            r"unlimited\s+liability",
            r"liable\s+for\s+(?:any\s+and\s+)?all\s+damages"
            r"(?!\s+(?:except|excluding))"
        ]
    },
    "Auto Renewal": {
        "severity": "MEDIUM",
        "description": "Contract auto-renews — action needed to cancel",
        "patterns": [
            r"automatically?\s+(?:renew|be\s+renewed)",
            r"auto[\-\s]?renew",
            r"deemed\s+(?:to\s+be\s+)?renewed"
        ]
    },
    "Broad Indemnification": {
        "severity": "HIGH",
        "description": "Broad indemnification obligation detected",
        "patterns": [
            r"indemnif[^\.]{0,100}"
            r"(?:any\s+and\s+all|unlimited)[^\.]{0,50}"
            r"(?:claim|loss|damage)",
            r"(?:defend|indemnify)\s+and\s+hold\s+harmless"
            r"[^\.]{0,150}third\s+part",
        ]
    },
    "IP Ownership Risk": {
        "severity": "HIGH",
        "description": "IP ownership fully assigned away",
        "patterns": [
            r"work\s+made\s+for\s+hire",
            r"hereby\s+assigns?\s+all\s+(?:right|title|interest)"
            r"[^\.]{0,60}intellectual\s+property"
        ]
    },
    "Non-Compete": {
        "severity": "MEDIUM",
        "description": "Non-compete restriction found",
        "patterns": [
            r"(?:shall\s+not|will\s+not)\s+"
            r"(?:directly\s+or\s+indirectly\s+)?compete",
            r"non[\-\s]compete\s+(?:agreement|clause|provision)",
        ]
    },
    "No Dispute Resolution": {
        "severity": "LOW",
        "description": "No dispute resolution mechanism found",
        "check_absence": True,
        "patterns": [
            r"arbitrat", r"mediat",
            r"dispute\s+resolution", r"\badr\b"
        ]
    }
}


def assess_risk(clauses_dict, text, contract_type):
    risks = []

    clause_key_map = {
        "parties":        "Contracting Parties",
        "termination":    "Termination",
        "liability_limit":"Liability Limit",
        "jurisdiction":   "Jurisdiction"
    }
    required = CRITICAL_CLAUSES.get(contract_type, [
        "parties", "termination", "jurisdiction"
    ])

    for req in required:
        display_name = clause_key_map.get(req, req)
        value = clauses_dict.get(display_name, [])
        if not value:
            risks.append({
                "name":        f"Missing: {display_name}",
                "severity":    "HIGH",
                "description": f"Critical clause not found — "
                               f"required for {contract_type}",
                "context":     ""
            })

    for flag_name, flag_info in RISK_FLAGS.items():
        if flag_info.get("check_absence"):
            found = any(
                re.search(p, text, re.IGNORECASE)
                for p in flag_info["patterns"]
            )
            if not found:
                risks.append({
                    "name":        flag_name,
                    "severity":    flag_info["severity"],
                    "description": flag_info["description"],
                    "context":     ""
                })
        else:
            for pattern in flag_info["patterns"]:
                match = re.search(
                    pattern, text, re.IGNORECASE | re.DOTALL
                )
                if match:
                    start   = max(0, match.start() - 20)
                    end     = min(len(text), match.end() + 60)
                    context = re.sub(
                        r'\s+', ' ', text[start:end]
                    ).strip()
                    risks.append({
                        "name":        flag_name,
                        "severity":    flag_info["severity"],
                        "description": flag_info["description"],
                        "context":     context[:120]
                    })
                    break

    scores      = {"HIGH": 3, "MEDIUM": 2, "LOW": 1}
    total_score = sum(scores.get(r["severity"], 0)
                      for r in risks)

    if total_score <= 2:
        level, color = "LOW RISK",    "#28a745"
    elif total_score <= 5:
        level, color = "MEDIUM RISK", "#ffa500"
    else:
        level, color = "HIGH RISK",   "#dc3545"

    return risks, total_score, level, color


# ══════════════════════════════════════════════════════════════
# MAIN APP
# ══════════════════════════════════════════════════════════════
def main():
    # Header
    st.title("Legal Contract Analyzer")
    st.markdown(
        "Upload a contract to get instant **classification**, "
        "**clause extraction**, and **risk assessment**."
    )
    st.divider()

    # Load models
    with st.spinner("Loading models..."):
        try:
            w2v, lstm_session, le = load_all_models()
            st.success("Models loaded successfully!")
        except Exception as e:
            st.error(f"Error loading models: {e}")
            st.info(
                "Make sure these files are in the same folder:\n"
                "- lstm_final.keras\n"
                "- legal_word2vec.model\n"
                "- label_encoder.pkl"
            )
            return

    st.divider()

    # Input section
    st.subheader("Input Contract")

    input_method = st.radio(
        "Choose input method:",
        ["Paste contract text", "Upload .txt file"],
        horizontal=True
    )

    contract_text = ""

    if input_method == "Paste contract text":
        contract_text = st.text_area(
            "Paste your contract text here:",
            height=250,
            placeholder="STRATEGIC ALLIANCE AGREEMENT\n\n"
                        "This Agreement is entered into as of "
                        "January 1, 2024..."
        )
    else:
        uploaded = st.file_uploader(
            "Upload contract (.txt)",
            type=["txt"]
        )
        if uploaded:
            contract_text = uploaded.read().decode(
                "utf-8", errors="ignore"
            )
            st.success(
                f"Loaded: {uploaded.name} "
                f"({len(contract_text.split())} words)"
            )
            with st.expander("Preview contract text"):
                st.text(contract_text[:1000] + "...")

    # Analyze button
    if st.button("Analyze Contract",
                 type="primary",
                 disabled=not contract_text.strip()):

        if len(contract_text.split()) < 50:
            st.warning(
                "Contract text too short. "
                "Please provide more content."
            )
            return

        with st.spinner("Analyzing contract..."):

            # ── Step 1: Classify ──────────────────────────────
            tokens  = preprocess_text(contract_text)
            seq     = tokens_to_vector(tokens, w2v)
            seq_inp = seq.reshape(1, 512, 200)

            input_name = lstm_session.get_inputs()[0].name
            seq_float  = seq_inp.astype(np.float32)
            probs      = lstm_session.run(
                None, {input_name: seq_float}
            )[0][0]
            pred_idx   = np.argmax(probs)
            pred_label = le.classes_[pred_idx]

            # ── Step 2: Extract clauses ───────────────────────
            clauses = extract_clauses(contract_text)

            # ── Step 3: Risk assessment ───────────────────────
            risks, score, level, color = assess_risk(
                clauses, contract_text, pred_label
            )

        # ════════════════════════════════════════════════════
        # RESULTS
        # ════════════════════════════════════════════════════
        st.divider()
        st.subheader("Analysis Results")

        # Top metrics row — confidence removed
        col1, col2, col3 = st.columns(3)

        with col1:
            st.metric("Contract Type", pred_label)
        with col2:
            st.metric("Risk Level", level)
        with col3:
            st.metric("Risk Score", score)

        st.divider()

        # Two columns for details
        left, right = st.columns(2)

        # ── Left: Classification + Clauses ───────────────────
        with left:
            st.subheader("Classification")
            st.write(f"**Detected Contract Type:** {pred_label}")

            st.divider()
            st.subheader("Extracted Clauses")

            # Map each clause name to a CSS theme class
            clause_theme = {
                "Contracting Parties": "ch-parties",
                "Effective Date":      "ch-date",
                "Payment Terms":       "ch-payment",
                "Termination":         "ch-term",
                "Liability Limit":     "ch-liability",
                "Jurisdiction":        "ch-juris",
                "Renewal":             "ch-renewal",
            }

            for clause_name, values in clauses.items():
                theme = clause_theme.get(clause_name, "ch-parties")
                if values:
                    html = f'<div class="{theme}">'
                    html += (
                        f'<div class="clause-section-header">'
                        f'{clause_name} &nbsp;'
                        f'<span style="font-weight:400;font-size:0.8em;">'
                        f'({len(values)} found)</span></div>'
                    )
                    for v in values:
                        html += (
                            f'<div class="clause-value">'
                            f'{v[:200]}</div>'
                        )
                    html += '</div>'
                    st.markdown(html, unsafe_allow_html=True)
                else:
                    st.markdown(
                        f'<div class="clause-missing">'
                        f'<b>{clause_name}</b>: Not found'
                        f'</div>',
                        unsafe_allow_html=True
                    )

        # ── Right: Risk Assessment ────────────────────────────
        with right:
            st.subheader("Risk Assessment")

            # Risk level banner — colors fully preserved
            st.markdown(
                f"<div style='background:{color}22; "
                f"border:2px solid {color}; "
                f"padding:15px; border-radius:8px; "
                f"text-align:center; margin-bottom:15px;'>"
                f"<h2 style='color:{color}; margin:0;'>"
                f"{level}</h2>"
                f"<p style='margin:5px 0 0 0;'>"
                f"Risk Score: {score} | "
                f"Total Flags: {len(risks)}</p>"
                f"</div>",
                unsafe_allow_html=True
            )

            # Risk summary counts
            high   = sum(1 for r in risks
                        if r["severity"] == "HIGH")
            medium = sum(1 for r in risks
                        if r["severity"] == "MEDIUM")
            low    = sum(1 for r in risks
                        if r["severity"] == "LOW")

            c1, c2, c3 = st.columns(3)
            c1.metric("High",   high)
            c2.metric("Medium", medium)
            c3.metric("Low",    low)

            st.write("**Risk Details:**")

            if not risks:
                st.success("No significant risks detected!")
            else:
                for risk in risks:
                    sev = risk["severity"]
                    css = (
                        "risk-high"        if sev == "HIGH"
                        else "risk-medium" if sev == "MEDIUM"
                        else "risk-low"
                    )
                    context_html = ""
                    if risk.get("context"):
                        context_html = (
                            f"<br><small><i>"
                            f"...{risk['context'][:100]}..."
                            f"</i></small>"
                        )

                    st.markdown(
                        f'<div class="{css}">'
                        f'<b>{risk["name"]}</b> '
                        f'[{sev}]<br>'
                        f'<small>{risk["description"]}</small>'
                        f'{context_html}'
                        f'</div>',
                        unsafe_allow_html=True
                    )

            st.divider()

            # Recommendations
            st.subheader("Recommendations")

            if level == "HIGH RISK":
                st.error(
                    "**High Risk Contract** — Legal review "
                    "strongly recommended before signing."
                )
            elif level == "MEDIUM RISK":
                st.warning(
                    "**Medium Risk Contract** — Review "
                    "flagged clauses carefully before signing."
                )
            else:
                st.success(
                    "**Low Risk Contract** — Standard review "
                    "recommended before signing."
                )

            missing = [r for r in risks
                      if "Missing" in r["name"]]
            if missing:
                st.write("**Missing clauses to address:**")
                for m in missing:
                    st.write(f"  - {m['name']}")

        # Download results
        st.divider()
        st.subheader("Export Results")

        report_lines = [
            "CONTRACT ANALYSIS REPORT",
            "=" * 50,
            f"Contract Type:   {pred_label}",
            f"Risk Level:      {level}",
            f"Risk Score:      {score}",
            "",
            "EXTRACTED CLAUSES",
            "-" * 30,
        ]
        for clause_name, values in clauses.items():
            report_lines.append(f"\n{clause_name}:")
            if values:
                for v in values:
                    report_lines.append(f"  - {v[:200]}")
            else:
                report_lines.append("  - Not found")

        report_lines += [
            "",
            "RISK FLAGS",
            "-" * 30,
        ]
        for risk in risks:
            report_lines.append(
                f"[{risk['severity']}] {risk['name']}: "
                f"{risk['description']}"
            )

        report_text = "\n".join(report_lines)

        st.download_button(
            label="Download Report (.txt)",
            data=report_text,
            file_name="contract_analysis_report.txt",
            mime="text/plain"
        )

    # ── Sidebar: Model Info ───────────────────────────────────
    with st.sidebar:
        st.header("About")
        st.markdown("""
        **Contract Analyzer** uses deep learning
        to analyze legal contracts.

        **Models Used:**
        - LSTM (80.83% CV accuracy)
        - NLTK NER
        - Regex pattern matching

        **Contract Types:**
        - Alliance & Cooperation
        - Distribution & Sales
        - License & IP
        - Services & Outsourcing
        - Development & Manufacturing

        **What it detects:**
        - Contract classification
        - Key clause extraction
        - Risk flags & scoring
        """)

        st.divider()
        st.header("Model Performance")
        perf_data = {
            "Model": ["ANN", "CNN", "LSTM"],
            "Accuracy": ["59.72%", "59.72%", "80.83%"]
        }
        import pandas as pd
        st.table(pd.DataFrame(perf_data))


if __name__ == "__main__":
    main()
