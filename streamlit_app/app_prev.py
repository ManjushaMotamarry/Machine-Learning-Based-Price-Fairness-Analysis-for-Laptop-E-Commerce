import streamlit as st
import joblib
import numpy as np
import pandas as pd
import re

st.set_page_config(page_title="Laptop Price Fairness Checker", layout="centered", page_icon="💻")

st.markdown("""
<style>
    /* Base */
    html, body, [class*="css"] {
        background-color: #0e0e0e !important;
        color: #f0f0f0 !important;
        font-family: 'Segoe UI', sans-serif;
    }
    .stApp { background-color: #0e0e0e; }

    /* Title */
    .hero {
        text-align: center;
        padding: 40px 0 10px 0;
    }
    .hero h1 {
        font-size: 2.8rem;
        font-weight: 800;
        background: linear-gradient(90deg, #4F8BF9, #a855f7, #ec4899);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        margin-bottom: 6px;
    }
    .hero p {
        color: #888;
        font-size: 1rem;
        margin-top: 0;
    }

    /* Card */
    .card {
        background: #1a1a1a;
        border: 1px solid #2a2a2a;
        border-radius: 16px;
        padding: 28px;
        margin: 20px 0;
    }

    /* Text area */
    .stTextArea textarea {
        background-color: #1a1a1a !important;
        color: #f0f0f0 !important;
        border: 1px solid #333 !important;
        border-radius: 10px !important;
        font-size: 14px !important;
    }
    .stTextArea textarea:focus {
        border-color: #4F8BF9 !important;
        box-shadow: 0 0 0 2px rgba(79,139,249,0.2) !important;
    }

    /* Number input */
    .stNumberInput input {
        background-color: #1a1a1a !important;
        color: #f0f0f0 !important;
        border: 1px solid #333 !important;
        border-radius: 10px !important;
    }

    /* Button */
    .stButton > button {
        background: linear-gradient(90deg, #4F8BF9, #a855f7);
        color: white !important;
        border: none !important;
        border-radius: 10px !important;
        padding: 0.6rem 2rem !important;
        font-size: 16px !important;
        font-weight: 700 !important;
        width: 100% !important;
        margin-top: 10px !important;
        transition: opacity 0.2s;
    }
    .stButton > button:hover { opacity: 0.85; }

    /* Metrics */
    [data-testid="metric-container"] {
        background: #1a1a1a;
        border: 1px solid #2a2a2a;
        border-radius: 12px;
        padding: 16px;
        text-align: center;
    }
    [data-testid="stMetricLabel"] { color: #888 !important; font-size: 13px !important; }
    [data-testid="stMetricValue"] { color: #f0f0f0 !important; font-size: 22px !important; font-weight: 700 !important; }

    /* Verdict box */
    .verdict-box {
        padding: 24px;
        border-radius: 16px;
        text-align: center;
        font-size: 24px;
        font-weight: 800;
        margin-top: 24px;
        letter-spacing: 0.5px;
    }
    .overpriced  { background: rgba(220,38,38,0.15); color: #f87171; border: 2px solid #f87171; }
    .underpriced { background: rgba(59,130,246,0.15); color: #60a5fa; border: 2px solid #60a5fa; }
    .fair        { background: rgba(34,197,94,0.15);  color: #4ade80; border: 2px solid #4ade80; }

    /* Expander */
    .streamlit-expanderHeader {
        background-color: #1a1a1a !important;
        color: #aaa !important;
        border-radius: 10px !important;
    }
    .streamlit-expanderContent {
        background-color: #141414 !important;
        border: 1px solid #2a2a2a !important;
        border-radius: 0 0 10px 10px !important;
    }

    /* Dataframe */
    .stDataFrame { border-radius: 10px; overflow: hidden; }

    /* Divider */
    hr { border-color: #2a2a2a !important; }

    /* Labels */
    label, .stTextArea label, .stNumberInput label {
        color: #aaa !important;
        font-size: 14px !important;
        font-weight: 500 !important;
    }

    /* Spinner */
    .stSpinner > div { border-top-color: #4F8BF9 !important; }
</style>
""", unsafe_allow_html=True)

# ── Constants ─────────────────────────────────────────────────────────────────
FEATURES = [
    'Brand', 'RAM_type', 'Storage_type', 'Graphics_integreted',
    'Touch_screen', 'Operating_system', 'GPU_tier', 'RAM_tier',
    'Storage_tier', 'Processor_gen_tier', 'Display_tier', 'ppi_tier',
    'is_gaming', 'brand_tier', 'performance_score', 'display_quality'
]

LABEL_MAPS = {
    'Brand':              {'Apple': 0, 'Asus': 1, 'Dell': 2, 'HP': 3, 'Infinix': 4, 'Lenovo': 5, 'MSI': 6, 'Samsung': 7, 'Other': 8},
    'RAM_type':           {'DDR5': 0, 'LPDDR4': 1, 'LPDDR4X': 2, 'LPDDR5': 3, 'LPDDR5X': 4, 'Other': 5},
    'Storage_type':       {'SSD': 0, 'Hard Disk & SSD': 1},
    'Operating_system':   {'DOS OS': 0, 'Mac OS': 1, 'Windows 10 OS': 2, 'Windows 11 OS': 3, 'Other': 4},
    'GPU_tier':           {'Integrated': 0, 'Budget_Mobile': 1, 'GTX_MX': 2, 'RTX_2000': 3, 'RTX_3000': 4, 'RTX_4000': 5, 'Apple_Silicon': 6, 'Other': 7},
    'RAM_tier':           {'Low': 0, 'Mid': 1, 'High': 2},
    'Storage_tier':       {'Low': 0, 'Mid': 1, 'High': 2},
    'Processor_gen_tier': {'Low': 0, 'Mid': 1, 'Modern': 2},
    'Display_tier':       {'Small': 0, 'Mid': 1, 'Large': 2, 'XLarge': 3},
    'ppi_tier':           {'Low': 0, 'Standard': 1, 'High': 2, 'Very_High': 3, 'Ultra': 4},
    'brand_tier':         {'Budget': 0, 'Mid': 1, 'Mid_High': 2, 'Premium': 3},
}

FAIRNESS_THRESHOLD = 15.45

# ── Model ─────────────────────────────────────────────────────────────────────
@st.cache_resource
def load_model():
    return joblib.load('xgb_tuned_model.pkl')

model = load_model()

# ── Feature Extraction ────────────────────────────────────────────────────────
def extract_features(text: str) -> dict:
    text_lower = text.lower()

    brand = 'Other'
    for b in ['Apple', 'Asus', 'Dell', 'HP', 'Infinix', 'Lenovo', 'MSI', 'Samsung']:
        if b.lower() in text_lower:
            brand = b
            break

    ram_match = re.search(r'(\d+)\s*gb\s*(ddr5|lpddr4x|lpddr4|lpddr5x|lpddr5)?', text_lower)
    ram_gb = int(ram_match.group(1)) if ram_match else 8
    ram_type_raw = ram_match.group(2).upper() if ram_match and ram_match.group(2) else 'Other'
    ram_tier = 'Low' if ram_gb <= 8 else ('Mid' if ram_gb <= 16 else 'High')

    storage_match = re.search(r'(\d+)\s*(tb|gb)\s*(ssd|hdd|hard disk)?', text_lower)
    storage_gb = int(storage_match.group(1)) * (1000 if storage_match.group(2) == 'tb' else 1) if storage_match else 256
    storage_tier = 'Low' if storage_gb <= 256 else ('Mid' if storage_gb <= 512 else 'High')
    storage_type = 'Hard Disk & SSD' if 'hdd' in text_lower or 'hard disk' in text_lower else 'SSD'

    display_match = re.search(r'(\d+\.?\d*)\s*[-\s]?inch', text_lower)
    display_size = float(display_match.group(1)) if display_match else 15.0
    display_tier = 'Small' if display_size < 13 else ('Mid' if display_size < 15 else ('Large' if display_size < 17 else 'XLarge'))

    if 'retina' in text_lower or '4k' in text_lower or 'oled' in text_lower:
        ppi_tier = 'Ultra'
    elif '2k' in text_lower or 'qhd' in text_lower:
        ppi_tier = 'Very_High'
    elif 'fhd' in text_lower or '1080' in text_lower:
        ppi_tier = 'High'
    else:
        ppi_tier = 'Standard'

    if 'rtx 40' in text_lower or 'rtx40' in text_lower:
        gpu_tier = 'RTX_4000'
    elif 'rtx 30' in text_lower or 'rtx30' in text_lower:
        gpu_tier = 'RTX_3000'
    elif 'rtx 20' in text_lower or 'rtx20' in text_lower:
        gpu_tier = 'RTX_2000'
    elif 'gtx' in text_lower or 'mx' in text_lower:
        gpu_tier = 'GTX_MX'
    elif any(x in text_lower for x in ['m1', 'm2', 'm3', 'm4']):
        gpu_tier = 'Apple_Silicon'
    elif 'integrated' in text_lower or 'iris' in text_lower or 'uhd' in text_lower:
        gpu_tier = 'Integrated'
    else:
        gpu_tier = 'Budget_Mobile'

    graphics_integrated = 1 if gpu_tier in ['Integrated', 'Apple_Silicon'] else 0

    if 'macos' in text_lower or 'mac os' in text_lower:
        os = 'Mac OS'
    elif 'windows 11' in text_lower:
        os = 'Windows 11 OS'
    elif 'windows 10' in text_lower:
        os = 'Windows 10 OS'
    elif 'dos' in text_lower:
        os = 'DOS OS'
    else:
        os = 'Other'

    gen_match = re.search(r'(\d+)th\s*gen', text_lower)
    if gen_match:
        gen = int(gen_match.group(1))
        proc_gen_tier = 'Low' if gen <= 8 else ('Mid' if gen <= 11 else 'Modern')
    elif any(x in text_lower for x in ['m1', 'm2', 'm3', 'm4']):
        proc_gen_tier = 'Modern'
    else:
        proc_gen_tier = 'Mid'

    if brand.lower() in ['apple', 'msi']:
        brand_tier = 'Premium'
    elif brand.lower() in ['dell', 'asus', 'samsung']:
        brand_tier = 'Mid_High'
    elif brand.lower() in ['hp', 'lenovo']:
        brand_tier = 'Mid'
    else:
        brand_tier = 'Budget'

    thread_match = re.search(r'(\d+)\s*thread', text_lower)
    threads = int(thread_match.group(1)) if thread_match else 8
    gen_num = int(gen_match.group(1)) if gen_match else 11
    performance_score = threads * gen_num
    ppi_map = {'Low': 100, 'Standard': 150, 'High': 200, 'Very_High': 250, 'Ultra': 300}
    display_quality = round(display_size * ppi_map[ppi_tier] / 100, 2)

    return {
        'Brand': brand, 'RAM_type': ram_type_raw, 'Storage_type': storage_type,
        'Graphics_integreted': graphics_integrated, 'Touch_screen': 1 if 'touch' in text_lower else 0,
        'Operating_system': os, 'GPU_tier': gpu_tier, 'RAM_tier': ram_tier,
        'Storage_tier': storage_tier, 'Processor_gen_tier': proc_gen_tier,
        'Display_tier': display_tier, 'ppi_tier': ppi_tier,
        'is_gaming': 1 if 'gaming' in text_lower else 0,
        'brand_tier': brand_tier, 'performance_score': performance_score,
        'display_quality': display_quality
    }

def encode_features(features: dict) -> pd.DataFrame:
    encoded = {col: LABEL_MAPS[col].get(features.get(col), 0) if col in LABEL_MAPS else features.get(col) for col in FEATURES}
    return pd.DataFrame([encoded])

# ── UI ────────────────────────────────────────────────────────────────────────
st.markdown("""
<div class="hero">
    <h1>💻 Laptop Price Fairness</h1>
    <p>Paste a product description and find out if you're getting a fair deal.</p>
</div>
""", unsafe_allow_html=True)

st.divider()

product_text = st.text_area("Product Description", height=180,
    placeholder="e.g. Dell XPS 15, Intel Core i7 13th Gen, 16GB DDR5 RAM, 512GB SSD, NVIDIA RTX 4060, 15.6 inch FHD, Windows 11...")

actual_price = st.number_input("Listed Price (INR)", min_value=0, step=1000)

if st.button("Check Fairness ✦"):
    if not product_text.strip():
        st.warning("Please paste a product description.")
    elif actual_price == 0:
        st.warning("Please enter the listed price.")
    else:
        with st.spinner("Analyzing specs and computing fairness..."):
            features = extract_features(product_text)
            X = encode_features(features)
            log_pred = model.predict(X)[0]
            predicted_price = np.exp(log_pred)
            price_diff_pct = (np.exp(np.log(actual_price) - log_pred) - 1) * 100

            if price_diff_pct > FAIRNESS_THRESHOLD:
                flag, css_class, icon = "Overpriced", "overpriced", "🔴"
            elif price_diff_pct < -FAIRNESS_THRESHOLD:
                flag, css_class, icon = "Underpriced", "underpriced", "🔵"
            else:
                flag, css_class, icon = "Fairly Priced", "fair", "🟢"

        st.divider()
        col1, col2, col3 = st.columns(3)
        col1.metric("Listed Price", f"Rs {actual_price:,.0f}")
        col2.metric("Predicted Price", f"Rs {predicted_price:,.0f}")
        col3.metric("Price Difference", f"{price_diff_pct:+.1f}%")

        st.markdown(f"""
        <div class="verdict-box {css_class}">
            {icon} &nbsp; {flag}
        </div>
        """, unsafe_allow_html=True)

        with st.expander("View Extracted Features"):
            feat_df = pd.DataFrame(features.items(), columns=["Feature", "Value"])
            st.dataframe(feat_df, use_container_width=True, hide_index=True)