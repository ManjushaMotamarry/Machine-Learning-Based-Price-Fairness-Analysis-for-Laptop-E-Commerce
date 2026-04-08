import streamlit as st
import joblib
import numpy as np
import pandas as pd
import re

# Load model
model = joblib.load('xgb_tuned_model.pkl')

FEATURES = [
    'Brand', 'RAM_type', 'Storage_type', 'Graphics_integreted',
    'Touch_screen', 'Operating_system', 'GPU_tier', 'RAM_tier',
    'Storage_tier', 'Processor_gen_tier', 'Display_tier', 'ppi_tier',
    'is_gaming', 'brand_tier', 'performance_score', 'display_quality'
]

print("Model loaded!")

LABEL_MAPS = {
    'Brand':            {'Apple': 0, 'Asus': 1, 'Dell': 2, 'HP': 3, 'Infinix': 4, 'Lenovo': 5, 'MSI': 6, 'Samsung': 7, 'Other': 8},
    'RAM_type':         {'DDR5': 0, 'LPDDR4': 1, 'LPDDR4X': 2, 'LPDDR5': 3, 'LPDDR5X': 4, 'Other': 5},
    'Storage_type':     {'SSD': 0, 'Hard Disk & SSD': 1},
    'Operating_system': {'DOS OS': 0, 'Mac OS': 1, 'Windows 10 OS': 2, 'Windows 11 OS': 3, 'Other': 4},
    'GPU_tier':         {'Integrated': 0, 'Budget_Mobile': 1, 'GTX_MX': 2, 'RTX_2000': 3, 'RTX_3000': 4, 'RTX_4000': 5, 'Apple_Silicon': 6, 'Other': 7},
    'RAM_tier':         {'Low': 0, 'Mid': 1, 'High': 2},
    'Storage_tier':     {'Low': 0, 'Mid': 1, 'High': 2},
    'Processor_gen_tier': {'Low': 0, 'Mid': 1, 'Modern': 2},
    'Display_tier':     {'Small': 0, 'Mid': 1, 'Large': 2, 'XLarge': 3},
    'ppi_tier':         {'Low': 0, 'Standard': 1, 'High': 2, 'Very_High': 3, 'Ultra': 4},
    'brand_tier':       {'Budget': 0, 'Mid': 1, 'Mid_High': 2, 'Premium': 3},
}
@st.cache_resource
def load_model():
    return joblib.load('xgb_tuned_model.pkl')

model = load_model()

FAIRNESS_THRESHOLD = 15.45

def extract_features(text: str) -> dict:
    text_lower = text.lower()

    # Brand
    brand = 'Other'
    for b in ['Apple', 'Asus', 'Dell', 'HP', 'Infinix', 'Lenovo', 'MSI', 'Samsung']:
        if b.lower() in text_lower:
            brand = b
            break

    # RAM amount and type
    ram_match = re.search(r'(\d+)\s*gb\s*(ddr5|lpddr4x|lpddr4|lpddr5x|lpddr5)?', text_lower)
    ram_gb = int(ram_match.group(1)) if ram_match else 8
    ram_type_raw = ram_match.group(2).upper() if ram_match and ram_match.group(2) else 'Other'
    ram_tier = 'Low' if ram_gb <= 8 else ('Mid' if ram_gb <= 16 else 'High')

    # Storage
    storage_match = re.search(r'(\d+)\s*(tb|gb)\s*(ssd|hdd|hard disk)?', text_lower)
    if storage_match:
        storage_gb = int(storage_match.group(1)) * (1000 if storage_match.group(2) == 'tb' else 1)
    else:
        storage_gb = 256
    storage_tier = 'Low' if storage_gb <= 256 else ('Mid' if storage_gb <= 512 else 'High')
    storage_type = 'Hard Disk & SSD' if 'hdd' in text_lower or 'hard disk' in text_lower else 'SSD'

    # Display
    display_match = re.search(r'(\d+\.?\d*)\s*[-\s]?inch', text_lower)
    display_size = float(display_match.group(1)) if display_match else 15.0
    display_tier = 'Small' if display_size < 13 else ('Mid' if display_size < 15 else ('Large' if display_size < 17 else 'XLarge'))

    # PPI tier (approximate from resolution keywords)
    if 'retina' in text_lower or '4k' in text_lower or 'oled' in text_lower:
        ppi_tier = 'Ultra'
    elif '2k' in text_lower or 'qhd' in text_lower:
        ppi_tier = 'Very_High'
    elif 'fhd' in text_lower or '1080' in text_lower:
        ppi_tier = 'High'
    else:
        ppi_tier = 'Standard'

    # GPU tier
    if 'rtx 40' in text_lower or 'rtx40' in text_lower:
        gpu_tier = 'RTX_4000'
    elif 'rtx 30' in text_lower or 'rtx30' in text_lower:
        gpu_tier = 'RTX_3000'
    elif 'rtx 20' in text_lower or 'rtx20' in text_lower:
        gpu_tier = 'RTX_2000'
    elif 'gtx' in text_lower or 'mx' in text_lower:
        gpu_tier = 'GTX_MX'
    elif 'm1' in text_lower or 'm2' in text_lower or 'm3' in text_lower or 'm4' in text_lower:
        gpu_tier = 'Apple_Silicon'
    elif 'integrated' in text_lower or 'iris' in text_lower or 'uhd' in text_lower:
        gpu_tier = 'Integrated'
    else:
        gpu_tier = 'Budget_Mobile'

    # Graphics integrated
    graphics_integrated = 1 if gpu_tier in ['Integrated', 'Apple_Silicon'] else 0

    # OS
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

    # Processor gen tier
    gen_match = re.search(r'(\d+)th\s*gen', text_lower)
    if gen_match:
        gen = int(gen_match.group(1))
        proc_gen_tier = 'Low' if gen <= 8 else ('Mid' if gen <= 11 else 'Modern')
    elif any(x in text_lower for x in ['m1', 'm2', 'm3', 'm4']):
        proc_gen_tier = 'Modern'
    else:
        proc_gen_tier = 'Mid'

    # Brand tier
    premium_brands = ['apple', 'msi']
    mid_high_brands = ['dell', 'asus', 'samsung']
    mid_brands = ['hp', 'lenovo']
    if brand.lower() in premium_brands:
        brand_tier = 'Premium'
    elif brand.lower() in mid_high_brands:
        brand_tier = 'Mid_High'
    elif brand.lower() in mid_brands:
        brand_tier = 'Mid'
    else:
        brand_tier = 'Budget'

    # Threads (approximate)
    thread_match = re.search(r'(\d+)\s*thread', text_lower)
    threads = int(thread_match.group(1)) if thread_match else 8
    gen_num = int(gen_match.group(1)) if gen_match else 11
    performance_score = threads * gen_num

    # Display quality
    ppi_map = {'Low': 100, 'Standard': 150, 'High': 200, 'Very_High': 250, 'Ultra': 300}
    display_quality = display_size * ppi_map[ppi_tier] / 100

    # Flags
    is_gaming = 1 if 'gaming' in text_lower else 0
    touch_screen = 1 if 'touch' in text_lower else 0

    return {
        'Brand': brand,
        'RAM_type': ram_type_raw,
        'Storage_type': storage_type,
        'Graphics_integreted': graphics_integrated,
        'Touch_screen': touch_screen,
        'Operating_system': os,
        'GPU_tier': gpu_tier,
        'RAM_tier': ram_tier,
        'Storage_tier': storage_tier,
        'Processor_gen_tier': proc_gen_tier,
        'Display_tier': display_tier,
        'ppi_tier': ppi_tier,
        'is_gaming': is_gaming,
        'brand_tier': brand_tier,
        'performance_score': performance_score,
        'display_quality': round(display_quality, 2)
    }

def encode_features(features: dict) -> pd.DataFrame:
    encoded = {}
    for col in FEATURES:
        val = features.get(col)
        if col in LABEL_MAPS:
            encoded[col] = LABEL_MAPS[col].get(val, 0)
        else:
            encoded[col] = val
    return pd.DataFrame([encoded])


st.set_page_config(page_title="Laptop Price Fairness Checker", layout="centered")
st.title("Laptop Price Fairness Checker")
st.markdown("Paste a laptop product description and find out if it is fairly priced.")

product_text = st.text_area("Paste product description here", height=200,
                             placeholder="e.g. Apple MacBook Pro 14-inch, M3 Pro chip, 18GB RAM, 512GB SSD, macOS...")

actual_price = st.number_input("Enter the listed price (in INR)", min_value=0, step=1000)

if st.button("Check Fairness"):
    if not product_text.strip():
        st.warning("Please paste a product description.")
    elif actual_price == 0:
        st.warning("Please enter the listed price.")
    else:
        with st.spinner("Analyzing..."):
            features = extract_features(product_text)
            
            st.subheader("Extracted Features")
            st.json(features)

            X = encode_features(features)
            log_pred = model.predict(X)[0]
            predicted_price = np.exp(log_pred)

            actual_log = np.log(actual_price)
            residual = actual_log - log_pred
            price_diff_pct = (np.exp(residual) - 1) * 100

            if price_diff_pct > FAIRNESS_THRESHOLD:
                flag = "Overpriced"
                color = "red"
            elif price_diff_pct < -FAIRNESS_THRESHOLD:
                flag = "Underpriced"
                color = "blue"
            else:
                flag = "Fairly Priced"
                color = "green"

            st.subheader("Results")
            col1, col2, col3 = st.columns(3)
            col1.metric("Listed Price", f"Rs {actual_price:,.0f}")
            col2.metric("Predicted Price", f"Rs {predicted_price:,.0f}")
            col3.metric("Price Difference", f"{price_diff_pct:+.1f}%")

            st.markdown(f"### Verdict: :{color}[{flag}]")