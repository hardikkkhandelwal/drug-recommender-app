import streamlit as st
import pandas as pd
import joblib

st.set_page_config(page_title="Pharma Saver", layout="centered")

@st.cache_resource
def load_assets():
    df = pd.read_parquet("medicine_data.parquet")
    tools = joblib.load("model_tools.pkl")
    return df, tools["model"], tools["mlb"]

st.title("💊 Pharma Alternative Finder")

try:
    df, model, mlb = load_assets()
    
    # Use a selectbox with a placeholder to keep it clean
    brand_name = st.selectbox("Search your medicine brand:", [""] + sorted(df["brand_name"].unique().tolist()))

    if brand_name:
        p = df[df["brand_name"] == brand_name].iloc[0]
        
        # UI Columns for the selected drug
        col1, col2 = st.columns(2)
        col1.metric("Selected Brand", brand_name)
        col2.metric("Price", f"₹{p['price_inr']}")
        
        st.write(f"**Ingredients:** {', '.join(p['ai_names'])}")
        
        if st.button("Find Cheaper Alternatives"):
            target_set = set(p["ai_names"])
            
            # Filter matches
            results = df[
                (df["ai_names"].apply(set) == target_set) & 
                (df["dosage_form"] == p["dosage_form"]) &
                (df["brand_name"] != brand_name)
            ].sort_values("price_inr")

            if not results.empty:
                st.success(f"Found {len(results)} alternatives!")
                st.dataframe(
                    results[["brand_name", "manufacturer", "price_inr"]],
                    hide_index=True,
                    use_container_width=True
                )
            else:
                st.warning("No cheaper alternatives found in the database.")

except Exception as e:
    st.error("Please ensure medicine_data.parquet and model_tools.pkl are uploaded to GitHub.")