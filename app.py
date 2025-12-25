import streamlit as st
import pandas as pd
import joblib

st.set_page_config(page_title="Pharma Saver", layout="centered")

@st.cache_resource
def load_assets():
    # Loading the compressed parquet data and the model tools
    df = pd.read_parquet("medicine_data.parquet")
    tools = joblib.load("model_tools.pkl")
    return df, tools["model"], tools["mlb"]

st.title("💊 Pharma Alternative Finder")

try:
    df, model, mlb = load_assets()
    
    # --- TEXT INPUT INSTEAD OF SELECTBOX ---
    search_query = st.text_input("Enter Brand Name (e.g., Crocin, Dolo):", "").strip()

    if search_query:
        # Check if the brand exists (case-insensitive search)
        match = df[df["brand_name"].str.lower() == search_query.lower()]
        
        if not match.empty:
            p = match.iloc[0]
            brand_name = p["brand_name"]
            
            # Display Selected Drug Info
            st.markdown("---")
            col1, col2 = st.columns(2)
            col1.metric("Brand Found", brand_name)
            col2.metric("Price", f"₹{p['price_inr']}")
            
            st.write(f"**Ingredients:** {', '.join(p['ai_names'])}")
            st.write(f"**Dosage Form:** {p['dosage_form']}")
            
            if st.button("Find Cheaper Alternatives"):
                target_set = set(p["ai_names"])
                
                # Filter for alternatives with identical ingredients and form
                results = df[
                    (df["ai_names"].apply(set) == target_set) & 
                    (df["dosage_form"] == p["dosage_form"]) &
                    (df["brand_name"].str.lower() != search_query.lower())
                ].sort_values("price_inr")

                if not results.empty:
                    st.success(f"Found {len(results)} alternatives!")
                    st.dataframe(
                        results[["brand_name", "manufacturer", "price_inr"]],
                        hide_index=True,
                        use_container_width=True
                    )
                else:
                    st.warning("No cheaper alternatives found for this exact composition.")
        else:
            st.error(f"Brand '{search_query}' not found. Please check the spelling.")

except Exception as e:
    st.error("Missing Files: Please ensure medicine_data.parquet and model_tools.pkl are pushed to GitHub.")