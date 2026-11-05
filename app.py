
import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA

st.set_page_config(page_title="App User Project - Vetri", layout="wide")

st.title("📱 APP USER BEHAVIOR SEGMENTATION")
st.markdown("**By Vetri | GUVI HCL Project | 4 Clusters using K-Means**")

st.sidebar.header("Upload Dataset")
uploaded_file = st.sidebar.file_uploader("Choose CSV file", type="csv")

if uploaded_file is not None:
    df = pd.read_csv(uploaded_file)
    st.success(f"File uploaded! {df.shape[0]} users, {df.shape[1]} columns")
    
    st.subheader("1. Data Preview")
    st.dataframe(df.head())
    
    # Feature selection
    features = ['sessions_per_week', 'avg_session_duration_min', 'daily_active_minutes', 'engagement_score']
    available_features = [f for f in features if f in df.columns]
    
    if len(available_features) >= 2:
        X = df[available_features]
        X = X.fillna(X.median())
        
        # Scaling
        scaler = StandardScaler()
        X_scaled = scaler.fit_transform(X)
        
        # KMeans
        kmeans = KMeans(n_clusters=4, random_state=42, n_init=10)
        df['cluster'] = kmeans.fit_predict(X_scaled)
        
        st.subheader("2. Cluster Counts")
        st.write(df['cluster'].value_counts())
        
        st.subheader("3. Business Insights - 4 User Groups")
        col1, col2, col3, col4 = st.columns(4)
        with col1:
            st.metric("Cluster 0", "High Value", "Loyal Users")
        with col2:
            st.metric("Cluster 1", "Moderate", "Regular")
        with col3:
            st.metric("Cluster 2", "At-Risk", "Need Attention")
        with col4:
            st.metric("Cluster 3", "Occasional", "Re-engage")
        
        # PCA Plot
        st.subheader("4. PCA Visualization - 4 Colour Clusters")
        pca = PCA(n_components=2)
        pca_data = pca.fit_transform(X_scaled)
        
        fig, ax = plt.subplots(figsize=(10,6))
        sns.scatterplot(x=pca_data[:,0], y=pca_data[:,1], hue=df['cluster'], palette='Set1', ax=ax, s=50)
        plt.title("App Users - 4 Clusters (Red, Blue, Green, Purple)")
        st.pyplot(fig)
        
        st.subheader("5. Business Actions")
        st.table(pd.DataFrame({
            "Cluster": ["0 - High", "1 - Moderate", "2 - Low/At-Risk", "3 - Occasional"],
            "Action": ["Loyalty Offers, Premium Features", "Personalized Recommendations", "Retention Campaign, Discounts", "Re-engagement Push Notifications"]
        }))
        
        st.balloons()
        st.success("Project Dashboard Ready! Share this link with teacher!")
else:
    st.info("👈 Left side la un app_user_behavior_dataset.csv file-a upload pannu")
    st.image("https://via.placeholder.com/600x300?text=Upload+CSV+to+see+4+Clusters", caption="Upload CSV to see magic")

