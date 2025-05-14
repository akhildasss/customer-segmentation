import streamlit as st
import pandas as pd
import numpy as np
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
import matplotlib.pyplot as plt
import seaborn as sns

# Load data
def load_data():
    df = pd.read_csv("data.csv", encoding='ISO-8859-1')
    df.dropna(subset=['CustomerID'], inplace=True)
    df = df[df['Quantity'] > 0]
    df['TotalPrice'] = df['Quantity'] * df['UnitPrice']
    df['InvoiceDate'] = pd.to_datetime(df['InvoiceDate'])
    return df

# RFM Calculation
def compute_rfm(df):
    snapshot_date = df['InvoiceDate'].max() + pd.Timedelta(days=1)
    rfm = df.groupby('CustomerID').agg({
        'InvoiceDate': lambda x: (snapshot_date - x.max()).days,
        'InvoiceNo': 'nunique',
        'TotalPrice': 'sum'
    })
    rfm.columns = ['Recency', 'Frequency', 'Monetary']
    return rfm

# Clustering
def segment_customers(rfm):
    scaler = StandardScaler()
    rfm_scaled = scaler.fit_transform(rfm)
    kmeans = KMeans(n_clusters=4, random_state=42)
    rfm['Segment'] = kmeans.fit_predict(rfm_scaled)
    return rfm

# Recommendations
def get_top_products(df, rfm):
    top_items = df.groupby('Description')['TotalPrice'].sum().sort_values(ascending=False).head(10)
    recommendations = {}
    for segment in rfm['Segment'].unique():
        segment_customers = rfm[rfm['Segment'] == segment].index
        seg_df = df[df['CustomerID'].isin(segment_customers)]
        seg_top_items = seg_df.groupby('Description')['TotalPrice'].sum().sort_values(ascending=False).head(5)
        recommendations[segment] = seg_top_items.index.tolist()
    return top_items.index.tolist(), recommendations

# Streamlit UI
st.title("Customer Segmentation & Recommendation System")

st.markdown("""
This app performs **customer segmentation** using RFM and **recommends products** based on clusters.
""")

df = load_data()
st.subheader("Raw Data")
st.dataframe(df.head())

rfm = compute_rfm(df)
rfn_segmented = segment_customers(rfm)

st.subheader("RFM Segments")
st.dataframe(rfn_segmented.reset_index())

# Visualize segments
st.subheader("Segment Distribution")
fig, ax = plt.subplots()
sns.countplot(data=rfm, x='Segment', ax=ax)
st.pyplot(fig)

# Show Recommendations
st.subheader("Top Products & Recommendations")
top_products, recommendations = get_top_products(df, rfm)

st.markdown("**Top 10 Products Overall:**")
st.write(top_products)

for segment, items in recommendations.items():
    st.markdown(f"**Segment {segment} Recommendations:**")
    st.write(items)

st.success("Customer segmentation and recommendations complete!")
