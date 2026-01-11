import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
from datetime import datetime
import os

sns.set(style="whitegrid")

os.makedirs('images', exist_ok=True)
os.makedirs('data', exist_ok=True)

def load_data(filepath):
    """
    Loads the Online Retail dataset.
    Checks if file exists to prevent crashing.
    """
    if not os.path.exists(filepath):
        print(f"❌ Error: File '{filepath}' not found.")
        print("Please download 'Online Retail.xlsx' from UCI Repository and place it in the 'data' folder.")
        return None

    print(f"⏳ Loading dataset from {filepath}... (This might take a moment)")
    df = pd.read_excel(filepath, engine='openpyxl')
    print(f"✅ Data Loaded Successfully! Shape: {df.shape}")
    return df

def clean_data(df):
    """
    Real-world data cleaning:
    - Standardizes column names (InvoiceNo -> Invoice, UnitPrice -> Price)
    - Removes nulls and negative quantities
    """
    df.rename(columns={
        'InvoiceNo': 'Invoice', 
        'CustomerID': 'Customer ID',
        'UnitPrice': 'Price'
    }, inplace=True)

    if 'Customer ID' not in df.columns:
        df.columns = df.columns.str.strip()
        if 'CustomerID' in df.columns:
            df.rename(columns={'CustomerID': 'Customer ID'}, inplace=True)
            
    if 'Customer ID' not in df.columns:
        raise KeyError("❌ Error: Could not find 'Customer ID' column. Please check the Excel file headers.")

    df.dropna(subset=['Customer ID'], inplace=True)
    
    df = df[~df['Invoice'].astype(str).str.startswith('C')]
    df = df[df['Quantity'] > 0]
    
    df['Total_Sales'] = df['Quantity'] * df['Price']
    
    df['InvoiceDate'] = pd.to_datetime(df['InvoiceDate'])
    
    print(f"✅ Data Cleaned. Final Shape: {df.shape}")
    return df

def perform_eda(df):
    """Generates insights and saves plots to 'images/' folder."""
    
    df['Month'] = df['InvoiceDate'].dt.to_period('M')
    monthly_sales = df.groupby('Month')['Total_Sales'].sum()
    
    plt.figure(figsize=(12, 6))
    monthly_sales.index = monthly_sales.index.astype(str)
    sns.lineplot(x=monthly_sales.index, y=monthly_sales.values, marker='o', color='teal')
    plt.title('Monthly Sales Trend', fontsize=14)
    plt.ylabel('Revenue (£)')
    plt.xticks(rotation=45)
    plt.tight_layout()
    plt.savefig('images/01_monthly_sales_trend.png')
    print("📸 Saved: images/01_monthly_sales_trend.png")
    plt.close()

    top_products = df.groupby('Description')['Quantity'].sum().sort_values(ascending=False).head(10)
    
    plt.figure(figsize=(10, 6))
    sns.barplot(y=top_products.index, x=top_products.values, hue=top_products.index, legend=False, palette='viridis')
    plt.title('Top 10 Best Selling Products', fontsize=14)
    plt.xlabel('Quantity Sold')
    plt.tight_layout()
    plt.savefig('images/02_top_products.png')
    print("📸 Saved: images/02_top_products.png")
    plt.close()

def perform_rfm_analysis(df):
    """
    Calculates RFM metrics.
    """
    snapshot_date = df['InvoiceDate'].max() + pd.Timedelta(days=1)
    
    rfm = df.groupby('Customer ID').agg({
        'InvoiceDate': lambda x: (snapshot_date - x.max()).days,
        'Invoice': 'nunique',
        'Total_Sales': 'sum'
    }).rename(columns={'InvoiceDate': 'Recency', 'Invoice': 'Frequency', 'Total_Sales': 'Monetary'})
    
    return rfm

def cluster_customers(rfm_df):
    """
    Segments customers using K-Means and saves the scatterplot.
    """
    rfm_log = np.log1p(rfm_df)
    
    scaler = StandardScaler()
    rfm_scaled = scaler.fit_transform(rfm_log)
    
    kmeans = KMeans(n_clusters=3, random_state=42, n_init=10)
    rfm_df['Cluster'] = kmeans.fit_predict(rfm_scaled)
    
    plt.figure(figsize=(10, 6))
    sns.scatterplot(data=rfm_df, x='Recency', y='Frequency', hue='Cluster', palette='deep', s=60)
    plt.title('Customer Segments: Recency vs Frequency', fontsize=14)
    plt.tight_layout()
    plt.savefig('images/03_customer_segments.png')
    print("📸 Saved: images/03_customer_segments.png")
    plt.close()
    
    return rfm_df

if __name__ == "__main__":
    file_path = 'data/Online Retail.xlsx'
    
    df = load_data(file_path)
    
    if df is not None:
        df = clean_data(df)
        
        perform_eda(df)
        
        rfm_df = perform_rfm_analysis(df)
        print("\nTop 5 Customers by RFM:\n", rfm_df.head())
        
        segmented_df = cluster_customers(rfm_df)
        
        rfm_df.to_csv('data/customer_segments.csv')
        print("✅ Analysis Complete! Results saved to 'data/customer_segments.csv'")