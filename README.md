# E-Commerce Sales & Customer Behavior Analysis

## 📌 Project Overview
This project analyzes the "Online Retail" dataset from the UCI Machine Learning Repository to identify customer segments and sales trends. The goal is to optimize marketing strategies by grouping customers based on their purchasing behavior using **RFM Analysis** and **K-Means Clustering**.

## 📂 Project Structure
- `data/`: Contains the raw dataset and the processed CSV results.
- `images/`: Generated charts and visualizations (saved automatically).
- `analysis.py`: Main Python script for data processing and modeling.

## 📊 Key Features
- **Real-World Data Cleaning**: Handled missing Customer IDs, negative quantities (returns), and date formatting.
- **Automated Visualization**: All plots are automatically saved to the `images/` folder.
- **RFM Segmentation**: Classified customers based on Recency, Frequency, and Monetary value.
- **Unsupervised Learning**: Applied K-Means clustering to discover hidden patterns in customer data.

## 🛠️ Tech Stack
- **Python**: Pandas, NumPy
- **Visualization**: Matplotlib, Seaborn
- **Machine Learning**: Scikit-Learn (K-Means)

## 🚀 How to Run
1. Clone the repository.
2. Download the dataset **`Online Retail.xlsx`** from [UCI Machine Learning Repository](https://archive.ics.uci.edu/dataset/352/online+retail+ii).
3. Place the file inside the `data/` folder.
4. Install dependencies: 
   ```bash
   pip install -r requirements.txt
