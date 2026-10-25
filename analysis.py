"""
================================================================================
ONLINE SHOPPING CUSTOMER BEHAVIOR ANALYSIS - DATA MINING PROJECT
================================================================================
Algorithms & Techniques Implemented:
  1. Data Auditing & Cleaning (Null / Missing Value Imputation & Dummy Sentinel Filtering)
  2. Exploratory Data Analysis (EDA) & Summary Statistics
  3. RFM (Recency, Frequency, Monetary) Customer Segmentation
  4. Unsupervised Machine Learning: K-Means Customer Clustering
  5. Market Basket Analysis: Association Rule Mining (Apriori / Lift Analysis)
  6. Churn & Customer Retention Profiling

Author: Data Mining & Analytics Team
Dataset: customer_shopping_data.csv
================================================================================
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
from itertools import combinations
from collections import Counter
import warnings

warnings.filterwarnings('ignore')
plt.style.use('seaborn-v0_8-whitegrid' if 'seaborn-v0_8-whitegrid' in plt.style.available else 'default')

def load_and_audit_data(filepath="customer_shopping_data.csv"):
    print("=" * 60)
    print("STEP 1: DATA INGESTION & QUALITY AUDIT (NULL / DUMMY DATA DETECTION)")
    print("=" * 60)
    
    df = pd.read_csv(filepath)
    print(f"[*] Initial Raw Dataset Dimensions: {df.shape[0]} rows, {df.shape[1]} columns")
    
    # 1. Null / Missing Value Report
    print("\n[!] Missing / Null Value Count by Column:")
    null_counts = df.isnull().sum()
    for col, cnt in null_counts.items():
        if cnt > 0:
            pct = (cnt / len(df)) * 100
            print(f"    - {col}: {cnt} missing ({pct:.1f}%)")
    
    # 2. Dummy / Sentinel Values Detection
    # Look for sentinels: negative amounts, age 999 or -99, placeholder IDs
    dummy_ids = df[df['Customer_ID'].str.contains('DUMMY|TEST|DEBUG|GUEST_0000', case=False, na=False)]
    corrupted_amounts = df[pd.to_numeric(df['Purchase_Amount'], errors='coerce') <= 0]
    invalid_ages = df[
        (pd.to_numeric(df['Age'], errors='coerce') < 18) | 
        (pd.to_numeric(df['Age'], errors='coerce') > 100)
    ]
    duplicate_txns = df.duplicated(subset=['Transaction_ID'], keep='first').sum()

    print("\n[!] Dummy / Corrupted Data Detected:")
    print(f"    - Dummy/Bot Customer Accounts: {len(dummy_ids)}")
    print(f"    - Negative or Zero Purchase Amounts: {len(corrupted_amounts)}")
    print(f"    - Out-of-bounds Sentinel Ages: {len(invalid_ages)}")
    print(f"    - Duplicate Transaction IDs: {duplicate_txns}")
    
    return df

def clean_and_preprocess(df):
    print("\n" + "=" * 60)
    print("STEP 2: DATA CLEANING & PREPROCESSING PIPELINE")
    print("=" * 60)
    
    initial_rows = len(df)
    
    # 1. Drop duplicates
    df = df.drop_duplicates(subset=['Transaction_ID'], keep='first')
    
    # 2. Filter out explicit dummy/bot testing records
    dummy_mask = df['Customer_ID'].str.contains('DUMMY|DEBUG|GUEST_0000', case=False, na=False) | \
                 df['Item_Category'].isin(['TEST_SAMPLE', 'UNKNOWN_CAT'])
    df = df[~dummy_mask].copy()
    
    # 3. Numeric Conversions
    for num_col in ['Age', 'Purchase_Amount', 'Review_Rating', 'Days_Since_Last_Purchase', 'Frequency_Annual']:
        df[num_col] = pd.to_numeric(df[num_col], errors='coerce')
    
    # 4. Handle Age Nulls & Sentinels (Median Imputation)
    median_age = df.loc[(df['Age'] >= 18) & (df['Age'] <= 90), 'Age'].median()
    df.loc[(df['Age'].isna()) | (df['Age'] < 18) | (df['Age'] > 90), 'Age'] = median_age
    
    # 5. Handle Purchase Amount Nulls & Negative Values (Median Positive Imputation)
    median_amount = df.loc[df['Purchase_Amount'] > 0, 'Purchase_Amount'].median()
    df.loc[(df['Purchase_Amount'].isna()) | (df['Purchase_Amount'] <= 0), 'Purchase_Amount'] = median_amount
    
    # 6. Review Rating (Valid 1 to 5)
    median_rating = df.loc[(df['Review_Rating'] >= 1) & (df['Review_Rating'] <= 5), 'Review_Rating'].median()
    df.loc[(df['Review_Rating'].isna()) | (df['Review_Rating'] < 1) | (df['Review_Rating'] > 5), 'Review_Rating'] = median_rating
    
    # 7. Recency and Frequency
    median_recency = df.loc[df['Days_Since_Last_Purchase'] >= 0, 'Days_Since_Last_Purchase'].median()
    df.loc[(df['Days_Since_Last_Purchase'].isna()) | (df['Days_Since_Last_Purchase'] < 0), 'Days_Since_Last_Purchase'] = median_recency
    
    median_freq = df.loc[df['Frequency_Annual'] >= 1, 'Frequency_Annual'].median()
    df.loc[(df['Frequency_Annual'].isna()) | (df['Frequency_Annual'] <= 0), 'Frequency_Annual'] = median_freq

    # 8. Clean Categorical columns
    df['Gender'] = df['Gender'].fillna('Other')
    df['Item_Category'] = df['Item_Category'].fillna('General')
    df['Payment_Method'] = df['Payment_Method'].fillna('Credit Card')
    
    print(f"[*] Data Cleaning Complete: Kept {len(df)} / {initial_rows} rows.")
    print(f"[*] Verified Zero Nulls Remaining: {df.isnull().sum().sum()} nulls.")
    return df

def perform_rfm_segmentation(df):
    print("\n" + "=" * 60)
    print("STEP 3: RFM CUSTOMER SEGMENTATION")
    print("=" * 60)
    
    # Recency (lower days = higher score 5)
    df['R_Score'] = pd.qcut(df['Days_Since_Last_Purchase'], q=5, labels=[5, 4, 3, 2, 1])
    # Frequency (higher frequency = higher score 5)
    df['F_Score'] = pd.qcut(df['Frequency_Annual'].rank(method='first'), q=5, labels=[1, 2, 3, 4, 5])
    # Monetary (higher amount = higher score 5)
    df['M_Score'] = pd.qcut(df['Purchase_Amount'].rank(method='first'), q=5, labels=[1, 2, 3, 4, 5])
    
    df['RFM_Score'] = df['R_Score'].astype(str) + df['F_Score'].astype(str) + df['M_Score'].astype(str)
    
    # Segment Assignment logic
    def assign_segment(row):
        r, f, m = int(row['R_Score']), int(row['F_Score']), int(row['M_Score'])
        if r >= 4 and f >= 4 and m >= 4:
            return 'Champions'
        elif r >= 3 and f >= 3 and m >= 3:
            return 'Loyal Customers'
        elif r >= 4 and f <= 3:
            return 'Potential Loyalists'
        elif r <= 2 and f >= 3 and m >= 3:
            return 'At Risk'
        elif r <= 2 and f <= 2:
            return 'Hibernating'
        elif r == 3 and f <= 3:
            return 'Need Attention'
        else:
            return 'Promising / New'
            
    df['RFM_Segment'] = df.apply(assign_segment, axis=1)
    
    print("\n[*] RFM Segment Distribution:")
    summary = df.groupby('RFM_Segment').agg(
        Count=('Customer_ID', 'count'),
        Avg_Recency_Days=('Days_Since_Last_Purchase', 'mean'),
        Avg_Annual_Orders=('Frequency_Annual', 'mean'),
        Avg_Spend=('Purchase_Amount', 'mean')
    ).round(2)
    print(summary)
    return df

def perform_kmeans_clustering(df, k=4):
    print("\n" + "=" * 60)
    print(f"STEP 4: K-MEANS CLUSTERING (k={k})")
    print("=" * 60)
    
    features = ['Age', 'Purchase_Amount', 'Frequency_Annual', 'Days_Since_Last_Purchase', 'Review_Rating']
    scaler = StandardScaler()
    scaled_data = scaler.fit_transform(df[features])
    
    kmeans = KMeans(n_clusters=k, random_state=42, n_init=10)
    df['Cluster'] = kmeans.fit_predict(scaled_data)
    
    cluster_labels = {
        0: 'VIP Power Buyers (High Spend & Frequency)',
        1: 'Deal Hunters & Tech Explorers',
        2: 'Steady Regular Family Shoppers',
        3: 'Disengaged / Churn Risk'
    }
    df['Cluster_Persona'] = df['Cluster'].map(cluster_labels)
    
    print("\n[*] K-Means Cluster Centroids (Denormalized Means):")
    cluster_stats = df.groupby('Cluster')[features].mean().round(2)
    print(cluster_stats)
    return df

def perform_market_basket_analysis(df, min_support=0.04, min_confidence=0.25):
    print("\n" + "=" * 60)
    print("STEP 5: MARKET BASKET ANALYSIS (APRIORI ASSOCIATION RULES)")
    print("=" * 60)
    
    # Parse items purchased string into list of transactions
    baskets = []
    for item_str in df['Items_Purchased'].dropna():
        items = [i.strip() for i in str(item_str).replace('"', '').split(';') if i.strip()]
        if len(items) > 1:
            baskets.append(set(items))
            
    total_baskets = len(baskets)
    print(f"[*] Analyzed {total_baskets} multi-item checkout baskets.")
    
    item_counts = Counter()
    pair_counts = Counter()
    
    for b in baskets:
        for item in b:
            item_counts[item] += 1
        for pair in combinations(sorted(b), 2):
            pair_counts[pair] += 1
            
    rules = []
    for (item_a, item_b), count_ab in pair_counts.items():
        support_ab = count_ab / total_baskets
        if support_ab >= min_support:
            # Rule: A -> B
            conf_a_to_b = count_ab / item_counts[item_a]
            lift_a_to_b = conf_a_to_b / (item_counts[item_b] / total_baskets)
            if conf_a_to_b >= min_confidence:
                rules.append((item_a, item_b, support_ab, conf_a_to_b, lift_a_to_b))
                
            # Rule: B -> A
            conf_b_to_a = count_ab / item_counts[item_b]
            lift_b_to_a = conf_b_to_a / (item_counts[item_a] / total_baskets)
            if conf_b_to_a >= min_confidence:
                rules.append((item_b, item_a, support_ab, conf_b_to_a, lift_b_to_a))
                
    rules_df = pd.DataFrame(rules, columns=['Antecedent', 'Consequent', 'Support', 'Confidence', 'Lift'])
    rules_df = rules_df.sort_values(by='Lift', ascending=False).reset_index(drop=True)
    
    print(f"\n[*] Top Discovered Association Rules (Sorted by Lift):")
    print(rules_df.head(8).to_string(index=False))
    return rules_df

def generate_visualizations(df):
    print("\n" + "=" * 60)
    print("STEP 6: GENERATING DATA MINING CHARTS & OUTPUTS")
    print("=" * 60)
    
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    fig.suptitle('Online Shopping Customer Behavior Analysis - Data Mining Overview', fontsize=16, fontweight='bold')
    
    # Chart 1: RFM Segments
    seg_counts = df['RFM_Segment'].value_counts()
    axes[0, 0].barh(seg_counts.index, seg_counts.values, color='#3B82F6')
    axes[0, 0].set_title('Customer Distribution by RFM Segment')
    axes[0, 0].set_xlabel('Number of Customers')
    
    # Chart 2: K-Means Cluster Scatter (Annual Spend vs Recency)
    colors = ['#10B981', '#3B82F6', '#F59E0B', '#EF4444']
    for c in sorted(df['Cluster'].unique()):
        subset = df[df['Cluster'] == c]
        axes[0, 1].scatter(subset['Days_Since_Last_Purchase'], subset['Purchase_Amount'], 
                           label=f'Cluster {c}', alpha=0.7, color=colors[c % len(colors)], edgecolors='none', s=45)
    axes[0, 1].set_title('K-Means Clusters: Spend vs Recency')
    axes[0, 1].set_xlabel('Days Since Last Purchase')
    axes[0, 1].set_ylabel('Purchase Amount ($)')
    axes[0, 1].legend()
    
    # Chart 3: Category Revenue Breakdown
    cat_spend = df.groupby('Item_Category')['Purchase_Amount'].sum().sort_values(ascending=False)
    axes[1, 0].bar(cat_spend.index, cat_spend.values, color='#8B5CF6')
    axes[1, 0].set_title('Total Revenue by Product Category ($)')
    axes[1, 0].tick_params(axis='x', rotation=30)
    
    # Chart 4: Rating Distribution vs Churn
    sns.boxplot(x='Review_Rating', y='Purchase_Amount', hue='Is_Churned', data=df, ax=axes[1, 1], palette='Set2')
    axes[1, 1].set_title('Spend & Review Rating by Churn Status')
    
    plt.tight_layout()
    plt.savefig('customer_behavior_mining_charts.png', dpi=200)
    print("[*] Saved visualization figure to 'customer_behavior_mining_charts.png'")
    
    # Save cleaned result datasets
    df.to_csv('cleaned_customer_behavior.csv', index=False)
    print("[*] Exported cleaned data to 'cleaned_customer_behavior.csv'")
    print("\n[SUCCESS] Data Mining Pipeline Finished Successfully!")

def main():
    df = load_and_audit_data()
    df_clean = clean_and_preprocess(df)
    df_rfm = perform_rfm_segmentation(df_clean)
    df_clustered = perform_kmeans_clustering(df_rfm)
    rules_df = perform_market_basket_analysis(df_clustered)
    generate_visualizations(df_clustered)

if __name__ == '__main__':
    main()
