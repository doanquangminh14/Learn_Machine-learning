"""
Module: Market Basket Analysis with Apriori Association Rules (assio.csv)
Mining frequent itemsets and association rules to identify purchasing patterns.
"""

import sys
import os
import pandas as pd
from collections import Counter
from apyori import apriori

# Fix terminal encoding issues on Windows
if sys.platform == 'win32':
    sys.stdout.reconfigure(encoding='utf-8')


def load_and_preprocess_transactions(file_path: str = "dataset/assio.csv"):
    """
    Load transaction dataset and clean missing/NaN entries.
    Returns a list of transactions (each is a list of cleaned items).
    """
    df = pd.read_csv(file_path)
    
    print("=" * 65)
    print("MARKET BASKET TRANSACTION DATASET EXPLORATION:")
    print(f"Total Transactions: {len(df)}")
    
    transactions = []
    for _, row in df.iterrows():
        items = []
        for item in row:
            if pd.notna(item):
                item_str = str(item).strip()
                if item_str.lower() not in ['nan', '', 'null', 'none']:
                    items.append(item_str)
        if items:
            transactions.append(items)
            
    # Analyze item frequencies
    all_items = [item for txn in transactions for item in txn]
    item_counts = Counter(all_items)
    
    print("\nItem Frequencies across All Baskets:")
    for item, count in item_counts.most_common():
        support_pct = (count / len(transactions)) * 100
        print(f" - {item:<15}: {count:2d} transactions ({support_pct:.1f}%)")
        
    print("=" * 65)
    return transactions


def mine_association_rules(transactions, min_support: float = 0.25, min_confidence: float = 0.5, min_lift: float = 1.0):
    """
    Extract association rules using the Apriori algorithm.
    """
    print(f"\nMining rules with min_support={min_support}, min_confidence={min_confidence}, min_lift={min_lift}...")
    
    rules_generator = apriori(
        transactions,
        min_support=min_support,
        min_confidence=min_confidence,
        min_lift=min_lift
    )
    
    results = list(rules_generator)
    
    records = []
    for record in results:
        items = tuple(record.items)
        support = record.support
        
        for stat in record.ordered_statistics:
            items_base = tuple(stat.items_base)
            items_add = tuple(stat.items_add)
            
            # Filter out empty antecedents/consequents for clear directional rules
            if len(items_base) > 0 and len(items_add) > 0:
                records.append({
                    "Antecedent (If Buy)": ", ".join(items_base),
                    "Consequent (Then Buy)": ", ".join(items_add),
                    "Support": round(support, 4),
                    "Confidence": round(stat.confidence, 4),
                    "Lift": round(stat.lift, 4)
                })
                
    rules_df = pd.DataFrame(records)
    if not rules_df.empty:
        rules_df = rules_df.sort_values(by=["Lift", "Confidence"], ascending=[False, False]).reset_index(drop=True)
    return rules_df


def print_business_insights(rules_df: pd.DataFrame):
    """Print actionable cross-selling recommendations from mined rules."""
    print("\n" + "=" * 65)
    print("TOP EXTRACTED ASSOCIATION RULES (Sorted by Lift & Confidence):")
    if rules_df.empty:
        print("No association rules found meeting the minimum thresholds.")
    else:
        print(rules_df.to_string(index=True))
        
        print("\nACTIONABLE CROSS-SELLING RECOMMENDATIONS:")
        for idx, row in rules_df.head(5).iterrows():
            print(f" {idx + 1}. Customers who purchase [{row['Antecedent (If Buy)']}] are {row['Lift']}x more likely to also buy [{row['Consequent (Then Buy)']}] (Confidence: {row['Confidence'] * 100:.1f}%)")
    print("=" * 65)


def run_pipeline():
    """Main execution entry point."""
    transactions = load_and_preprocess_transactions()
    rules_df = mine_association_rules(transactions, min_support=0.3, min_confidence=0.5, min_lift=1.0)
    print_business_insights(rules_df)
    return rules_df


if __name__ == "__main__":
    run_pipeline()
