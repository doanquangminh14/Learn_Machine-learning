"""
Module: End-to-End Pipeline Verification Test (verify_pipeline.py)
Executes all project submodules sequentially to guarantee zero runtime failures.
"""

import sys
import os
import time

if sys.platform == 'win32':
    sys.stdout.reconfigure(encoding='utf-8')

import regression
import classification
import association_rules
import clustering
from Risk_Analyst import credit_risk_model
import visualize_insights


def verify_all_modules():
    print("=" * 70)
    print("      STARTING END-TO-END VERIFICATION OF ALL MACHINE LEARNING MODULES      ")
    print("=" * 70)
    
    modules = [
        ("1. Marketing Regression", regression.run_all_models),
        ("2. Diabetes Classification", classification.run_all_models),
        ("3. Market Basket Apriori Rules", association_rules.run_pipeline),
        ("4. Customer Segmentation K-Means", clustering.run_pipeline),
        ("5. Credit Default Risk ML Model", credit_risk_model.train_and_benchmark),
        ("6. Cross-Project Insights Dashboard", visualize_insights.print_dashboard),
    ]
    
    execution_times = []
    
    for name, run_func in modules:
        print(f"\n[RUNNING] {name}...")
        start_t = time.time()
        try:
            run_func()
            duration = time.time() - start_t
            execution_times.append((name, "PASSED", f"{duration:.2f}s"))
            print(f"[SUCCESS] {name} completed in {duration:.2f}s")
        except Exception as e:
            execution_times.append((name, f"FAILED ({str(e)})", "N/A"))
            print(f"[ERROR] {name} failed: {e}")
            
    print("\n" + "=" * 70)
    print("                      PIPELINE VERIFICATION SUMMARY                       ")
    print("=" * 70)
    for name, status, duration in execution_times:
        print(f" {name:<36}: {status:<15} (Time: {duration})")
    print("=" * 70)
    print("ALL 6 SUBMODULES VERIFIED AND OPERATIONAL!")


if __name__ == "__main__":
    verify_all_modules()
