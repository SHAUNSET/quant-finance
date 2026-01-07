"""
MASTER SCRIPT: Run all portfolio theory examples
"""

import subprocess
import time

def run_script(script_name, description):
    print(f"\n{'='*60}")
    print(f"RUNNING: {description}")
    print(f"{'='*60}")
    
    try:
        result = subprocess.run(['python', script_name], 
                              capture_output=True, text=True)
        print(result.stdout)
        if result.stderr:
            print(f"Warnings: {result.stderr}")
    except Exception as e:
        print(f"Error running {script_name}: {e}")
    
    time.sleep(2)  # Pause between scripts

def main():
    print("="*60)
    print("QUANT FINANCE PORTFOLIO THEORY DEMONSTRATION")
    print("="*60)
    
    # Run in logical order
    scripts = [
        ("efficient_frontier.py", "1. Basic 2-Asset Efficient Frontier"),
        ("utility_theory.py", "2. Basic Utility Theory (2 Assets)"),
        ("3_asset_efficient_frontier.py", "3. Interactive 3-Asset Frontier"),
        ("efficient_frontier_enhanced.py", "4. Enhanced 5-Asset Frontier"),
        ("utility_theory_enhanced.py", "5. Enhanced Utility Theory"),
        ("basic_riskfreeasset.py", "6. Risk-Free Asset Basics"),
        ("risk_free_story_friends.py", "7. Complete Risk-Free Story")
    ]
    
    for script, description in scripts:
        run_script(script, description)
    
    print("\n" + "="*60)
    print("ALL SCRIPTS COMPLETED!")
    print("="*60)

if __name__ == "__main__":
    main()