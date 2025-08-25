#!/usr/bin/env python3
"""
Individual Model Runner
======================

This script allows you to run any of the advanced embedding models individually.
Use this when you want to test a specific model or run experiments separately.

Usage:
    python run_individual_model.py --model 1  # Run MPNet Base v2
    python run_individual_model.py --model 2  # Run DistilRoBERTa v1  
    python run_individual_model.py --model 3  # Run Multi-QA MPNet

Author: Research Team
Date: August 24, 2025
"""

import argparse
import sys
import subprocess
from pathlib import Path

# Available models
MODELS = {
    0: {
        'name': 'MiniLM L6 v2',
        'script': 'model_0_minilm_l6_v2.py',
        'description': 'all-MiniLM-L6-v2 - Lightweight and efficient baseline model'
    },
    1: {
        'name': 'MPNet Base v2',
        'script': 'model_1_mpnet_base_v2.py',
        'description': 'all-mpnet-base-v2 - High-performance sentence transformer'
    },
    2: {
        'name': 'DistilRoBERTa v1', 
        'script': 'model_2_distilroberta_v1.py',
        'description': 'all-distilroberta-v1 - Fast and efficient transformer'
    },
    3: {
        'name': 'Multi-QA MPNet',
        'script': 'model_3_multi_qa_mpnet.py', 
        'description': 'multi-qa-mpnet-base-dot-v1 - Optimized for QA and similarity'
    }
}

def list_models():
    """Display available models."""
    print("Available Advanced Embedding Models:")
    print("=" * 60)
    for num, model in MODELS.items():
        print(f"{num}. {model['name']}")
        print(f"   {model['description']}")
        print(f"   Script: {model['script']}")
        print()

def run_model(model_num):
    """Run the specified model."""
    if model_num not in MODELS:
        print(f"❌ Invalid model number: {model_num}")
        print("Available models: 1, 2, 3")
        return False
    
    model_info = MODELS[model_num]
    script_path = Path(model_info['script'])
    
    if not script_path.exists():
        print(f"❌ Script not found: {script_path}")
        return False
    
    print(f"🚀 Starting {model_info['name']}...")
    print(f"📄 Script: {model_info['script']}")
    print(f"📝 Description: {model_info['description']}")
    print("-" * 60)
    
    try:
        # Run the model script
        result = subprocess.run([sys.executable, str(script_path)], check=True)
        print(f"\n✅ {model_info['name']} completed successfully!")
        return True
    except subprocess.CalledProcessError as e:
        print(f"\n❌ {model_info['name']} failed with error code: {e.returncode}")
        return False
    except KeyboardInterrupt:
        print(f"\n⚠️ {model_info['name']} interrupted by user")
        return False
    except Exception as e:
        print(f"\n❌ Unexpected error running {model_info['name']}: {str(e)}")
        return False

def main():
    """Main execution function."""
    parser = argparse.ArgumentParser(
        description="Run individual advanced embedding models",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
    python run_individual_model.py --model 0    # Run MiniLM L6 v2 (baseline)
    python run_individual_model.py --model 1    # Run MPNet Base v2
    python run_individual_model.py --model 2    # Run DistilRoBERTa v1
    python run_individual_model.py --model 3    # Run Multi-QA MPNet
    python run_individual_model.py --list       # List all available models
        """
    )
    
    parser.add_argument(
        '--model', '-m',
        type=int,
        choices=[0, 1, 2, 3],
        help='Model number to run (0=MiniLM, 1=MPNet, 2=DistilRoBERTa, 3=Multi-QA MPNet)'
    )
    
    parser.add_argument(
        '--list', '-l',
        action='store_true',
        help='List all available models'
    )
    
    args = parser.parse_args()
    
    if args.list:
        list_models()
        return
    
    if args.model is None:
        print("❌ Please specify a model to run with --model or use --list to see available models")
        parser.print_help()
        return
    
    success = run_model(args.model)
    sys.exit(0 if success else 1)

if __name__ == "__main__":
    main()
