#!/usr/bin/env python3
"""
Visualization script for Baseline 1 results (Few-Shot Linear Probe)
Generates publication-quality figures for CS229 milestone report
"""

import json
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
import argparse

# Set publication-quality style
plt.style.use('seaborn-v0_8-paper')
sns.set_palette("husl")
plt.rcParams['figure.dpi'] = 300
plt.rcParams['savefig.dpi'] = 300
plt.rcParams['font.size'] = 10
plt.rcParams['axes.labelsize'] = 11
plt.rcParams['axes.titlesize'] = 12
plt.rcParams['xtick.labelsize'] = 9
plt.rcParams['ytick.labelsize'] = 9
plt.rcParams['legend.fontsize'] = 9
plt.rcParams['figure.titlesize'] = 13


def load_results(results_dir):
    """Load results from JSON file."""
    results_file = Path(results_dir) / "baseline1_results.json"
    if not results_file.exists():
        raise FileNotFoundError(f"Results file not found: {results_file}")
    
    with open(results_file, 'r') as f:
        results = json.load(f)
    
    return results


def plot_data_efficiency_curve(results, output_dir):
    """
    Plot 1: Data Efficiency Curve
    Shows test accuracy vs K-shot (number of labeled examples per class)
    """
    fig, ax = plt.subplots(figsize=(8, 5))
    
    k_values = []
    val_means = []
    val_stds = []
    test_means = []
    test_stds = []
    
    for exp in results['experiments']:
        k = exp['k']
        k_values.append(k)
        
        # Extract accuracies across seeds
        val_accs = [s['val_accuracy'] for s in exp['seeds']]
        test_accs = [s['test_accuracy'] for s in exp['seeds']]
        
        val_means.append(np.mean(val_accs))
        val_stds.append(np.std(val_accs))
        test_means.append(np.mean(test_accs))
        test_stds.append(np.std(test_accs))
    
    # Sort by K value
    sort_idx = np.argsort(k_values)
    k_values = np.array(k_values)[sort_idx]
    val_means = np.array(val_means)[sort_idx]
    val_stds = np.array(val_stds)[sort_idx]
    test_means = np.array(test_means)[sort_idx]
    test_stds = np.array(test_stds)[sort_idx]
    
    # Plot with error bars
    ax.errorbar(k_values, test_means * 100, yerr=test_stds * 100, 
                marker='o', linewidth=2, capsize=5, capthick=2,
                label='Test Accuracy', color='#2E86AB', markersize=8)
    ax.errorbar(k_values, val_means * 100, yerr=val_stds * 100, 
                marker='s', linewidth=2, capsize=5, capthick=2, alpha=0.7,
                label='Val Accuracy', color='#A23B72', markersize=7)
    
    ax.set_xlabel('K-shot (labeled examples per class)', fontweight='bold')
    ax.set_ylabel('Accuracy (%)', fontweight='bold')
    ax.set_title('Few-Shot Learning: Data Efficiency on UCF-101', fontweight='bold', pad=15)
    ax.grid(True, alpha=0.3, linestyle='--')
    ax.legend(frameon=True, fancybox=True, shadow=True)
    
    # Add data labels
    for k, test_mean, val_mean in zip(k_values, test_means, val_means):
        ax.annotate(f'{test_mean*100:.1f}%', 
                   xy=(k, test_mean*100), 
                   xytext=(0, 10), 
                   textcoords='offset points',
                   ha='center', 
                   fontsize=8,
                   bbox=dict(boxstyle='round,pad=0.3', facecolor='white', alpha=0.7))
    
    # Add total labeled videos on secondary x-axis
    ax2 = ax.twiny()
    ax2.set_xlim(ax.get_xlim())
    ax2.set_xticks(k_values)
    ax2.set_xticklabels([f'{k*101}' for k in k_values])
    ax2.set_xlabel('Total labeled videos', fontweight='bold', color='gray')
    ax2.tick_params(axis='x', colors='gray')
    
    plt.tight_layout()
    output_path = Path(output_dir) / 'data_efficiency_curve.png'
    plt.savefig(output_path, bbox_inches='tight')
    plt.savefig(output_path.with_suffix('.pdf'), bbox_inches='tight')
    print(f"✓ Saved: {output_path}")
    plt.close()


def plot_performance_comparison(results, output_dir):
    """
    Plot 2: Performance Comparison Bar Chart
    Compares Val vs Test accuracy for each K-shot
    """
    fig, ax = plt.subplots(figsize=(10, 5))
    
    k_values = []
    val_means = []
    test_means = []
    
    for exp in results['experiments']:
        k = exp['k']
        k_values.append(k)
        val_accs = [s['val_accuracy'] for s in exp['seeds']]
        test_accs = [s['test_accuracy'] for s in exp['seeds']]
        val_means.append(np.mean(val_accs))
        test_means.append(np.mean(test_accs))
    
    # Sort by K
    sort_idx = np.argsort(k_values)
    k_values = np.array(k_values)[sort_idx]
    val_means = np.array(val_means)[sort_idx] * 100
    test_means = np.array(test_means)[sort_idx] * 100
    
    x = np.arange(len(k_values))
    width = 0.35
    
    bars1 = ax.bar(x - width/2, val_means, width, label='Validation', 
                   color='#A23B72', alpha=0.8, edgecolor='black', linewidth=0.5)
    bars2 = ax.bar(x + width/2, test_means, width, label='Test', 
                   color='#2E86AB', alpha=0.8, edgecolor='black', linewidth=0.5)
    
    # Add value labels on bars
    for bars in [bars1, bars2]:
        for bar in bars:
            height = bar.get_height()
            ax.text(bar.get_x() + bar.get_width()/2., height,
                   f'{height:.1f}%',
                   ha='center', va='bottom', fontsize=8)
    
    ax.set_xlabel('K-shot (labeled per class)', fontweight='bold')
    ax.set_ylabel('Accuracy (%)', fontweight='bold')
    ax.set_title('Validation vs Test Accuracy Across K-shot Settings', fontweight='bold', pad=15)
    ax.set_xticks(x)
    ax.set_xticklabels([f'K={k}' for k in k_values])
    ax.legend(frameon=True, fancybox=True, shadow=True)
    ax.grid(True, alpha=0.3, axis='y', linestyle='--')
    
    plt.tight_layout()
    output_path = Path(output_dir) / 'performance_comparison.png'
    plt.savefig(output_path, bbox_inches='tight')
    plt.savefig(output_path.with_suffix('.pdf'), bbox_inches='tight')
    print(f"✓ Saved: {output_path}")
    plt.close()


def plot_learning_curves(results, output_dir):
    """
    Plot 3: Learning Curves (if iteration data available)
    Shows how validation accuracy improves during training
    """
    # Note: This requires storing iteration-wise validation accuracy
    # For now, create a placeholder showing convergence behavior
    fig, ax = plt.subplots(figsize=(8, 5))
    
    # Extract stopped iterations for each K-shot
    k_values = []
    stopped_iters = []
    
    for exp in results['experiments']:
        k = exp['k']
        for seed_result in exp['seeds']:
            if 'stopped_at_iter' in seed_result:
                k_values.append(k)
                stopped_iters.append(seed_result['stopped_at_iter'])
    
    if k_values:
        # Group by K
        k_unique = sorted(set(k_values))
        avg_iters = []
        std_iters = []
        
        for k in k_unique:
            k_iters = [stopped_iters[i] for i, kv in enumerate(k_values) if kv == k]
            avg_iters.append(np.mean(k_iters))
            std_iters.append(np.std(k_iters))
        
        ax.bar(range(len(k_unique)), avg_iters, yerr=std_iters, 
               capsize=5, color='#F18F01', alpha=0.8, edgecolor='black', linewidth=0.5)
        ax.set_xlabel('K-shot', fontweight='bold')
        ax.set_ylabel('Iterations Until Convergence', fontweight='bold')
        ax.set_title('Training Convergence Speed vs K-shot', fontweight='bold', pad=15)
        ax.set_xticks(range(len(k_unique)))
        ax.set_xticklabels([f'K={k}' for k in k_unique])
        ax.grid(True, alpha=0.3, axis='y', linestyle='--')
        
        # Add value labels
        for i, (avg, std) in enumerate(zip(avg_iters, std_iters)):
            ax.text(i, avg + std + 1, f'{int(avg)}', 
                   ha='center', va='bottom', fontsize=9)
    else:
        ax.text(0.5, 0.5, 'Iteration data not available', 
               ha='center', va='center', transform=ax.transAxes, fontsize=12)
    
    plt.tight_layout()
    output_path = Path(output_dir) / 'convergence_speed.png'
    plt.savefig(output_path, bbox_inches='tight')
    plt.savefig(output_path.with_suffix('.pdf'), bbox_inches='tight')
    print(f"✓ Saved: {output_path}")
    plt.close()


def plot_variance_analysis(results, output_dir):
    """
    Plot 4: Variance Analysis
    Shows stability across different random seeds
    """
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))
    
    k_values = []
    test_stds = []
    val_stds = []
    test_means = []
    
    for exp in results['experiments']:
        k = exp['k']
        k_values.append(k)
        
        val_accs = [s['val_accuracy'] for s in exp['seeds']]
        test_accs = [s['test_accuracy'] for s in exp['seeds']]
        
        val_stds.append(np.std(val_accs) * 100)
        test_stds.append(np.std(test_accs) * 100)
        test_means.append(np.mean(test_accs) * 100)
    
    # Sort
    sort_idx = np.argsort(k_values)
    k_values = np.array(k_values)[sort_idx]
    val_stds = np.array(val_stds)[sort_idx]
    test_stds = np.array(test_stds)[sort_idx]
    test_means = np.array(test_means)[sort_idx]
    
    # Plot 1: Standard Deviation
    x = np.arange(len(k_values))
    width = 0.35
    
    ax1.bar(x - width/2, val_stds, width, label='Val Std', 
           color='#A23B72', alpha=0.8, edgecolor='black', linewidth=0.5)
    ax1.bar(x + width/2, test_stds, width, label='Test Std', 
           color='#2E86AB', alpha=0.8, edgecolor='black', linewidth=0.5)
    
    ax1.set_xlabel('K-shot', fontweight='bold')
    ax1.set_ylabel('Standard Deviation (%)', fontweight='bold')
    ax1.set_title('Performance Variance Across Seeds', fontweight='bold')
    ax1.set_xticks(x)
    ax1.set_xticklabels([f'K={k}' for k in k_values])
    ax1.legend(frameon=True)
    ax1.grid(True, alpha=0.3, axis='y', linestyle='--')
    
    # Plot 2: Coefficient of Variation (relative stability)
    cv = (test_stds / test_means) * 100
    
    ax2.plot(k_values, cv, marker='o', linewidth=2, markersize=8, 
            color='#C73E1D')
    ax2.fill_between(k_values, 0, cv, alpha=0.3, color='#C73E1D')
    
    ax2.set_xlabel('K-shot', fontweight='bold')
    ax2.set_ylabel('Coefficient of Variation (%)', fontweight='bold')
    ax2.set_title('Relative Stability (CV = Std/Mean)', fontweight='bold')
    ax2.grid(True, alpha=0.3, linestyle='--')
    
    # Add value labels
    for k, cv_val in zip(k_values, cv):
        ax2.annotate(f'{cv_val:.2f}%', 
                    xy=(k, cv_val), 
                    xytext=(0, 8), 
                    textcoords='offset points',
                    ha='center', fontsize=8)
    
    plt.tight_layout()
    output_path = Path(output_dir) / 'variance_analysis.png'
    plt.savefig(output_path, bbox_inches='tight')
    plt.savefig(output_path.with_suffix('.pdf'), bbox_inches='tight')
    print(f"✓ Saved: {output_path}")
    plt.close()


def generate_results_table(results, output_dir):
    """
    Generate LaTeX table for milestone report
    """
    table = []
    table.append("\\begin{table}[h]")
    table.append("\\centering")
    table.append("\\caption{Few-Shot Linear Probe Results on UCF-101}")
    table.append("\\label{tab:fewshot_results}")
    table.append("\\begin{tabular}{c|cc|cc|c}")
    table.append("\\hline")
    table.append("K-shot & Val Acc (\\%) & Val F1 & Test Acc (\\%) & Test F1 & Labeled Videos \\\\")
    table.append("\\hline")
    
    for exp in sorted(results['experiments'], key=lambda x: x['k']):
        k = exp['k']
        
        val_accs = [s['val_accuracy'] for s in exp['seeds']]
        val_f1s = [s['val_macro_f1'] for s in exp['seeds']]
        test_accs = [s['test_accuracy'] for s in exp['seeds']]
        test_f1s = [s['test_macro_f1'] for s in exp['seeds']]
        
        val_acc_mean = np.mean(val_accs) * 100
        val_acc_std = np.std(val_accs) * 100
        val_f1_mean = np.mean(val_f1s)
        val_f1_std = np.std(val_f1s)
        test_acc_mean = np.mean(test_accs) * 100
        test_acc_std = np.std(test_accs) * 100
        test_f1_mean = np.mean(test_f1s)
        test_f1_std = np.std(test_f1s)
        
        n_labeled = k * 101
        
        table.append(f"{k} & "
                    f"{val_acc_mean:.2f}$\\pm${val_acc_std:.2f} & "
                    f"{val_f1_mean:.3f}$\\pm${val_f1_std:.3f} & "
                    f"{test_acc_mean:.2f}$\\pm${test_acc_std:.2f} & "
                    f"{test_f1_mean:.3f}$\\pm${test_f1_std:.3f} & "
                    f"{n_labeled} \\\\")
    
    table.append("\\hline")
    table.append("\\end{tabular}")
    table.append("\\end{table}")
    
    # Save to file
    output_path = Path(output_dir) / 'results_table.tex'
    with open(output_path, 'w') as f:
        f.write('\n'.join(table))
    
    print(f"✓ Saved LaTeX table: {output_path}")
    
    # Also print markdown version
    print("\n" + "="*80)
    print("MARKDOWN TABLE (for milestone):")
    print("="*80)
    print("\n| K-shot | Val Acc (%) | Test Acc (%) | Test F1 | Labeled Videos |")
    print("|--------|-------------|--------------|---------|----------------|")
    
    for exp in sorted(results['experiments'], key=lambda x: x['k']):
        k = exp['k']
        val_accs = [s['val_accuracy'] for s in exp['seeds']]
        test_accs = [s['test_accuracy'] for s in exp['seeds']]
        test_f1s = [s['test_macro_f1'] for s in exp['seeds']]
        
        val_acc_mean = np.mean(val_accs) * 100
        val_acc_std = np.std(val_accs) * 100
        test_acc_mean = np.mean(test_accs) * 100
        test_acc_std = np.std(test_accs) * 100
        test_f1_mean = np.mean(test_f1s)
        test_f1_std = np.std(test_f1s)
        n_labeled = k * 101
        
        print(f"| K={k} | {val_acc_mean:.2f}±{val_acc_std:.2f} | "
              f"{test_acc_mean:.2f}±{test_acc_std:.2f} | "
              f"{test_f1_mean:.3f}±{test_f1_std:.3f} | {n_labeled} |")
    
    print("\n" + "="*80 + "\n")


def generate_summary_stats(results, output_dir):
    """Generate summary statistics text file"""
    output_path = Path(output_dir) / 'summary_stats.txt'
    
    with open(output_path, 'w') as f:
        f.write("="*80 + "\n")
        f.write("BASELINE 1: FEW-SHOT LINEAR PROBE - SUMMARY STATISTICS\n")
        f.write("="*80 + "\n\n")
        
        f.write(f"Model: X-CLIP base (patch16)\n")
        f.write(f"Dataset: UCF-101 (101 classes)\n")
        f.write(f"Feature dim: 512\n")
        f.write(f"K-shot values tested: {results['k_shots']}\n")
        f.write(f"Random seeds: {results['seeds']}\n\n")
        
        f.write("RESULTS BY K-SHOT:\n")
        f.write("-"*80 + "\n\n")
        
        for exp in sorted(results['experiments'], key=lambda x: x['k']):
            k = exp['k']
            f.write(f"K = {k} ({k*101} labeled videos, {k*101/9324*100:.2f}% of training data)\n")
            f.write("-"*40 + "\n")
            
            val_accs = np.array([s['val_accuracy'] for s in exp['seeds']]) * 100
            test_accs = np.array([s['test_accuracy'] for s in exp['seeds']]) * 100
            test_f1s = np.array([s['test_macro_f1'] for s in exp['seeds']])
            
            f.write(f"  Val Accuracy:  {val_accs.mean():.2f}% ± {val_accs.std():.2f}%\n")
            f.write(f"  Test Accuracy: {test_accs.mean():.2f}% ± {test_accs.std():.2f}%\n")
            f.write(f"  Test Macro-F1: {test_f1s.mean():.3f} ± {test_f1s.std():.3f}\n")
            f.write(f"  Seeds: {[s['seed'] for s in exp['seeds']]}\n")
            f.write(f"  Individual runs: {test_accs.tolist()}\n\n")
        
        # Best performing K
        best_k = max(results['experiments'], 
                    key=lambda x: np.mean([s['test_accuracy'] for s in x['seeds']]))
        best_test_acc = np.mean([s['test_accuracy'] for s in best_k['seeds']]) * 100
        
        f.write("="*80 + "\n")
        f.write(f"BEST PERFORMANCE: K={best_k['k']} achieves {best_test_acc:.2f}% test accuracy\n")
        f.write("="*80 + "\n")
    
    print(f"✓ Saved summary: {output_path}")


def main():
    parser = argparse.ArgumentParser(description="Visualize Baseline 1 Results")
    parser.add_argument(
        "--results_dir",
        type=str,
        default="/mnt/amlfs-03/shared/ayaanm/yanav/results/baseline1",
        help="Directory containing baseline1_results.json"
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        default="/mnt/amlfs-03/shared/ayaanm/yanav/results/baseline1/figures",
        help="Output directory for figures"
    )
    
    args = parser.parse_args()
    
    # Create output directory
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    print("="*80)
    print("BASELINE 1 VISUALIZATION")
    print("="*80)
    print(f"Results directory: {args.results_dir}")
    print(f"Output directory: {args.output_dir}")
    print()
    
    # Load results
    print("Loading results...")
    results = load_results(args.results_dir)
    print(f"✓ Loaded results with {len(results['experiments'])} K-shot configurations\n")
    
    # Generate all visualizations
    print("Generating visualizations...")
    plot_data_efficiency_curve(results, output_dir)
    plot_performance_comparison(results, output_dir)
    plot_learning_curves(results, output_dir)
    plot_variance_analysis(results, output_dir)
    
    print("\nGenerating tables and summaries...")
    generate_results_table(results, output_dir)
    generate_summary_stats(results, output_dir)
    
    print("\n" + "="*80)
    print("✓ ALL VISUALIZATIONS COMPLETE!")
    print("="*80)
    print(f"\nOutputs saved to: {output_dir}")
    print("\nGenerated files:")
    print("  - data_efficiency_curve.png/pdf")
    print("  - performance_comparison.png/pdf")
    print("  - convergence_speed.png/pdf")
    print("  - variance_analysis.png/pdf")
    print("  - results_table.tex (LaTeX)")
    print("  - summary_stats.txt")
    print("\n" + "="*80)


if __name__ == "__main__":
    main()

