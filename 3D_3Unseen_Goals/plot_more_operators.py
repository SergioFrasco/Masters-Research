import json
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
import sys


def plot_results(results, output_dir):
    """Create bar chart visualization of results by category."""
    
    print(f"Generating plots from {len(results)} tasks...")
    
    # Filter out the "NOT (green AND box)" task
    results = {k: v for k, v in results.items() if k != "NOT (green AND box)"}
    print(f"After filtering: {len(results)} tasks")
    
    # Organize by category
    categories = ["AND", "OR", "NOT", "COMPLEX"]
    category_results = {cat: [] for cat in categories}
    category_names = {cat: [] for cat in categories}
    category_errors = {cat: [] for cat in categories}
    
    for task_name, data in results.items():
        cat = data['category']
        category_results[cat].append(data['success_rate'])
        category_names[cat].append(task_name)
        category_errors[cat].append(data.get('std_reward', 0))
    
    # Print summary
    print("\n" + "="*70)
    print("RESULTS SUMMARY")
    print("="*70)
    for category in categories:
        if category_results[category]:
            avg = np.mean(category_results[category])
            print(f"\n{category} Tasks (avg: {avg:.3f}):")
            for name, sr in zip(category_names[category], category_results[category]):
                print(f"  {name}: {sr:.3f}")
    print("="*70 + "\n")
    
    # Create figure with subplots
    fig, axes = plt.subplots(2, 2, figsize=(18, 12))
    fig.suptitle('SR Agent Performance on Logical Compositions', fontsize=18, fontweight='bold')
    
    colors = {
        "AND": "#2ecc71",      # Green
        "OR": "#3498db",       # Blue
        "NOT": "#e74c3c",      # Red
        "COMPLEX": "#9b59b6"   # Purple
    }
    
    for idx, (ax, category) in enumerate(zip(axes.flat, categories)):
        if not category_results[category]:
            ax.set_visible(False)
            continue
        
        success_rates = category_results[category]
        task_names = category_names[category]
        errors = category_errors[category]
        
        x_pos = np.arange(len(task_names))
        
        bars = ax.bar(x_pos, success_rates, yerr=errors, 
                     color=colors[category], alpha=0.7, capsize=5, 
                     edgecolor='black', linewidth=1.5)
        
        # Add value labels on bars
        for i, (bar, val) in enumerate(zip(bars, success_rates)):
            height = bar.get_height()
            ax.text(bar.get_x() + bar.get_width()/2., height + 0.02,
                   f'{val:.2f}',
                   ha='center', va='bottom', fontsize=13, fontweight='bold')
        
        ax.set_xlabel('Task', fontsize=12, fontweight='bold')
        ax.set_ylabel('Success Rate', fontsize=12, fontweight='bold')
        ax.set_title(f'{category} Tasks', fontsize=14, fontweight='bold')
        ax.set_xticks(x_pos)
        ax.set_xticklabels(task_names, rotation=45, ha='right', fontsize=13)
        ax.set_ylim([0, 1.0])
        ax.grid(axis='y', alpha=0.3, linestyle='--')
        ax.axhline(y=0.5, color='gray', linestyle='--', linewidth=1.5, alpha=0.7, label='50% baseline')
        ax.legend(loc='lower right')
        ax.tick_params(axis='y', labelsize=13)
    
    plt.tight_layout()
    
    # Save figure
    plot_file = output_dir / "logical_composition_performance.png"
    plt.savefig(plot_file, dpi=300, bbox_inches='tight')
    print(f"✓ Detailed plot saved to: {plot_file}")
    plt.close()
    
    # Create summary comparison plot
    fig, ax = plt.subplots(figsize=(12, 7))
    
    # Calculate average success rate per category
    avg_success_by_category = []
    std_by_category = []
    category_labels = []
    
    for cat in categories:
        if category_results[cat]:
            avg_success = np.mean(category_results[cat])
            std_success = np.std(category_results[cat])
            avg_success_by_category.append(avg_success)
            std_by_category.append(std_success)
            category_labels.append(cat)
    
    x_pos = np.arange(len(category_labels))
    bars = ax.bar(x_pos, avg_success_by_category, 
                  yerr=std_by_category,
                  color=[colors[cat] for cat in category_labels],
                  alpha=0.7, edgecolor='black', linewidth=2.5,
                  capsize=8)
    
    # Add value labels
    for bar, val, std in zip(bars, avg_success_by_category, std_by_category):
        height = bar.get_height()
        ax.text(bar.get_x() + bar.get_width()/2., height + 0.02,
               f'{val:.3f}\n±{std:.3f}',
               ha='center', va='bottom', fontsize=13, fontweight='bold')
    
    ax.set_xlabel('Logic Type', fontsize=14, fontweight='bold')
    ax.set_ylabel('Average Success Rate', fontsize=14, fontweight='bold')
    ax.set_title('SR Agent Performance by Logic Type\n(Zero-Shot Compositional Generalization)', 
                 fontsize=16, fontweight='bold')
    ax.set_xticks(x_pos)
    ax.set_xticklabels(category_labels, fontsize=13, fontweight='bold')
    ax.set_ylim([0, 1.0])
    ax.grid(axis='y', alpha=0.3, linestyle='--')
    ax.axhline(y=0.5, color='gray', linestyle='--', linewidth=2, alpha=0.7, label='50% baseline')
    ax.legend(loc='lower right', fontsize=11)
    
    plt.tight_layout()
    
    summary_file = output_dir / "category_summary.png"
    plt.savefig(summary_file, dpi=300, bbox_inches='tight')
    print(f"✓ Summary plot saved to: {summary_file}")
    plt.close()
    
    # Create a single comprehensive plot
    fig, ax = plt.subplots(figsize=(16, 8))
    
    # Combine all tasks in order: AND, OR, NOT, COMPLEX
    all_task_names = []
    all_success_rates = []
    all_colors = []
    all_errors = []
    
    for cat in categories:
        if category_results[cat]:
            all_task_names.extend(category_names[cat])
            all_success_rates.extend(category_results[cat])
            all_colors.extend([colors[cat]] * len(category_results[cat]))
            all_errors.extend(category_errors[cat])
    
    x_pos = np.arange(len(all_task_names))
    bars = ax.bar(x_pos, all_success_rates, 
                  yerr=all_errors,
                  color=all_colors, alpha=0.7, 
                  edgecolor='black', linewidth=1.2,
                  capsize=4)
    
    # Add value labels on bars
    for bar, val in zip(bars, all_success_rates):
        height = bar.get_height()
        ax.text(bar.get_x() + bar.get_width()/2., height + 0.01,
               f'{val:.2f}',
               ha='center', va='bottom', fontsize=12, fontweight='bold',
               rotation=0)
    
    ax.set_xlabel('Task', fontsize=13, fontweight='bold')
    ax.set_ylabel('Success Rate', fontsize=13, fontweight='bold')
    ax.set_title('All Logical Composition Tasks - SR Agent Performance', 
                 fontsize=15, fontweight='bold')
    ax.set_xticks(x_pos)
    ax.set_xticklabels(all_task_names, rotation=60, ha='right', fontsize=12)
    ax.set_ylim([0, 1.0])
    ax.grid(axis='y', alpha=0.3, linestyle='--')
    ax.axhline(y=0.5, color='gray', linestyle='--', linewidth=1.5, alpha=0.7)
    ax.tick_params(axis='y', labelsize=12)
    
    # Add category separators
    cumulative = 0
    for cat in categories:
        if category_results[cat]:
            count = len(category_results[cat])
            if cumulative > 0:
                ax.axvline(x=cumulative - 0.5, color='black', linestyle='-', 
                          linewidth=2, alpha=0.5)
            cumulative += count
    
    # Create legend
    from matplotlib.patches import Patch
    legend_elements = [Patch(facecolor=colors[cat], edgecolor='black', 
                            label=cat, alpha=0.7) for cat in categories if category_results[cat]]
    ax.legend(handles=legend_elements, loc='lower right', fontsize=11)
    
    plt.tight_layout()
    
    comprehensive_file = output_dir / "all_tasks_comprehensive.png"
    plt.savefig(comprehensive_file, dpi=300, bbox_inches='tight')
    print(f"✓ Comprehensive plot saved to: {comprehensive_file}")
    plt.close()
    
    print(f"\n{'='*70}")
    print("All plots generated successfully!")
    print(f"{'='*70}\n")


def main():
    """Main execution function."""
    
    # Determine results file path
    if len(sys.argv) > 1:
        results_path = Path(sys.argv[1])
    else:
        results_path = Path("logical_composition_results/logical_composition_results.json")
    
    if not results_path.exists():
        print(f"Error: Results file not found at {results_path}")
        print("\nUsage:")
        print("  python plot_logical_results.py [path_to_results.json]")
        print("\nOr place results.json at:")
        print("  logical_composition_results/logical_composition_results.json")
        sys.exit(1)
    
    # Load results
    print(f"Loading results from: {results_path}")
    with open(results_path, 'r') as f:
        results = json.load(f)
    
    print(f"Found {len(results)} tasks")
    
    # Determine output directory (same as results file)
    output_dir = results_path.parent
    
    # Generate plots
    plot_results(results, output_dir)
    
    print("\n✓ Done!")


if __name__ == "__main__":
    main()