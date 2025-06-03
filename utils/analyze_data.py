import json
import os
import glob
from collections import Counter, defaultdict
import matplotlib.pyplot as plt

# --- Configuration ---
INPUT_DIR = "generated_data"  # Directory where JSON files are stored
OUTPUT_PLOTS_DIR = "vulnerability_plots" # Directory to save the generated plots

# Define a consistent order for severities for plotting
SEVERITY_ORDER = ["CRITICAL", "HIGH", "MEDIUM", "LOW"]
STATUS_ORDER = ["fixed", "affected", "under_investigation", "will_not_fix", "fix_deferred", "end_of_life", "UNKNOWN"]

LLM_NAME_DICT = {
    "gpt-4.1": "GPT-4.1",
    "deepseek-chat": "DeepSeek-V3",
    "grok-3": "Grok-3",
}


def plot_and_save_histogram(data_counter, title, xlabel, ylabel, filename_base, output_dir, category_order=None):
    """
    Generates a bar chart from a Counter object and saves it.
    """
    if not data_counter:
        print(f"  No data to plot for '{title}'. Skipping plot generation.")
        return

    if category_order:
        # Filter and order categories based on predefined order
        categories = [cat for cat in category_order]
        counts = [data_counter[cat] for cat in categories]
    else:
        # Sort alphabetically if no specific order
        sorted_items = sorted(data_counter.items())
        if not sorted_items: # handle case where data_counter had items but none matched category_order
            print(f"  No data to plot for '{title}' after filtering. Skipping plot generation.")
            return
        categories = [item[0] for item in sorted_items]
        counts = [item[1] for item in sorted_items]

    if not categories: # Final check if categories list is empty
        print(f"  No categories left to plot for '{title}'. Skipping plot generation.")
        return

    plt.figure(figsize=(12, 7))
    bars = plt.bar(categories, counts, color=['skyblue', 'lightcoral', 'lightgreen', 'gold', 'lightsalmon', 'plum'])

    # Add counts on top of bars
    for bar in bars:
        yval = bar.get_height()
        plt.text(bar.get_x() + bar.get_width()/2.0, yval + 0.05 * max(counts, default=1), # Adjust offset based on max count
                 int(yval) if yval.is_integer() else f'{yval:.2f}', # Format as int if whole number
                 ha='center', va='bottom')

    plt.title(title, fontsize=16)
    plt.xlabel(xlabel, fontsize=12)
    plt.ylabel(ylabel, fontsize=12)
    plt.xticks(rotation=45, ha="right")
    plt.yticks(range(0, max(counts, default=1) + 2, max(1, (max(counts, default=1) // 10) +1 ))) # Make y-axis ticks integers
    plt.grid(axis='y', linestyle='--')
    plt.tight_layout() # Adjust layout to prevent labels from being cut off

    plot_filename = os.path.join(output_dir, filename_base + ".png")
    plt.savefig(plot_filename)
    print(f"  Plot saved: {plot_filename}")
    plt.close() # Close the figure to free memory


def analyze_trivy_results_and_plot():
    """
    Analyzes Trivy scan results from multiple JSON files and generates plots.
    """
    if not os.path.exists(OUTPUT_PLOTS_DIR):
        os.makedirs(OUTPUT_PLOTS_DIR)
        print(f"Created output directory for plots: {OUTPUT_PLOTS_DIR}")

    aggregated_data = defaultdict(lambda: {
        "severities": Counter(),
        "statuses": Counter(),
        "total_vulnerabilities": 0,
    })
    overall_severities = Counter()
    overall_statuses = Counter()
    total_vulnerabilities_overall = 0

    json_files = glob.glob(os.path.join(INPUT_DIR, "result_prompt_*.json"))

    if not json_files:
        print(f"No JSON files found in directory: {INPUT_DIR}")
        return

    print(f"Found {len(json_files)} JSON files to process.")

    for file_path in json_files:
        print(f"\nProcessing file: {os.path.basename(file_path)}...")
        try:
            with open(file_path, 'r', encoding='utf-8') as f:
                main_scan_data = json.load(f)
        except json.JSONDecodeError:
            print(f"  Error: Could not decode JSON from {file_path}. Skipping.")
            continue
        except FileNotFoundError:
            print(f"  Error: File not found {file_path}. Skipping.")
            continue

        trivy_results_dict = main_scan_data.get("trivy_results")
        if not trivy_results_dict or not isinstance(trivy_results_dict, dict):
            print(f"  No 'trivy_results' dictionary found or it's not a dict in {file_path}. Skipping Trivy analysis for this file.")
            continue

        for llm_name, trivy_json_str in trivy_results_dict.items():
            if trivy_json_str is None or trivy_json_str.lower() == "null":
                continue

            try:
                trivy_scan_data = json.loads(trivy_json_str)
            except json.JSONDecodeError:
                print(f"  Error: Could not decode Trivy JSON string for LLM '{llm_name}' in {file_path}. Skipping this LLM.")
                continue

            if not trivy_scan_data:
                continue

            if "Results" in trivy_scan_data and trivy_scan_data["Results"]:
                for result_item in trivy_scan_data["Results"]:
                    if result_item and "Vulnerabilities" in result_item and result_item["Vulnerabilities"]:
                        for vuln in result_item["Vulnerabilities"]:
                            severity = vuln.get("Severity", "UNKNOWN").upper() # Standardize to uppercase
                            status = vuln.get("Status", "UNKNOWN").lower() # Standardize to lowercase

                            aggregated_data[llm_name]["severities"][severity] += 1
                            aggregated_data[llm_name]["statuses"][status] += 1
                            aggregated_data[llm_name]["total_vulnerabilities"] += 1

                            overall_severities[severity] += 1
                            overall_statuses[status] += 1
                            total_vulnerabilities_overall +=1

    # --- Generate Plots ---
    print("\n\n--- Generating Plots ---")

    if not aggregated_data:
        print("No vulnerability data found to plot.")
        return

    # Plot per LLM
    for llm_name, data in sorted(aggregated_data.items()):
        print(f"\nGenerating plots for Model: {llm_name}")

        # Sanitize LLM name for filename
        safe_llm_name = llm_name.replace('.', '_').replace('-', '_')

        plot_and_save_histogram(
            data_counter=data["severities"],
            title=f"Vulnerability Severities for {LLM_NAME_DICT[llm_name]}\n(Total: {data['total_vulnerabilities']})",
            xlabel="Severity",
            ylabel="Number of Vulnerabilities",
            filename_base=f"llm_{safe_llm_name}_severities",
            output_dir=OUTPUT_PLOTS_DIR,
            category_order=SEVERITY_ORDER
        )
        plot_and_save_histogram(
            data_counter=data["statuses"],
            title=f"Vulnerability Fix Statuses for {LLM_NAME_DICT[llm_name]}\n(Total: {data['total_vulnerabilities']})",
            xlabel="Fix Status",
            ylabel="Number of Vulnerabilities",
            filename_base=f"llm_{safe_llm_name}_statuses",
            output_dir=OUTPUT_PLOTS_DIR,
            category_order=STATUS_ORDER
        )

    # Plot overall summaries
    print("\nGenerating overall summary plots...")
    plot_and_save_histogram(
        data_counter=overall_severities,
        title=f"Overall Vulnerability Severities (All Models)\n(Total: {total_vulnerabilities_overall})",
        xlabel="Severity",
        ylabel="Number of Vulnerabilities",
        filename_base="overall_severities",
        output_dir=OUTPUT_PLOTS_DIR,
        category_order=SEVERITY_ORDER
    )
    plot_and_save_histogram(
        data_counter=overall_statuses,
        title=f"Overall Vulnerability Fix Statuses (All Models)\n(Total: {total_vulnerabilities_overall})",
        xlabel="Fix Status",
        ylabel="Number of Vulnerabilities",
        filename_base="overall_statuses",
        output_dir=OUTPUT_PLOTS_DIR,
        category_order=STATUS_ORDER
    )

if __name__ == "__main__":
    analyze_trivy_results_and_plot()
    print("\n--- Script Finished ---")
