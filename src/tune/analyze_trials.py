#!/usr/bin/env python3
import optuna
import argparse
import json # For JSON output
import csv # For CSV output
import sys

def analyze_study_data(study_name, storage_url, output_file=None, output_format='txt'):
    """
    Loads an Optuna study from a database and prints raw data for each trial,
    including hyperparameters and intermediate success rates.
    """
    try:
        study = optuna.load_study(study_name=study_name, storage=storage_url)
        print(f"Successfully loaded study '{study_name}' with {len(study.trials)} trials.")
    except KeyError:
        print(f"Error: Study '{study_name}' not found in the storage '{storage_url}'.")
        print("Please make sure the --study-name and --storage arguments are correct.")
        return
    except Exception as e:
        if "has no trials" in str(e):
            print(f"Study '{study_name}' loaded successfully, but it contains no trials yet.")
            return
        print(f"An error occurred while loading the study: {e}")
        return

    all_trial_data = [] # To store structured data for CSV/JSON
    for trial in study.trials:
        trial_data = {
            "trial_number": trial.number,
            "state": trial.state.name, # Use the human-readable name
            "max_success_rate": trial.value if trial.value is not None else "N/A",
            "hyperparameters": trial.params,
            "intermediate_results": trial.intermediate_values if trial.intermediate_values else {}
        }
        all_trial_data.append(trial_data)

    if output_file:
        try:
            if output_format == 'json':
                with open(output_file, 'w') as f:
                    json.dump(all_trial_data, f, indent=2, default=str)
                print(f"\nRaw data saved to {output_file} in JSON format.")
            elif output_format == 'csv':
                headers = set()
                for td in all_trial_data:
                    headers.update(td["hyperparameters"].keys())
                fieldnames = ["trial_number", "state", "max_success_rate"] + sorted(list(headers)) + ["intermediate_results"]
                with open(output_file, 'w', newline='') as f:
                    writer = csv.DictWriter(f, fieldnames=fieldnames, extrasaction='ignore')
                    writer.writeheader()
                    for td in all_trial_data:
                        row = {
                            "trial_number": td["trial_number"],
                            "state": td["state"],
                            "max_success_rate": f"{td['max_success_rate']:.2f}" if isinstance(td['max_success_rate'], float) else td['max_success_rate']
                        }
                        row.update(td["hyperparameters"])
                        row["intermediate_results"] = json.dumps(td["intermediate_results"]) if td["intermediate_results"] else "N/A"
                        writer.writerow(row)
                print(f"\nRaw data saved to {output_file} in CSV format.")
            else: # Default to text
                output_lines = [f"\n{'='*80}", f"RAW DATA FOR OPTUNA STUDY: '{study_name}'", f"{'='*80}\n"]
                for td in all_trial_data:
                    output_lines.append(f"--- TRIAL {td['trial_number']} ---")
                    output_lines.append(f"  State: {td['state']}")
                    output_lines.append(f"  Value (Max Success Rate): {td['max_success_rate']:.2f}%" if isinstance(td['max_success_rate'], float) else f"  Value: {td['max_success_rate']}")
                    output_lines.append("  Hyperparameters:")
                    if td['hyperparameters']:
                        for key, value in td['hyperparameters'].items():
                            output_lines.append(f"    {key}: {value}")
                    else:
                        output_lines.append("    No hyperparameters recorded.")
                    output_lines.append("  Intermediate Results (Success Rate at Evaluation Episodes):")
                    if td['intermediate_results']:
                        for step, value in td['intermediate_results'].items():
                            output_lines.append(f"    Ep {step}: {value:.1f}%")
                    else:
                        output_lines.append("    No intermediate values recorded.")
                    output_lines.append("-" * 40 + "\n")
                with open(output_file, 'w') as f:
                    f.write("".join(output_lines))
                print(f"\nRaw data saved to {output_file} in plain text format.")
        except Exception as e:
            print(f"Error saving output to file {output_file}: {e}")
    else:
        # Print to console
        print(f"\n{'='*80}\nRAW DATA FOR OPTUNA STUDY: '{study_name}'\n{'='*80}\n")
        for td in all_trial_data:
            print(f"--- TRIAL {td['trial_number']} ---")
            print(f"  State: {td['state']}")
            print(f"  Value (Max Success Rate): {td['max_success_rate']:.2f}%" if isinstance(td['max_success_rate'], float) else f"  Value: {td['max_success_rate']}")
            print("  Hyperparameters:")
            if td['hyperparameters']:
                for key, value in td['hyperparameters'].items():
                    print(f"    {key}: {value}")
            else:
                print("    No hyperparameters recorded.")
            print("  Intermediate Results (Success Rate at Evaluation Episodes):")
            if td['intermediate_results']:
                for step, value in td['intermediate_results'].items():
                    print(f"    Ep {step}: {value:.1f}%")
            else:
                print("    No intermediate values recorded.")
            print("-" * 40)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Analyze Optuna study raw data.")
    parser.add_argument('--study-name', type=str, default="sac-lstm-tuning",
                        help="Name of the Optuna study to load.")
    parser.add_argument('--storage', type=str, default="sqlite:///sac_tuning.db",
                        help="Database URL for Optuna storage.")
    parser.add_argument('--output-file', type=str, default=None,
                        help="Optional: Path to save the output to a file instead of printing to console.")
    parser.add_argument('--output-format', type=str, default='txt', choices=['txt', 'csv', 'json'],
                        help="Format for the output file (txt, csv, or json). Only applicable if --output-file is provided.")
    args = parser.parse_args()

    analyze_study_data(args.study_name, args.storage, args.output_file, args.output_format)