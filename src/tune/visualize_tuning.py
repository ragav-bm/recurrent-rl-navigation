#!/usr/bin/env python3
import optuna
import argparse
import sys

def visualize_study(study_name, storage_url):
    """
    Loads an Optuna study from a database and generates visualization plots.
    """
    try:
        # Load the study from the database file.
        study = optuna.load_study(study_name=study_name, storage=storage_url)
        print(f"Successfully loaded study '{study_name}' with {len(study.trials)} trials.")
    except KeyError:
        print(f"Error: Study '{study_name}' not found in the storage '{storage_url}'.")
        print("Please make sure the --study-name and --storage arguments are correct.")
        return
    except Exception as e:
        print(f"An error occurred while loading the study: {e}")
        return

    # --- Generate and show plots ---
    # You need a web browser to see the plots. They are interactive!

    # 1. Optimization History Plot
    # Shows the progress of the score over trials. You can see if the tuning is
    # still improving or if it has plateaued.
    try:
        print("Generating optimization history plot... (Opens in browser)")
        fig_history = optuna.visualization.plot_optimization_history(study)
        fig_history.show()
    except Exception as e:
        print(f"Could not generate optimization history plot. (Maybe not enough trials yet?): {e}")

    # 2. Parameter Importances Plot
    # Shows which hyperparameters were most influential on the outcome. This is
    # one of the most useful plots for understanding your model.
    try:
        print("Generating parameter importances plot... (Opens in browser)")
        fig_importance = optuna.visualization.plot_param_importances(study)
        fig_importance.show()
    except Exception as e:
        print(f"Could not generate parameter importances plot. (Maybe not enough trials yet?): {e}")

    # 3. Intermediate Values Plot
    # Shows the learning curves for all trials. This is the best way to see
    # why a trial was (or was not) pruned. You can see it dip below the median.
    try:
        print("Generating intermediate values plot... (Opens in browser)")
        fig_intermediate = optuna.visualization.plot_intermediate_values(study)
        fig_intermediate.show()
    except Exception as e:
        print(f"Could not generate intermediate values plot. (Maybe not enough trials yet?): {e}")

    # 3. Slice Plot
    # Shows how each hyperparameter affects the score individually.
    try:
        print("Generating slice plot... (Opens in browser)")
        fig_slice = optuna.visualization.plot_slice(study)
        fig_slice.show()
    except Exception as e:
        print(f"Could not generate slice plot. (Maybe not enough trials yet?): {e}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Visualize Optuna study results.")
    parser.add_argument('--study-name', type=str, default="sac-lstm-tuning",
                        help="Name of the Optuna study to load.")
    parser.add_argument('--storage', type=str, default="sqlite:///sac_tuning.db",
                        help="Database URL for Optuna storage.")
    args = parser.parse_args()

    # Before generating plots, you need to install plotly
    try:
        import plotly
    except ImportError:
        print("Plotly is not installed. Please install it to visualize results:")
        print("pip install plotly")
        sys.exit(1)

    visualize_study(args.study_name, args.storage)