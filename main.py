import functools
import os
import sys
import pandas as pd
import shutil
import questionary
from rich.console import Console
from rich.panel import Panel
from tf_agents.system import system_multiprocessing as multiprocessing
from absl import app, flags

import experiments.experiment_ppo as experiment_ppo

FLAGS = flags.FLAGS
flags.DEFINE_boolean('clean_run', False, 'Clean the checkpoints and results folders before running')

def main(_):
    if FLAGS.clean_run:
        for folder in ['checkpoints', 'results']:
            for filename in os.listdir(folder):
                file_path = os.path.join(folder, filename)
                try:
                    if os.path.isfile(file_path) or os.path.islink(file_path):
                        os.unlink(file_path)
                    elif os.path.isdir(file_path):
                        shutil.rmtree(file_path)
                except Exception as e:
                    print('Failed to delete %s. Reason: %s' % (file_path, e))

    console = Console()

    header_panel = Panel(
        "[bold blue]Next-Financial-Decision-Model[/bold blue] Version 0.0.1",
        title = "[bold magenta]WELCOME[/bold magenta]",
        subtitle = "[italic]EXPERIMENTAL - USE AT OWN RISK[/italic]",
        expand = True
    )

    console.print(header_panel)
    options = ["Train Model", "Exit"]

    while True:
        choice = questionary.select(
            "Here is what you can do:",
            choices=options
        ).ask()

        if choice == "Train Model":
            data_dir = "data"
            data_files = os.listdir(data_dir)
            data_options = [os.path.join(data_dir, file) for file in data_files]
            
            if len(data_options) == 0:
                console.print("No data files found in the data directory. Please add data files to the data directory and try again.")
                continue

            choice = questionary.select(
                "Choose a dataset to train on:",
                choices=data_options
            ).ask()

            data_set_df = pd.read_csv(choice)
            experiment_ppo.train(data_set_df)

            break

        elif choice == "Exit":
            sys.exit()

    sys.exit()

if __name__ == "__main__":
    multiprocessing.handle_main(functools.partial(app.run, main))