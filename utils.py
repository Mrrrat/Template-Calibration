import argparse
import os
from argparse import ArgumentParser
from pathlib import Path

import pandas as pd
import torch

from models import NAMES_TO_CHECKPOINTS
try:
    import nirvana_dl
except ImportError:
    nirvana_dl = None
try:
    import wandb
except ImportError:
    wandb = None

def str2bool(v):
    if v.lower() in ('yes', 'true', 't', 'y', '1'):
        return True
    elif v.lower() in ('no', 'false', 'f', 'n', '0'):
        return False
    else:
        raise argparse.ArgumentTypeError('Boolean value expected.')


def parse_args():
    parser = ArgumentParser()
    parser.add_argument("-d", "--dataset", required=True, help='dataset name', nargs='+',
                        choices=['sst2', 'dbpedia', 'agnews', 'trec'])
    parser.add_argument('-m', '--models', required=True, help='models name', nargs='+', type=str)
    parser.add_argument("--seed", help='Seed for reproducibility.', type=int, default=[59], nargs='+')
    parser.add_argument("--num_shots", type=int, help='number of examples for ICL.', default=[0], nargs='+')
    # examples selection
    parser.add_argument("--examples_selection_method", required=True, nargs='+',
                        help="method for selecting examples for ICL.")
    parser.add_argument("--example_ids", type=int, nargs="+",
                        help="ids of the train samples to use as examples for ICL.")
    parser.add_argument("--examples_path",
                        help="specify path to .json file where the retrieved examples are stored.")
    # prediction methods
    parser.add_argument("--prediction_method", default=['direct'], nargs='+',
                        choices=["direct", "channel", "calibrate"],
                        help="Method of prediction on test inputs. "
                             "It is recommended to run Channel and Calibrate methods with setting labels_loss=True."
                        )
    parser.add_argument("--labels_loss", action='store_true',
                        help="Whether to calculate loss over the whole sequence or only on the label tokens.")
    #calibration
    parser.add_argument("--num_templates", type=int, help='number of randomly generated templates.', nargs='+', default=10)
    parser.add_argument("--max_ensemble_templates", type=int, help='number of max number generated templates for ensemble.', nargs='+', default=10)
    parser.add_argument("--n_train_templates", type=int, help='number of templates for training.', nargs='+', default=10)
    parser.add_argument("--select_best", help='Select Best Templates for Ensemble', type=str2bool, nargs='+', default='False')
    # template calibration
    parser.add_argument("--steps", help='Number of steps', type=int, default=1000)
    parser.add_argument("--sigma", help='Sigma', nargs='+', type=float, default=1)
    parser.add_argument("--loss", help='Loss', nargs='+', choices=["pairwise", "mean"])
    #peft template calibration
    parser.add_argument("--batch_size", type=int, default=4,
                        help="Batch size for training")
    parser.add_argument("--epochs", type=int, default=3,
                        help="Epochs for training")
    parser.add_argument("--all_projections", action='store_true',
                        help="Whether to lora over all porjections or not.")
    # inference args
    parser.add_argument("--lr", help='Learning rate', type=float, default=3e-4)
    parser.add_argument("--data_size", help='Data Size', nargs='+', type=int, default=1000)
    parser.add_argument("--eval_batch_size", type=int, default=16,
                        help="Batch size for inference.")
    parser.add_argument("--precision", choices=['fp16', 'fp32', 'bf16', 'int8'], default='fp16',
                        help='floating point precision for inference model')
    parser.add_argument("--quantization", action='store_true',
                        help="Use of quantization")
    parser.add_argument("--gradient_accumulation_steps", help='Gradient Accumulation Steps', type=int, default=1)
    # hf args
    parser.add_argument("--cache_dir", help="Path to huggingface cache")
    parser.add_argument("--local_files_only", action='store_true',
                        help="turn this on if you want to make sure that you do not download the same weights from HF "
                             "hub again to another path occasionally.")
    
    parser.add_argument("--templates_path",
                        help="Path to a *.json file containing pre-determined set of templates.")
    parser.add_argument("--template_seed", type=int, default=59,
                        help='Seed for generating random templates.',
                        )
    # infrastructure args
    parser.add_argument("--save_dir", default=".", help="Where to save the results.")
    parser.add_argument("--use_wandb", default=True, action=argparse.BooleanOptionalAction, 
                        help="Write --no-use_wandb to disable WandB.")
    parser.add_argument("--wandb_entity", default=None)
    parser.add_argument("--hf_token", default=None)
    args = parser.parse_args()
    return args


def get_results_torch(save_dir, name="results"):
    res_path = Path(save_dir, name)

    if Path.exists(res_path):
        results = torch.load(res_path)
    else:
        results = {}
    return results


def get_results_pd(save_dir, name="results.csv"):
    res_path = Path(save_dir, name)
    if Path.exists(res_path):
        results = pd.read_csv(res_path)
    else:
        results = pd.DataFrame(columns=["dataset", "model", "seed", "example_selection_method", "n_shots",
                                        "example_ids", "prediction_method", "batch_size", "precision",
                                        "template_seed",
                                        "template", "score",
                                        ])
    return results


def find_current_run(config: dict, results: pd.DataFrame) -> list:
    """for a given setup find existing runs (if any)"""
    results_values = results[list(config)]
    found_runs = results.loc[(results_values == pd.Series(config)).all(axis=1)]
    scores = found_runs["score"].tolist()

    return scores


def save_results_torch(res_obj, save_dir, name):
    os.makedirs(save_dir, exist_ok=True)
    torch.save(res_obj, Path(save_dir, name))


# def save_results_pd(res_df, run_config, template, score, name="results.csv", save_dir="."):
#     os.makedirs(save_dir, exist_ok=True)
#     run_config.update({"template": str(template), "score": score})
#     res_df = pd.concat([res_df, pd.DataFrame([run_config])], ignore_index=True)
#     res_df.to_csv(Path(save_dir, name), index=False)

#     return res_df

def save_results_pd(res_df, config, name='results.csv', save_dir="."):
    os.makedirs(save_dir, exist_ok=True)
    res_df = pd.concat([res_df, pd.DataFrame([config])], ignore_index=True)
    res_df.to_csv(Path(save_dir, name), index=False)
    return res_df

def save_results_nirvana():
    if nirvana_dl is not None:
        nirvana_dl.snapshot.dump_snapshot()
