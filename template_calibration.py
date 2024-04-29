from itertools import product

from data import load_split_dataset, TensorDataset
from evaluate import evaluate_setup, predict
from examples import get_examples
from models import load_generator
from templates import get_templates
from utils import parse_args, get_results_pd, find_current_run, save_results_pd, save_results_nirvana

import numpy as np
import os
import torch
import math
from torch import nn
from tqdm.auto import tqdm
try:
    import wandb
except ImportError:
    wandb = None

class Linear3d(nn.Module):
    def __init__(self, n_templates, input_dim, output_dim):
        super().__init__()
        self.input_dim = input_dim
        self.output_dim = output_dim
        self.W = nn.Parameter(torch.Tensor(n_templates, input_dim, output_dim))
        self.b = nn.Parameter(torch.Tensor(n_templates, 1, output_dim))
        
        nn.init.xavier_uniform_(self.W, gain=nn.init.calculate_gain('relu'))
        fan_in, _ = nn.init._calculate_fan_in_and_fan_out(self.W)
        bound = 1 / math.sqrt(fan_in)
        nn.init.uniform_(self.b, -bound, bound)

    def forward(self, input):
        return torch.einsum('tsc,tcd->tsd', input, self.W) + self.b

    def make_linear(self, n):
        W = self.W[n]
        b = self.b[n][0]
        with torch.no_grad():
            layer = nn.Linear(self.input_dim, self.output_dim)
            layer.weight.copy_(W.T)
            layer.bias.copy_(b)
            return layer
   
class TemplateCalibrator(nn.Module):
    def __init__(self, n_templates, n_classes):
        super().__init__()

        self.model = nn.Sequential(
            Linear3d(n_templates, n_classes, 32*n_classes),
            nn.Sigmoid(),
            Linear3d(n_templates, 32*n_classes, 32*n_classes),
            nn.Sigmoid(),
            Linear3d(n_templates, 32*n_classes, n_classes),
            nn.Softmax(dim=2)
        )
        self.n_templates = n_templates
        self.n_classes = n_classes
            
    def forward(self, input):
        return (self.model(input))

    def get_model(self, n):
        model = nn.Sequential(
            Linear(n_classes, 128),
            nn.Sigmoid(),
            Linear(128, 128),
            nn.Sigmoid(),
            Linear(128, n_classes),
            nn.Softmax(dim=2)
        )
        with torch.no_grad():
            model[0] = self.model[0].make_linear(n)
            model[2] = self.model[2].make_linear(n)
            model[4] = self.model[4].make_linear(n)
        return model

def calc_loss_mean(output, x, sigma):
    # \sum_k\sum_i D_KL (Pki_Wi || (1/k * \sum_k Pki_Wi))
    mean_x = x.mean(0, keepdim=True).repeat(x.size(0), 1, 1)
    main_loss = nn.KLDivLoss()(torch.log(output), mean_x)
    return main_loss, main_loss, 0
    
def calc_loss_pairwise(output, x, sigma):
    # \sum_k\sum_{i,j} D_KL (Pki_Wi || Pkj_Wj) + \sigma \sum_i D_KL (Pki_Wi || Pk*)
    output_ = torch.transpose(output, 0, 1)
    output_ = output_.unsqueeze(2)
    output_ = output_.repeat(1, 1, output_.size(1), 1)
    output_t = torch.transpose(output_, 1, 2)
    log_output_, log_output_t, log_output, log_x = torch.log(output_), torch.log(output_t),torch.log(output), torch.log(x)
    main_loss = (output_ * (log_output_ - log_output_t)).mean()
    gan_loss = (x * (log_x - log_output)).mean()
    return main_loss + sigma * gan_loss, main_loss, gan_loss


if __name__ == "__main__":
    rank = int(os.environ['RANK']) if 'RANK' in os.environ else 0
    args = parse_args()
    results_df = get_results_pd(args.save_dir)

    for model in args.models:
        precision = torch.float16 if args.precision == 'fp16' else torch.bfloat16 if args.precision == 'bf16' else torch.float32 if args.precision == 'fp32' else torch.int8
        generator = load_generator(model, cache_dir=args.cache_dir, precision=args.precision,
                                   local_files_only=args.local_files_only, device_map=rank,
                                   )
        for dataset, seed, prediction_method, selection_method, num_shots, data_size, num_templates, select_best in product(
                args.dataset, args.seed, args.prediction_method, args.examples_selection_method,
                args.num_shots, args.data_size, args.num_templates, args.select_best
        ):
            print(f"Model:{model}, Dataset:{dataset}")
            labels_loss = True

            templates = get_templates(dataset, num_shots, num_templates, args.templates_path, args.template_seed)

            train, val, labels_mp = load_split_dataset(dataset, cache_dir=None)
            train, val = train[:data_size], train[data_size:data_size * 2 + 1000]
            labels = list(labels_mp.values())

            selected_examples = get_examples(dataset, train, selection_method=selection_method, seed=seed, num_shots=num_shots,
                                                 example_ids=None,
                                                 examples_path=None,
                                                 )
            examples, example_ids = selected_examples["examples"], selected_examples["example_ids"]

            template_probs = []
            
            cur_acc = 0
            cur_pred = None
            for template in templates:
                eval_dataset = TensorDataset([x.strip() for x in val['input']],
                                                 generator.tokenizer, labels, template,
                                                 examples=examples,
                                                 method=prediction_method,
                                                 )
                results, probs = predict(generator, eval_dataset, labels, batch_size=args.eval_batch_size, method=prediction_method,
                                         labels_loss=labels_loss, calibrate_dataset=None, precision=precision)

                if select_best:
                    if cur_pred is not None:
                        cur_pred += probs
                    else:
                        cur_pred = probs
                        
                    acc = (np.array(labels)[cur_pred.argmax(1)] == np.array(val['target'])).mean()
                    if acc > cur_acc:
                        template_probs.append(probs)
                        cur_acc = acc
                    else:
                        cur_pred -= probs
                else:
                    template_probs.append(probs)

            x = torch.stack(template_probs)

            for sigma, loss in product(args.sigma, args.loss):
                method_name = f"{prediction_method}_{labels_loss}_{loss}"
                # config is used to store and find results saved locally
                config = {'dataset': dataset, 'model': model, 'seed': seed,
                          'example_selection_method': selection_method, 'n_shots': num_shots,
                          'prediction_method': method_name, 'batch_size': args.eval_batch_size,
                          'precision': args.precision,
                          'template_seed': args.template_seed,
                          'steps': args.steps,
                          'lr': args.lr,
                          'sigma': sigma,
                          'loss': loss,
                          'data_size': data_size,
                          'num_templates': num_templates,
                          'selected_templates': len(template_probs),
                          'select_best': select_best
                          }
    
                if args.use_wandb:
                    wandb.init(name=f"{dataset}_{model}_{num_templates}_{loss}_{seed}_{data_size}",entity=args.wandb_entity, reinit=True, config=config)
    
                calibrator = TemplateCalibrator(len(template_probs), len(labels))
      
                optimizer = torch.optim.Adam(calibrator.parameters(), lr=args.lr)
                scheduler = torch.optim.lr_scheduler.ExponentialLR(optimizer, gamma=0.9995)

                train_, test_ = x[:, :data_size], x[:, data_size:]
                train_target, test_target = train_.clone().detach(), test_.clone().detach()
                
                train_default_acc = (np.array(labels)[train_.argmax(2)] == np.array(val['target'][:data_size])).mean(1).mean()
                test_default_acc = (np.array(labels)[test_.argmax(2)] == np.array(val['target'][data_size:])).mean(1).mean()
                test_default_acc_best = (np.array(labels)[test_.argmax(2)] == np.array(val['target'][data_size:])).mean(1).max()
                ens_train_default_acc = (np.array(labels)[train_.mean(0).argmax(1)] == np.array(val['target'][:data_size])).mean()
                ens_test_default_acc = (np.array(labels)[test_.mean(0).argmax(1)] == np.array(val['target'][data_size:])).mean()
                
                wandb.log({'Train Default Accuracy': train_default_acc,
                           'Test Default Accuracy': test_default_acc,
                           'Ensemble Train Default Accuracy': ens_train_default_acc,
                           'Ensemble Test Default Accuracy': ens_test_default_acc
                          })

                calc_loss = None
                if loss == 'pairwise': 
                    calc_loss = calc_loss_pairwise
                elif loss == 'mean':
                    calc_loss = calc_loss_mean
                else:
                    raise NotImplementedError
                
                for epoch in tqdm(range(args.steps)):
                    calibrator.train()
                    optimizer.zero_grad()             
                    train_output = calibrator(train_)
                    train_acc = (np.array(labels)[train_output.argmax(2)] == np.array(val['target'][:data_size])).mean()
                    train_loss, main_train_loss, gan_train_loss = calc_loss(train_output, train_target, sigma)
                    main_train_loss.backward()
                    optimizer.step()
                    scheduler.step()
                
                    calibrator.eval()
                    test_output = calibrator(test_)
                    test_acc_mean = (np.array(labels)[test_output.argmax(2)] == np.array(val['target'][data_size:])).mean()
                    test_acc_best = (np.array(labels)[test_output.argmax(2)] == np.array(val['target'][data_size:])).mean(axis=1).max()
                    
                    test_loss, main_test_loss, gan_test_loss = calc_loss(test_output, test_target, sigma)
                    
                    wandb.log({'Train Loss': train_loss,
                               'Train Main Loss' : main_train_loss,
                               'Train Gan Loss': gan_train_loss,
                               'Test Loss': test_loss,
                               'Test Main Loss' : main_test_loss,
                               'Test Gan Loss': gan_test_loss,
                               'Train Accuracy': train_acc,
                               'Test Accuracy Mean': test_acc_mean,
                               'Test Accuracy Best': test_acc_best
                                })
                    
    
                calibrator.eval()
                default_std = torch.std(test_, dim=1).mean()
                pred = calibrator(test_)
                trained_std = torch.std(pred, dim=1).mean()
                _, default_kl, _ = calc_loss_pairwise(test_, test_, 0)
                _, trained_kl, _ = calc_loss_pairwise(pred, pred, 0)            
                wandb.log({'Default Std': default_std,
                           'Trained Std': trained_std,
                           'Default KL': default_kl,
                           'Trained KL': trained_kl
                          })
    
                torch.save(calibrator.state_dict(), 'calibrator.pt')
                config.update({'Test Default Accuracy Mean': round(test_default_acc, 4),
                               'Test Default Accuracy Best': round(test_default_acc_best, 4),
                               'Test Ensemble Accuracy': round(ens_test_default_acc, 4),
                               'Test Calibrated Accuracy Mean': round(test_acc_mean, 4),
                               'Test Calibrated Accuracy Best': round(test_acc_best, 4),
                               'Default Std': round(default_std.item(), 4),
                               'Trained Std': round(trained_std.item(), 4),
                               'Default KL': round(default_kl.item(), 4),
                               'Trained KL': round(trained_kl.item(), 4)
                              })
                results_df = save_results_pd(results_df, config, save_dir=args.save_dir)
                
            
