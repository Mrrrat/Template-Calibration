from itertools import product

from data import load_split_dataset, TensorDataset, ReproducibleRandomSampler, EnsembleDataset
from evaluate import evaluate_setup, predict, get_loss_
from examples import get_examples
from models import load_generator, Generator
from templates import get_templates
from utils import parse_args, get_results_pd, find_current_run, save_results_pd, save_results_nirvana

from transformers import AutoTokenizer, AutoModelForCausalLM, BitsAndBytesConfig
from peft import get_peft_model, LoraConfig, TaskType, prepare_model_for_kbit_training
from torch.utils.data import Dataset, DataLoader
from transformers import DataCollatorForLanguageModeling

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


if __name__ == "__main__":
    rank = int(os.environ['RANK']) if 'RANK' in os.environ else 0
    args = parse_args()
    results_df = get_results_pd(args.save_dir)

    for model_name in args.models:
        precision = torch.float16 if args.precision == 'fp16' else torch.bfloat16 if args.precision == 'bf16' else torch.float32 if args.precision == 'fp32' else torch.int8
        tokenizer = AutoTokenizer.from_pretrained(model_name, padding_side="right", token='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')
        bnb_config = BitsAndBytesConfig(
           load_in_4bit=True,
        #    bnb_4bit_quant_type="nf4",
        #    bnb_4bit_use_double_quant=True,
           bnb_4bit_compute_dtype=precision
        )
        model = AutoModelForCausalLM.from_pretrained(model_name, device_map="auto", quantization_config=bnb_config, token='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')
        
        if 'llama' in model_name and 'llama-3' not in model_name:
            tokenizer.add_special_tokens({'pad_token': '[PAD]'})
            tokenizer.pad_token_id = tokenizer.unk_token_id
            model.config.pad_token_id = tokenizer.unk_token_id
        else:
            tokenizer.pad_token = tokenizer.eos_token
            model.config.pad_token_id = model.config.eos_token_id

       # for param in model.parameters():
       #    param.requires_grad = False  # freeze the model - train adapters later
       #    if param.ndim == 1:
       #      # cast the small parameters (e.g. layernorm) to fp32 for stability
       #      param.data = param.data.to(torch.float32)

        model.gradient_checkpointing_enable()
        model.enable_input_require_grads()
        model = prepare_model_for_kbit_training(model)
        
        peft_config = LoraConfig(inference_mode=False, r=8, lora_alpha=32, lora_dropout=0.1, peft_type=TaskType.CAUSAL_LM)
        # target_modules=["q_proj", "v_proj"]

        model = get_peft_model(model, peft_config)
        
        generator = Generator(model=model, tokenizer=tokenizer)

        for dataset, seed, prediction_method, selection_method, num_shots, data_size, num_templates, select_best in product(
                args.dataset, args.seed, args.prediction_method, args.examples_selection_method,
                args.num_shots, args.data_size, args.num_templates, args.select_best
        ):
            print(f"Model:{model_name}, Dataset:{dataset}")
            labels_loss = True

            templates = get_templates(dataset, num_shots, num_templates, args.templates_path, args.template_seed)

            train, test, labels_mp = load_split_dataset(dataset, cache_dir=None)
            train, test = train[:data_size], test[:10]
            labels = list(labels_mp.values())

            selected_examples = get_examples(dataset, train, selection_method=selection_method, seed=seed, num_shots=num_shots,
                                                 example_ids=None,
                                                 examples_path=None,
                                                 )
            
            examples, example_ids = selected_examples["examples"], selected_examples["example_ids"]
            
            train_template_probs, test_template_probs = [], []
            
            cur_acc = 0
            cur_pred = None
            for template in templates:
                test_dataset = TensorDataset([x.strip() for x in test['input']],
                                                 generator.tokenizer, labels, template,
                                                 examples=examples,
                                                 method=prediction_method,
                                                 )
                _, test_probs = predict(generator, test_dataset, labels, batch_size=args.eval_batch_size, method=prediction_method,
                                         labels_loss=labels_loss, calibrate_dataset=None, precision=precision)
                
                
                train_dataset = TensorDataset([x.strip() for x in train['input']],
                                                 generator.tokenizer, labels, template,
                                                 examples=examples,
                                                 method=prediction_method,
                                                 )
                _, train_probs = predict(generator, train_dataset, labels, batch_size=args.eval_batch_size, method=prediction_method,
                                         labels_loss=labels_loss, calibrate_dataset=None, precision=precision)
                

                if select_best:
                    if cur_pred is not None:
                        cur_pred += train_probs
                    else:
                        cur_pred = torch.clone(train_probs)
                        
                    acc = (np.array(labels)[cur_pred.argmax(1)] == np.array(train['target'])).mean()
                    if acc > cur_acc:
                        train_template_probs.append(train_probs)
                        cur_acc = acc
                    else:
                        cur_pred -= train_probs
                else:
                    train_template_probs.append(train_probs)

                test_template_probs.append(test_probs)

            print

            train_ensemble = torch.stack(train_template_probs).mean(dim=0)
            test_ensemble = torch.stack(test_template_probs).mean(dim=0)

            train_dataset = EnsembleDataset([x.strip() for x in train['input']],
                               generator.tokenizer, labels, templates,
                               train_ensemble, examples=examples,
                               method=prediction_method)
            test_dataset = EnsembleDataset([x.strip() for x in test['input']],
                               generator.tokenizer, labels, templates,
                               test_ensemble, examples=examples,
                               method=prediction_method)
            
            collator = DataCollatorForLanguageModeling(generator.tokenizer, mlm=False)
            train_sampler = ReproducibleRandomSampler(len(train['input']) * len(templates), len(labels), seed=seed)
            test_sampler = ReproducibleRandomSampler(len(test['input']) * len(templates), len(labels), seed=seed)

            train_dataloader = DataLoader(train_dataset, sampler=train_sampler, batch_size=args.batch_size * len(labels), collate_fn=collator, drop_last=True)
            test_dataloader = DataLoader(test_dataset, sampler=test_sampler, batch_size=args.batch_size * len(labels), collate_fn=collator)

            for batch in tqdm(train_dataloader):
                loss = get_loss_(model, batch, len(labels), labels_loss=False, precision=torch.float16)
                print(loss)
                break

            # for sigma, loss in product(args.sigma, args.loss):
            #     method_name = f"{prediction_method}_{labels_loss}_{loss}"
            #     # config is used to store and find results saved locally
            #     config = {'dataset': dataset, 'model': model, 'seed': seed,
            #               'example_selection_method': selection_method, 'n_shots': num_shots,
            #               'prediction_method': method_name, 'batch_size': args.eval_batch_size,
            #               'precision': args.precision,
            #               'template_seed': args.template_seed,
            #               'steps': args.steps,
            #               'lr': args.lr,
            #               'sigma': sigma,
            #               'loss': loss,
            #               'data_size': data_size,
            #               'num_templates': num_templates,
            #               'selected_templates': len(template_probs),
            #               'select_best': select_best
            #               }
    
            #     if args.use_wandb:
            #         wandb.init(name=f"{dataset}_{model}_{num_templates}_{loss}_{seed}_{data_size}",entity=args.wandb_entity, reinit=True, config=config)
    
            #     calibrator = TemplateCalibrator(len(template_probs), len(labels))
      
            #     optimizer = torch.optim.Adam(calibrator.parameters(), lr=args.lr)
            #     scheduler = torch.optim.lr_scheduler.ExponentialLR(optimizer, gamma=0.9995)

            #     train_, test_ = x[:, :data_size], x[:, data_size:]
            #     train_target, test_target = train_.clone().detach(), test_.clone().detach()
                
            #     train_default_acc = (np.array(labels)[train_.argmax(2)] == np.array(val['target'][:data_size])).mean(1).mean()
            #     test_default_acc = (np.array(labels)[test_.argmax(2)] == np.array(val['target'][data_size:])).mean(1).mean()
            #     test_default_acc_best = (np.array(labels)[test_.argmax(2)] == np.array(val['target'][data_size:])).mean(1).max()
            #     ens_train_default_acc = (np.array(labels)[train_.mean(0).argmax(1)] == np.array(val['target'][:data_size])).mean()
            #     ens_test_default_acc = (np.array(labels)[test_.mean(0).argmax(1)] == np.array(val['target'][data_size:])).mean()
                
            #     wandb.log({'Train Default Accuracy': train_default_acc,
            #                'Test Default Accuracy': test_default_acc,
            #                'Ensemble Train Default Accuracy': ens_train_default_acc,
            #                'Ensemble Test Default Accuracy': ens_test_default_acc
            #               })

            #     calc_loss = None
            #     if loss == 'pairwise': 
            #         calc_loss = calc_loss_pairwise
            #     elif loss == 'mean':
            #         calc_loss = calc_loss_mean
            #     else:
            #         raise NotImplementedError
                
            #     for epoch in tqdm(range(args.steps)):
            #         calibrator.train()
            #         optimizer.zero_grad()             
            #         train_output = calibrator(train_)
            #         train_acc = (np.array(labels)[train_output.argmax(2)] == np.array(val['target'][:data_size])).mean()
            #         train_loss, main_train_loss, gan_train_loss = calc_loss(train_output, train_target, sigma)
            #         main_train_loss.backward()
            #         optimizer.step()
            #         scheduler.step()
                
            #         calibrator.eval()
            #         test_output = calibrator(test_)
            #         test_acc_mean = (np.array(labels)[test_output.argmax(2)] == np.array(val['target'][data_size:])).mean()
            #         test_acc_best = (np.array(labels)[test_output.argmax(2)] == np.array(val['target'][data_size:])).mean(axis=1).max()
                    
            #         test_loss, main_test_loss, gan_test_loss = calc_loss(test_output, test_target, sigma)
                    
            #         wandb.log({'Train Loss': train_loss,
            #                    'Train Main Loss' : main_train_loss,
            #                    'Train Gan Loss': gan_train_loss,
            #                    'Test Loss': test_loss,
            #                    'Test Main Loss' : main_test_loss,
            #                    'Test Gan Loss': gan_test_loss,
            #                    'Train Accuracy': train_acc,
            #                    'Test Accuracy Mean': test_acc_mean,
            #                    'Test Accuracy Best': test_acc_best
            #                     })
                    
    
            #     calibrator.eval()
            #     default_std = torch.std(test_, dim=1).mean()
            #     pred = calibrator(test_)
            #     trained_std = torch.std(pred, dim=1).mean()
            #     _, default_kl, _ = calc_loss_pairwise(test_, test_, 0)
            #     _, trained_kl, _ = calc_loss_pairwise(pred, pred, 0)            
            #     wandb.log({'Default Std': default_std,
            #                'Trained Std': trained_std,
            #                'Default KL': default_kl,
            #                'Trained KL': trained_kl
            #               })
    
            #     torch.save(calibrator.state_dict(), 'calibrator.pt')
            #     config.update({'Test Default Accuracy Mean': round(test_default_acc, 4),
            #                    'Test Default Accuracy Best': round(test_default_acc_best, 4),
            #                    'Test Ensemble Accuracy': round(ens_test_default_acc, 4),
            #                    'Test Calibrated Accuracy Mean': round(test_acc_mean, 4),
            #                    'Test Calibrated Accuracy Best': round(test_acc_best, 4),
            #                    'Default Std': round(default_std.item(), 4),
            #                    'Trained Std': round(trained_std.item(), 4),
            #                    'Default KL': round(default_kl.item(), 4),
            #                    'Trained KL': round(trained_kl.item(), 4)
            #                   })
            #     results_df = save_results_pd(results_df, config, save_dir=args.save_dir)
                
            
