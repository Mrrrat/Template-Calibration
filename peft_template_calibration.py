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
from transformers import DataCollatorForLanguageModeling, get_linear_schedule_with_warmup
import bitsandbytes as bnb

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
        tokenizer = AutoTokenizer.from_pretrained(model_name, padding_side="right", token=args.hf_token)
        # bnb_config = BitsAndBytesConfig(
        #    load_in_4bit=True,
        # #    bnb_4bit_quant_type="nf4",
        # #    bnb_4bit_use_double_quant=True,
        #    bnb_4bit_compute_dtype=precision
        # )
        model = AutoModelForCausalLM.from_pretrained(model_name, device_map="auto", torch_dtype=precision, token=args.hf_token) #quantization_config=bnb_config
        
        tokenizer.pad_token = tokenizer.eos_token
        model.config.pad_token_id = model.config.eos_token_id
        tokenizer.padding_side = "right"

        model.gradient_checkpointing_enable()
        model.enable_input_require_grads()
        #model = prepare_model_for_kbit_training(model)
        
        target_moules = ["q_proj", "v_proj"]
        if args.all_projections:
            target_modules = ["q_proj", "o_proj", "k_proj", "v_proj", "gate_proj", "up_proj", "down_proj"]
            
        peft_config = LoraConfig(inference_mode=False, r=8, lora_alpha=32, target_modules=target_modules, lora_dropout=0.1, peft_type=TaskType.CAUSAL_LM)

        model = get_peft_model(model, peft_config)
        
        generator = Generator(model=model, tokenizer=tokenizer)

        for dataset, seed, prediction_method, selection_method, num_shots, data_size, num_templates, select_best, max_ensemble_templates, n_train_templates in product(
                args.dataset, args.seed, args.prediction_method, args.examples_selection_method,
                args.num_shots, args.data_size, args.num_templates, args.select_best, args.max_ensemble_templates, args.n_train_templates
        ):
            print(f"Model:{model_name}, Dataset:{dataset}")
            config = {'dataset': dataset, 
                      'model': model_name, 
                      'seed': seed,
                      'example_selection_method': selection_method, 
                      'n_shots': num_shots,
                      'prediction_method': prediction_method, 
                      'eval_batch_size': args.eval_batch_size,
                      'batch_size': args.batch_size,
                      'precision': args.precision,
                      'template_seed': args.template_seed,
                      'data_size': data_size,
                      'num_templates': num_templates,
                      'select_best': select_best,
                      'epochs': args.epochs,
                      'max_ensemble_templates': max_ensemble_templates,
                      'n_train_templates': n_train_templates
                      }
            labels_loss = True

            templates = get_templates(dataset, num_shots, num_templates, args.templates_path, args.template_seed)

            train, test, labels_mp = load_split_dataset(dataset, cache_dir=None)
            print(f'Train size: {len(train)}, Test size: {len(test)}')
            train, test = train[:data_size], test[:1000]
            print(f'Train size: {len(train)}, Test size: {len(test)}')
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

                if len(train_template_probs) < max_ensemble_templates:
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

            config['Selected Templates'] = len(train_template_probs)
            
            train_template_probs = torch.stack(train_template_probs)
            test_template_probs = torch.stack(test_template_probs)

            config['Default Std'] = torch.std(test_template_probs, dim=0).mean()
            print(f"Default Std: {config['Default Std']}")
            config['Default Mean Accuracy'] = (np.array(labels)[test_template_probs.argmax(2)] == np.array(test['target'])).mean(1).mean()
            print(f"Default Mean Accuracy: {config['Default Mean Accuracy']}")
            config['Default Ensemble Accuracy'] = (np.array(labels)[train_template_probs.argmax(2)] == np.array(train['target'])).mean(1).mean()
            print(f"Default Ensemble Accuracy: {config['Default Ensemble Accuracy']}")

            train_ensemble = train_template_probs.mean(dim=0)
            test_ensemble = test_template_probs.mean(dim=0)

            train_templates = templates[:n_train_templates]

            train_dataset = EnsembleDataset([x.strip() for x in train['input']],
                               generator.tokenizer, labels, train_templates,
                               train_ensemble, examples=examples,
                               method=prediction_method)
            test_dataset = EnsembleDataset([x.strip() for x in test['input']],
                               generator.tokenizer, labels, templates,
                               test_ensemble, examples=examples,
                               method=prediction_method)
            
            collator = DataCollatorForLanguageModeling(generator.tokenizer, mlm=False)
            train_sampler = ReproducibleRandomSampler(len(train['input']) * len(train_templates), len(labels), seed=seed)
            test_sampler = ReproducibleRandomSampler(len(test['input']) * len(templates), len(labels), seed=seed)

            train_dataloader = DataLoader(train_dataset, sampler=train_sampler, batch_size=args.batch_size * len(labels), collate_fn=collator, drop_last=True)
            test_dataloader = DataLoader(test_dataset, shuffle=False, batch_size=args.batch_size * len(labels), collate_fn=collator, drop_last=True)

            #optimizer = bnb.optim.PagedAdamW8bit(generator.model.parameters(), lr=args.lr, betas=(0.9, 0.995))
            optimizer = bnb.optim.PagedAdamW(generator.model.parameters(), lr=args.lr, betas = (0.9, 0.999))
            scaler = torch.cuda.amp.GradScaler()
            scheduler = get_linear_schedule_with_warmup(optimizer,
                                                        num_warmup_steps=200,
                                                        num_training_steps=args.epochs * len(train_dataloader))

            if args.use_wandb:
                wandb.init(name=f"{dataset}_{model_name}_{num_templates}_{seed}_{data_size}",entity=args.wandb_entity, reinit=True, config=config)
            
            for epoch in tqdm(range(args.epochs)):
                generator.model.train()
                for step, batch in tqdm(enumerate(train_dataloader)):
                    loss = get_loss_(generator.model, batch, len(labels), labels_loss=False, precision=torch.float16)

                    if precision == torch.float16:
                        scaler.scale(loss).backward()
                    else:
                        loss.backward()
                    
                    if step % args.gradient_accumulation_steps == 0:
                        if precision == torch.float16:
                            scaler.unscale_(optimizer)
                            scaler.step(optimizer)
                            scaler.update()
                        else:
                            optimizer.step()
    
                        if scheduler is not None:
                            scheduler.step()
                            
                        optimizer.zero_grad()
                        wandb.log({'Loss': loss})

                generator.model.eval()
                if epoch < args.epochs - 1:
                    test_template_probs = []
                    for template in templates:
                        test_dataset = TensorDataset([x.strip() for x in test['input']],
                                                         generator.tokenizer, labels, template,
                                                         examples=examples,
                                                         method=prediction_method,
                                                         )
                        _, test_probs = predict(generator, test_dataset, labels, batch_size=args.eval_batch_size, method=prediction_method,
                                                 labels_loss=labels_loss, calibrate_dataset=None, precision=precision)
        
                        test_template_probs.append(test_probs)
                    
                    test_template_probs = torch.stack(test_template_probs)
                    print(f"Trained Std after {epoch} epoch: {torch.std(test_template_probs, dim=0).mean()}")
                    print(f"Trained Mean Accuracy after {epoch} epoch: {(np.array(labels)[test_template_probs.argmax(2)] == np.array(test['target'])).mean(1).mean()}")

            generator.model.eval()

            test_template_probs = []
            for template in templates:
                test_dataset = TensorDataset([x.strip() for x in test['input']],
                                                 generator.tokenizer, labels, template,
                                                 examples=examples,
                                                 method=prediction_method,
                                                 )
                _, test_probs = predict(generator, test_dataset, labels, batch_size=args.eval_batch_size, method=prediction_method,
                                         labels_loss=labels_loss, calibrate_dataset=None, precision=precision)

                test_template_probs.append(test_probs)
            
            test_template_probs = torch.stack(test_template_probs)

            config['Trained Std'] = torch.std(test_template_probs, dim=0).mean()
            print(f"Trained Std: {config['Trained Std']}")
            config['Trained Mean Accuracy'] = (np.array(labels)[test_template_probs.argmax(2)] == np.array(test['target'])).mean(1).mean()
            print(f"Trained Mean Accuracy: {config['Trained Mean Accuracy']}")
            
            results_df = save_results_pd(results_df, config, save_dir=args.save_dir)

            
