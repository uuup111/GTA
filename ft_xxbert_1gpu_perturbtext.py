# os.environ["WANDB_MODE"] = "offline"
import glob
import logging
import os
import warnings
import random

from data.load_cora import get_raw_text_cora

warnings.filterwarnings("ignore", category=UserWarning, message="torch.utils._pytree._register_pytree_node is deprecated")
from datetime import datetime
from functools import partial
# project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
# sys.path.append(project_root)
import optuna
from alive_progress import alive_bar
from sklearn.metrics import f1_score
from data.load_data import get_dataloader
import wandb
import torch
import torch.nn as nn
from torch.cuda.amp import GradScaler, autocast
from transformers import AdamW, get_linear_schedule_with_warmup
from utils.util import init_random_state, get_class_count, save_checkpoint, load_checkpoint, load_checkpoint_2, \
    perturb_text, build_candidate_all_word_pool
from models.bert import load_pretrained_model_and_tokenizer

# 配置日志记录器
logging.basicConfig(
    level=logging.DEBUG,
    format='%(asctime)s - %(levelname)s - %(message)s',
    datefmt='%Y-%m-%d %H:%M:%S',
    filename='ft_xxbert_1gpu_perturbtext.out',
    filemode='a'
)

def train_model(model, train_dataloader, optimizer, scheduler, scaler, device):
    total_loss = 0
    model.train()
    for batch in train_dataloader:
        # input_ids, attention_mask, labels = batch
        input_ids, attention_mask, labels = [b.to(device) for b in batch]
        optimizer.zero_grad()
        with autocast():
            outputs = model(input_ids=input_ids, attention_mask=attention_mask, labels=labels)
            loss = outputs.loss
            total_loss += loss.item()
        scaler.scale(loss).backward()
        # logging.info(f"data_name:{data_name}")
        scaler.step(optimizer)
        scaler.update()
        scheduler.step()
    avg_loss = total_loss / len(train_dataloader)
    return avg_loss

def evaluate_model(model, dataloader, device):
    model.eval()
    total_loss = 0
    correct = 0
    total = 0
    all_labels = []
    all_predictions = []
    for batch in dataloader:
        # input_ids, attention_mask, labels = batch
        input_ids, attention_mask, labels = [b.to(device) for b in batch]
        with torch.no_grad():
            with autocast():
                logits = model(input_ids, attention_mask=attention_mask).logits
                loss = nn.functional.cross_entropy(logits, labels)
                total_loss += loss.item()
                predictions = torch.argmax(logits, dim=-1)
                correct += (predictions == labels).sum().item()
                total += labels.size(0)
                all_labels.extend(labels.cpu().numpy())
                all_predictions.extend(predictions.cpu().numpy())
    torch.cuda.empty_cache()
    avg_loss = total_loss / len(dataloader)
    accuracy = correct / total
    f1 = f1_score(all_labels, all_predictions, average='weighted')
    return avg_loss, accuracy, f1

def objective(trial_params, return_dict, data_name):

    # logging.info(f"data_name:{data_name}")
    if data_name is None:
        exit(0)
    try:

        if '__file__' in globals():
            current_dir = os.path.dirname(__file__)
        else:
            current_dir = os.getcwd()
        perturb_acc=[]

        for node_p in range(1, 10):
            node_p = node_p * 0.1
            for perturb_word_p in range(1, 10):
                perturb_word_p = perturb_word_p * 0.1

                test_accs = []
                test_f1s = []
                test_rocaucs = []
                device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
                random.randint(0, 9999)
                for seed in range(10):
                    logging.info(f'\nseed {seed}:\n')
                    init_random_state(seed=seed)
                    trial_params['seed'] = seed
                    model_name = trial_params['model_name']
                    # if model_name in ['roberta', 'deberta']:
                    #     continue
                    batch_size = trial_params['batch_size']
                    max_length = trial_params['max_length']
                    num_epochs = trial_params['num_epochs']
                    learning_rate = trial_params['learning_rate']
                    warmup_ratio = trial_params['warmup_ratio']
                    use_pe = trial_params['use_pe']
                    wandb.init(project=f"finetune_{model_name}_{data_name}_pe", config={"device": device})
                    wandb.login(key="aa1dda53df6b74ec3df6babb69c99a43e67db74e")
                    wandb_run_name = wandb.run.name
                    # logging.info(f"Run Name: {wandb_run_name}")
                    token_file = os.path.join(current_dir,f"res/{data_name}/pe{use_pe}_tokens_{model_name}_maxlength{max_length}.pt")
                    statedict_path = f"res/{data_name}/pe_finetune_{model_name}_bs{batch_size}_ml{max_length}_seed{seed}.pth" #lr{learning_rate}_wr_{warmup_ratio}
                    checkpoint_path = f"res/{data_name}/ckpt_pe{use_pe}_finetune_{model_name}_bs{batch_size}_ml{max_length}_seed{seed}_lr{learning_rate}_wr_{warmup_ratio}.pth.tar"
                    files = glob.glob(statedict_path)
                    if len(files) > 0:
                        continue
                    wandb.log(trial_params)
                    num_classes = get_class_count(data_name)
                    model, tokenizer = load_pretrained_model_and_tokenizer(current_dir, model_name, num_classes)

                    # train_dataloader, val_dataloader, test_dataloader, input_dim, num_classes = token_dataloader(graph, text_list, token_file, tokenizer, max_length, batch_size, input_dim, num_classes)
                    # if data_name.lower() == "pubmed":
                    #     graph, text_list, input_dim, num_classes = get_raw_text_pubmed(current_dir, use_text=use_text, use_pe=use_pe, seed=seed)
                    # elif data_name.lower() == "arxiv_2023":
                    #     graph, text_list, input_dim, num_classes = get_raw_text_arxiv_2023(current_dir, use_text=use_text, use_pe=use_pe, seed=seed)
                    # elif data_name.lower() == "cora":
                    graph, text_list, input_dim, num_classes = get_raw_text_cora(current_dir, use_text=True, use_pe=True, seed=seed)
                    # elif data_name.lower() == "ogbn_arxiv":
                    #     graph, text_list, input_dim, num_classes = get_raw_text_arxiv_origin(current_dir, use_text=use_text, use_pe=use_pe, seed=seed)
                    #
                    # elif data_name.lower() == "ogbn_products":
                    #     graph, text_list, input_dim, num_classes = get_raw_text_products(current_dir, use_text=use_text, use_pe=use_pe, seed=seed)
                    # elif data_name.lower() == "amazon_ratings":
                    #     graph, text_list, input_dim, num_classes = get_raw_text_amazon_ratings(current_dir, use_text=use_text, use_pe=use_pe, seed=seed)
                    # else:
                    #     exit(0)
                    candidate_words = build_candidate_all_word_pool(text_list, source="brown")


                    perturb_text(text_list, candidate_words, node_p, perturb_word_p)

                    train_dataloader, val_dataloader, test_dataloader, input_dim, num_classes = get_dataloader(device, token_file, seed, data_name, current_dir, tokenizer, max_length, batch_size, use_text=True, use_pe=use_pe)
                    model = model.to(device)
                    total_steps = len(train_dataloader) * num_epochs
                    optimizer = AdamW(model.parameters(), lr=learning_rate)
                    try:
                        wandb.config.optimizer = optimizer
                        # wandb.config.scheduler = scheduler
                    except Exception as e:
                        pass
                    scheduler = get_linear_schedule_with_warmup(optimizer, num_warmup_steps=int(warmup_ratio * total_steps), num_training_steps=total_steps )
                    scaler = GradScaler()
                    patience = 10
                    best_val_acc = float('-inf')
                    best_model = None
                    patience_counter = 0
                    start_epoch, loss = load_checkpoint_2(model, optimizer, scaler, filename=checkpoint_path)
                    torch.cuda.empty_cache()
                    with alive_bar(num_epochs) as bar:
                        for epoch in range(start_epoch, num_epochs):
                            model.train()
                            train_loss = train_model(model, train_dataloader, optimizer, scheduler, scaler, device)
                            # logging.info(f"epoch: {epoch + 1}, train_loss: {train_loss}")
                            model.eval()
                            with torch.no_grad():
                                val_loss, val_acc, val_f1 = evaluate_model(model, val_dataloader, device)
                                test_loss, test_acc, test_f1 = evaluate_model(model, test_dataloader, device)
                                # logging.info(f"epoch: {epoch + 1}, train_loss: {train_loss}, validation_accuracy: {val_acc}, validation_f1: {val_f1}, validation_loss: {val_loss}")
                                wandb.log({"epoch": epoch + 1, "train_loss": train_loss, "validation_accuracy": val_acc, "validation_f1": val_f1, "validation_loss": val_loss, })
                                save_checkpoint(model, optimizer, epoch, train_loss, filename=checkpoint_path)
                                if val_acc > best_val_acc:
                                    best_val_acc = val_acc
                                    best_model = model
                                    patience_counter = 0
                                else:
                                    patience_counter += 1
                                if patience_counter >= patience:
                                    # logging.info("Early stopping triggered.")
                                    break
                            bar()
                    torch.cuda.empty_cache()
                    model.eval()
                    with torch.no_grad():
                        if best_model:
                            torch.save(best_model.state_dict(), os.path.join(current_dir, statedict_path))
                            test_loss, test_acc, test_f1 = evaluate_model(best_model, test_dataloader, device)
                            # logging.info(f"Time:{datetime.now()}, Test Accuracy: {test_acc:.4f}, Test F1: {test_f1:.4f}, Test Loss: {test_loss:.4f}, ")
                            wandb.log({"test_loss": test_loss, "test_acc": test_acc, "test_f1": test_f1})
                            torch.cuda.empty_cache()
                            test_accs.append(test_acc)
                            test_f1s.append(test_f1)
                if len(test_accs) > 0:
                    test_accs = torch.tensor(test_accs)
                    test_f1s = torch.tensor(test_f1s)
                    test_acc_mean = test_accs.mean()
                    test_acc_std = test_accs.std()
                    logging.info(f'Final Test: {test_accs.mean():.4f} ± {test_accs.std():.4f}')
                    wandb.log({"Final_Test_Accuracy_Mean": test_acc_mean, "Final_Test_Accuracy_STD": test_acc_std,
                               "Final_Test_Accuracy_MAX": test_accs.max(),
                               "Final_Test_F1_Mean": test_f1s.mean(), "Final_Test_F1_STD": test_f1s.std(),
                               "Final_Test_F1_MAX": test_f1s.max()
                               }
                              )
                    return_dict['result'] = test_acc_mean if test_acc_mean is not None else float('-inf')
                else:
                    test_acc_mean = 0.0
                    test_acc_std = 0.0
                    # logging.info("No test accuracies found. Skipping logging.")
                    return_dict['result'] = float('-inf')
                logging.info(f"node_p:{node_p}, perturb_word_p:{perturb_word_p}, test_acc_mean:{test_acc_mean}, test_acc_std:{test_acc_std}\n")
                # print(node_p, perturb_word_p, test_acc_mean, test_acc_std)
                perturb_acc.append([node_p, perturb_word_p, test_acc_mean, test_acc_std])
    except Exception as e:
        print(e)
        if 'model' in locals():
            del model
        raise RuntimeError("Error.")
        # return_dict['result'] = float('-inf')
    print("result:\n",perturb_acc)
def run_trial(trial, data_name):
    learning_rates = [2e-5] #, 1e-3, 2e-3, 3e-3, 5e-3 1e-5, , 3e-5, 5e-5
    trial_params = {
        'model_name': trial.suggest_categorical('model_name', ['distilbert']),  # , 'bert', 'roberta', 'deberta' 'distilbert'
        'batch_size': trial.suggest_categorical('batch_size', [512]),#64, 128, 256, 512, 1024
        'max_length': trial.suggest_categorical('max_length', [256]),#512128,
        'num_epochs': trial.suggest_categorical('num_epochs', [100]),
        'learning_rate': trial.suggest_categorical('learning_rate', learning_rates),  # 5e-4
        'warmup_ratio': trial.suggest_categorical('warmup_ratio', [0.1]), #0.01, 0.02, 0.05,
        'use_pe': trial.suggest_categorical('use_pe', [True]),#, False
    }
    logging.info(f"{trial_params}")
    return_dict = {}
    return_dict['result'] = float('-inf')
    objective(trial_params, return_dict, data_name)

    return return_dict['result']

if __name__ == '__main__':
    for data_name in ['cora']:#,,, 'ogbn_products', 'pubmed','arxiv_2023'
        if '__file__' in globals():
            current_dir = os.path.dirname(__file__)
        else:
            current_dir = os.getcwd()
        db_path = os.path.join(current_dir, f"../sqlite/{data_name}_ft_xxbert.db")
        study = optuna.create_study(direction='maximize', study_name=f"{data_name}_ft_xxbert11",storage=f"sqlite:///{db_path}", load_if_exists=True)
        study.optimize(partial(run_trial, data_name=data_name), n_trials=200, n_jobs=1)
        if study.best_trial is not None:
            logging.info(f"Best hyperparameters:{study.best_params}")
            logging.info(f"Best test accuracy:{study.best_value}")
            logging.info(f'Best trial:')
            trial = study.best_trial
            logging.info(f'  Value: {trial.value}')
            logging.info('  Params: ')
            for key, value in trial.params.items():
                logging.info(f'{key}: {value}')
        else:
            logging.info("No trials completed successfully.")
