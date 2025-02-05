import os
cuda_visible_devices = os.environ.get('CUDA_VISIBLE_DEVICES')
print(f'CUDA_VISIBLE_DEVICES: {cuda_visible_devices}')

import gc
import argparse
import torch
from accelerate import Accelerator, DeepSpeedPlugin
from accelerate import DistributedDataParallelKwargs
from torch import nn, optim
from torch.optim import lr_scheduler
from tqdm import tqdm
import pmdarima as pm
from statsforecast import StatsForecast
from statsforecast.models import AutoARIMA
from statsforecast.arima import arima_string

from models import Autoformer, DLinear, TimeMamba, TimeLLM

from data_provider.data_factory import data_provider
import time
import random
import numpy as np

import pandas as pd
from utils.metrics import metric

import wandb 
from torchsummary import summary

os.environ['CURL_CA_BUNDLE'] = ''
os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "max_split_size_mb:64"

from utils.tools import del_files, EarlyStopping, adjust_learning_rate, vali, load_content

parser = argparse.ArgumentParser(description='Time-LLM')

# basic config
parser.add_argument('--task_name', type=str, required=True, default='long_term_forecast',
                    help='task name, options:[long_term_forecast, short_term_forecast, imputation, classification, anomaly_detection]')
parser.add_argument('--is_training', type=int, required=True, default=1, help='status')
parser.add_argument('--model_id', type=str, required=True, default='test', help='model id')
parser.add_argument('--model_comment', type=str, required=True, default='none', help='prefix when saving test results')
parser.add_argument('--model', type=str, required=True, default='Autoformer',
                    help='model name, options: [Autoformer, DLinear]')
parser.add_argument('--seed', type=int, default=2021, help='random seed')

# data loader
parser.add_argument('--data', type=str, required=True, default='ETTm1', help='dataset type')
parser.add_argument('--root_path', type=str, default='./dataset', help='root path of the data file')
parser.add_argument('--data_path', type=str, default='ETTh1.csv', help='data file')
parser.add_argument('--features', type=str, default='M',
                    help='forecasting task, options:[M, S, MS]; '
                         'M:multivariate predict multivariate, S: univariate predict univariate, '
                         'MS:multivariate predict univariate')
parser.add_argument('--target', type=str, default='OT', help='target feature in S or MS task')
parser.add_argument('--loader', type=str, default='modal', help='dataset type')
parser.add_argument('--freq', type=str, default='h',
                    help='freq for time features encoding, '
                         'options:[s:secondly, t:minutely, h:hourly, d:daily, b:business days, w:weekly, m:monthly], '
                         'you can also use more detailed freq like 15min or 3h')
parser.add_argument('--checkpoints', type=str, default='./checkpoints/', help='location of model checkpoints')

# forecasting task
parser.add_argument('--seq_len', type=int, default=96, help='input sequence length')
parser.add_argument('--label_len', type=int, default=48, help='start token length')
parser.add_argument('--pred_len', type=int, default=96, help='prediction sequence length')
parser.add_argument('--seasonal_patterns', type=str, default='Monthly', help='subset for M4')
parser.add_argument('--dsampfactor', type=int, default=1, help='for downsampling purposes')

# model define
parser.add_argument('--enc_in', type=int, default=7, help='encoder input size')
parser.add_argument('--dec_in', type=int, default=7, help='decoder input size')
parser.add_argument('--c_out', type=int, default=7, help='output size')
parser.add_argument('--d_model', type=int, default=16, help='dimension of model')
parser.add_argument('--n_heads', type=int, default=8, help='num of heads')
parser.add_argument('--e_layers', type=int, default=2, help='num of encoder layers')
parser.add_argument('--d_layers', type=int, default=1, help='num of decoder layers')
parser.add_argument('--d_ff', type=int, default=32, help='dimension of fcn')
parser.add_argument('--moving_avg', type=int, default=25, help='window size of moving average')
parser.add_argument('--factor', type=int, default=1, help='attn factor')
parser.add_argument('--dropout', type=float, default=0.1, help='dropout')
parser.add_argument('--embed', type=str, default='timeF',
                    help='time features encoding, options:[timeF, fixed, learned]')
parser.add_argument('--activation', type=str, default='gelu', help='activation')
parser.add_argument('--output_attention', action='store_true', help='whether to output attention in encoder')
parser.add_argument('--patch_len', type=int, default=16, help='patch length')
parser.add_argument('--stride', type=int, default=8, help='stride')
parser.add_argument('--prompt_domain', type=int, default=0, help='')
parser.add_argument('--llm_model', type=str, default='Mamba', help='LLM model') # LLAMA, GPT2, BERT, Mamba
parser.add_argument('--llm_dim', type=int, default='768', help='LLM model dimension')#Mamba:768 LLama7b:4096; GPT2-small:768; BERT-base:768
parser.add_argument('--num_params', type=str, default='130m', help='string of our param size to append to huggingface')
parser.add_argument('--rand_init', type=int, default=0, help='if nonzero, initialize weights of LLM randomly')

# optimization
parser.add_argument('--num_workers', type=int, default=10, help='data loader num workers')
parser.add_argument('--itr', type=int, default=1, help='experiments times')
parser.add_argument('--train_epochs', type=int, default=10, help='train epochs')
parser.add_argument('--align_epochs', type=int, default=10, help='alignment epochs')
parser.add_argument('--batch_size', type=int, default=32, help='batch size of train input data')
parser.add_argument('--eval_batch_size', type=int, default=8, help='batch size of model evaluation')
parser.add_argument('--patience', type=int, default=3, help='early stopping patience')
parser.add_argument('--learning_rate', type=float, default=0.0001, help='optimizer learning rate')
parser.add_argument('--des', type=str, default='test', help='exp description')
parser.add_argument('--loss', type=str, default='MSE', help='loss function')
parser.add_argument('--lradj', type=str, default='type1', help='adjust learning rate')
parser.add_argument('--pct_start', type=float, default=0.2, help='pct_start')
parser.add_argument('--use_amp', action='store_true', help='use automatic mixed precision training', default=False)
parser.add_argument('--llm_layers', type=int, default=6)
parser.add_argument('--percent', type=int, default=100)
parser.add_argument('--col_percent', type=int, default=100)


parser.add_argument('--use_wandb', type=int, default=1)
parser.add_argument('--verbose', type=int, default=0)
#parser.add_argument('--saveName',type=str,default="NULL",help='for smooth pipelining')
parser.add_argument('--early_break', type=int, default=0)
parser.add_argument('--save_checkpoints', type=int, default=0)


args = parser.parse_args()
ddp_kwargs = DistributedDataParallelKwargs(find_unused_parameters=True)
deepspeed_plugin = DeepSpeedPlugin(hf_ds_config='./ds_config_zero2.json')
if args.use_wandb:
    wandb.init(project = 'TimeMamba')
    #log the hyperparameters
    wandb.config.update({
        'layer count': args.llm_layers,
        'd_model': args.d_model,
        'train epochs': args.train_epochs,
        'model id': args.model_id,
        'model' : args.model,
        'LLM used': args.llm_model,
        'dsampfactor': args.dsampfactor,
        'percent': args.percent,
        'col_percent': args.col_percent,
        'rand_init': args.rand_init
    })

def print_gpu_memory_usage():
    allocated = torch.cuda.memory_allocated() / (1024 ** 2)  # Convert bytes to MB
    cached = torch.cuda.memory_reserved() / (1024 ** 2)  # Convert bytes to MB
    print(f"Allocated memory: {allocated:.2f} MB")
    print(f"Cached memory: {cached:.2f} MB")

accelerator = Accelerator(kwargs_handlers=[ddp_kwargs], deepspeed_plugin=deepspeed_plugin)
print("accelerator device: ", accelerator.device)
fix_seed = args.seed
random.seed(fix_seed)
torch.manual_seed(fix_seed)
np.random.seed(fix_seed)
# setting record of experiments
setting = '{}_{}_{}_{}_ft{}_sl{}_ll{}_pl{}_dm{}_nh{}_el{}_dl{}_df{}_fc{}_eb{}_{}_{}'.format(
    args.task_name,
    args.model_id,
    args.model,
    args.data,
    args.features,
    args.seq_len,
    args.label_len,
    args.pred_len,
    args.d_model,
    args.n_heads,
    args.e_layers,
    args.d_layers,
    args.d_ff,
    args.factor,
    args.embed,
    args.des, args.seed)

train_data, train_loader = data_provider(args, 'train')
vali_data, vali_loader = data_provider(args, 'val')
test_data, test_loader = data_provider(args, 'test')


print("Using Framework: ", args.model)
if args.model == 'TimeLLM':
    model = TimeLLM.Model(args).float()
elif args.model == 'Autoformer':
    model = Autoformer.Model(args).float()
elif args.model == 'DLinear':
    model = DLinear.Model(args).float()
'''
elif args.model == 'ARIMA':
    #print(train_data.data_x[:, 7])
    season_length = 12 # Monthly data 
    Y_test = (test_data.data_x)
    print("taking in series of length: ", args.seq_len)
    print("predicting for the next steps of horizon length: ", args.pred_len)
    input_seq = Y_test[0:args.seq_len, -1]
    actual = Y_test[args.seq_len+1:args.seq_len+1+args.pred_len, -1]
    
    # Create a DataFrame for statsforecast
    df = pd.DataFrame({
        'unique_id': 1,  # Single series identifier
        'ds': pd.date_range(start='2022-01-01', periods=len(input_seq), freq='MS'), #the start date and freq don't affect anything
        'y': input_seq
    })
    horizon = len(actual) # number of predictions
    models = [AutoARIMA(season_length=season_length)]
    sf = StatsForecast(models=models, freq='MS')
    Y_hat_df = sf.forecast(df=df, h=horizon, fitted=True)
    print("y_hat head:", Y_hat_df.head())
    forecast = Y_hat_df.to_numpy()[:,2]
    print("forecast:", forecast)
    actual = test_data.data_x[args.seq_len+1:args.seq_len+1+args.pred_len, 6]
    print("actual:", actual)
    metrics = metric(forecast, actual)
    print("metrics: ", metrics)
    if args.use_wandb:
        wandb.log({f"mae {args.seed}":metrics[0],f"mse {args.seed}":metrics[1], f"rmse {args.seed}":metrics[2], f"mape {args.seed}":metrics[3], f"mspe {args.seed}":metrics[4]})
    exit()
'''

print_gpu_memory_usage()

path = args.model_comment
args.content = load_content(args)
if args.verbose:
    print("os.path.exists: ", path)
    print("accel local main process: ", accelerator.is_local_main_process)
if not os.path.exists(path) and accelerator.is_local_main_process:
    os.makedirs(path)

train_steps = len(train_loader)
#train_steps = 120

early_stopping = EarlyStopping(accelerator=accelerator, patience=args.patience)

trained_parameters = []

for p in model.parameters():
    if p.requires_grad is True:
        trained_parameters.append(p)

model_optim = optim.Adam(trained_parameters, lr=args.learning_rate)

earlyUnwrap = accelerator.unwrap_model(model)
num_params=sum(p.numel() for p in earlyUnwrap.parameters())
print(f'Total number of parameters: {num_params}')
if args.use_wandb:
    wandb.config.update({'num_params':num_params})

if args.lradj == 'COS':
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(model_optim, T_max=20, eta_min=1e-8)
else:
    scheduler = lr_scheduler.OneCycleLR(optimizer=model_optim,
                                        steps_per_epoch=train_steps,
                                        pct_start=args.pct_start,
                                        epochs=args.train_epochs,
                                        max_lr=args.learning_rate)

criterion = nn.MSELoss()
mae_metric = nn.L1Loss()


train_loader, vali_loader, test_loader, model, model_optim, scheduler = accelerator.prepare(
    train_loader, vali_loader, test_loader, model, model_optim, scheduler)

if args.use_amp:
    scaler = torch.cuda.amp.GradScaler()

start_event = torch.cuda.Event(enable_timing=True)
end_event = torch.cuda.Event(enable_timing=True)
start_event.record()
for epoch in range(args.train_epochs):
    train_loss = [] 
    model.train()
    
    for i, (batch_x, batch_y, batch_x_mark, batch_y_mark) in tqdm(enumerate(train_loader)):
        #for testing purposes
        if args.early_break!=0 and iter_count > args.early_break:
            break
        iter_count += 1
        model_optim.zero_grad()

        batch_x = batch_x.float().to(accelerator.device)
        batch_y = batch_y.float().to(accelerator.device)
        batch_x_mark = batch_x_mark.float().to(accelerator.device)
        batch_y_mark = batch_y_mark.float().to(accelerator.device)
        
        # decoder input
        dec_inp = torch.zeros_like(batch_y[:, -args.pred_len:, :]).float().to(
            accelerator.device)
        dec_inp = torch.cat([batch_y[:, :args.label_len, :], dec_inp], dim=1).float().to(
            accelerator.device)


        # encoder - decoder
        if args.use_amp:
            #print("using amp")
            with torch.cuda.amp.autocast():
                if args.output_attention:
                    outputs = model(batch_x, batch_x_mark, dec_inp, batch_y_mark)[0]
                else:
                    outputs = model(batch_x, batch_x_mark, dec_inp, batch_y_mark)

                f_dim = -1 if args.features == 'MS' else 0
                outputs = outputs[:, -args.pred_len:, f_dim:]
                batch_y = batch_y[:, -args.pred_len:, f_dim:].to(accelerator.device)
                loss = criterion(outputs, batch_y)
                train_loss.append(loss.item())
        else:
            #print("no amp")
            if args.output_attention:
                outputs = model(batch_x, batch_x_mark, dec_inp, batch_y_mark)[0]
                #print("no amp output attention: ", outputs)
            else:
                outputs = model(batch_x, batch_x_mark, dec_inp, batch_y_mark)
                #print("no amp no output attention: ", outputs)

            f_dim = -1 if args.features == 'MS' else 0
            outputs = outputs[:, -args.pred_len:, f_dim:]
            batch_y = batch_y[:, -args.pred_len:, f_dim:]
            if args.model == "DLinear":
                batch_y = batch_y.to(torch.bfloat16)
            loss = criterion(outputs, batch_y)
            if args.model == "DLinear":
                loss = loss.to(torch.bfloat16)
            train_loss.append(loss.item())
          
        if args.verbose and ((i + 1) % 10000 == 0): #this doesn't happen with 10,000
            #accelerator.print("\ttime taken for ",n," iters: ",iterStartTime)
            accelerator.print(
                "\titers: {0}, epoch: {1} | loss: {2:.7f}".format(i + 1, epoch + 1, loss.item()))
        
        #break 
        
        if args.use_amp:
            scaler.scale(loss).backward()
            scaler.step(model_optim)
            scaler.update()
        else:
            accelerator.backward(loss) #here's backprop
            model_optim.step() #and this is adam taking a step

        if args.lradj == 'TST':
            adjust_learning_rate(accelerator, model_optim, scheduler, epoch + 1, args, printout=False)
            scheduler.step()
    #accelerator.print("Epoch: {} cost time: {}".format(epoch + 1, time.time() - epoch_time))
    if args.verbose:
        train_loss = np.average(train_loss)
    vali_loss, vali_mae_loss = vali(args, accelerator, model, vali_data, vali_loader, criterion, mae_metric)
    test_loss, test_mae_loss = vali(args, accelerator, model, test_data, test_loader, criterion, mae_metric)
    if args.verbose:
        accelerator.print(
            "Epoch: {0} | Train Loss: {1:.7f} Vali Loss: {2:.7f} Test Loss: {3:.7f} MAE Loss: {4:.7f}".format(
                epoch + 1, train_loss, vali_loss, test_loss, test_mae_loss))
        if args.use_wandb:
            wandb.log({f"train loss":train_loss, f"vali loss": vali_loss, f"MSE loss": test_loss, f"MAE loss": test_mae_loss})

    if not os.path.exists(path):
        os.makedirs(path)
    early_stopping(vali_loss, model, path)
    if early_stopping.early_stop:
        #accelerator.print("Early stopping")
        if args.use_wandb:
            wandb.log({f"actual epochs": epoch+1})
            wandb.log({f"MSE loss": test_loss, f"MAE loss": test_mae_loss})
        break

    if args.lradj != 'TST':
        if args.lradj == 'COS':
            scheduler.step()
            if args.verbose:
                accelerator.print("lr = {:.10f}".format(model_optim.param_groups[0]['lr']))
        else:
            if epoch == 0:
                args.learning_rate = model_optim.param_groups[0]['lr']
                if args.verbose:
                    accelerator.print("lr = {:.10f}".format(model_optim.param_groups[0]['lr']))
            adjust_learning_rate(accelerator, model_optim, scheduler, epoch + 1, args, printout=True)

    else:
        if args.verbose:
            accelerator.print('Updating learning rate to {}'.format(scheduler.get_last_lr()[0]))

accelerator.wait_for_everyone()
end_event.record()
torch.cuda.synchronize()
elapsed_time = start_event.elapsed_time(end_event)
print("gpu time:", elapsed_time)

best_model_path = path + '/' + 'checkpoint'
accelerator.wait_for_everyone()
unwrapped_model = accelerator.unwrap_model(model)
unwrapped_model.load_state_dict(torch.load(best_model_path, map_location=lambda storage, loc: storage))
num_params = sum(p.numel() for p in unwrapped_model.parameters())
print(f'Total number of parameters: {num_params}')
if args.use_wandb:
    wandb.config.update({'num_params':num_params})
    wandb.config.update({'gpu time':elapsed_time})

accelerator.wait_for_everyone()
if accelerator.is_local_main_process:
    path = './checkpoints'  # unique checkpoint saving path

    if args.save_checkpoints == 0:
        #del_files(path)  # delete checkpoint files
        os.remove(best_model_path)
        accelerator.print('success delete checkpoints at path : ', path)
        #print('success delete checkpoints at path : ', path)
accelerator.print('done!')
#print('done!')
