import os
cuda_visible_devices = os.environ.get('CUDA_VISIBLE_DEVICES')
print(f'CUDA_VISIBLE_DEVICES: {cuda_visible_devices}')
import git
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
from models import PatchTST  

from data_provider.data_factory import data_provider
import time
import random
import numpy as np

import pandas as pd
from utils.metrics import metric
import matplotlib.pyplot as plt
import wandb 
from torchsummary import summary

import sys
sys.path.insert(0, "/home/nesl/oliver/timeSeriesMamba/Mamba4Cast/src_torch")
from training.models import SSMModel, SSMModelMulti

os.environ['CURL_CA_BUNDLE'] = ''
os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "max_split_size_mb:64"

from utils.tools import del_files, EarlyStopping, adjust_learning_rate, vali, load_content
if __name__ == '__main__':
        
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
    parser.add_argument('--data_pretrain', type=str, default='None', help='dataset type')
    parser.add_argument('--root_path', type=str, default='./dataset', help='root path of the data file')
    parser.add_argument('--data_path', type=str, default='ETTh1.csv', help='data file')
    parser.add_argument('--data_path_test', type=str, default='None', help='data file, make sure is set when cov split')
    parser.add_argument('--data_path_val', type=str, default='None', help='data file for covariate split')
   
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
    parser.add_argument('--pretrain', type=int, default=0)

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
    parser.add_argument('--init_seed', type=int, default=0, help='seed for rand_init only')
    parser.add_argument('--finetune_llm', type=int, default=0, help='if nonzero, allow LLM weights to be trained')
    parser.add_argument('--boundary_file', type=str, default=None, help='if not None, prevents training windows to be takena cross a concatenated datafile')

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
    parser.add_argument('--train_percent', type=int, default=100)
    parser.add_argument('--split_type', type=str, default="temporal")

    parser.add_argument('--visualize', action='store_true', help='visualize a test example after training')
    parser.add_argument('--use_wandb', type=int, default=1)
    parser.add_argument('--verbose', type=int, default=1)
    #parser.add_argument('--saveName',type=str,default="NULL",help='for smooth pipelining')
    parser.add_argument('--early_break', type=int, default=0)
    parser.add_argument('--save_checkpoints', type=int, default=0)
    parser.add_argument('--univar', type=int, default=0)

    parser.add_argument('--source', type=str, default="None")


    args = parser.parse_args()
    ddp_kwargs = DistributedDataParallelKwargs(find_unused_parameters=True)
    
    
    if args.llm_model == "Moirai":
        deepspeed_plugin = DeepSpeedPlugin(hf_ds_config='./float_ds_config.json')
    else:
        deepspeed_plugin = DeepSpeedPlugin(hf_ds_config='./ds_config_zero2.json')
    
    #deepspeed_plugin = DeepSpeedPlugin(hf_ds_config='./ds_config_zero2.json')
    if args.use_wandb:

        wandb.init(project = 'TimeMamba')

        repo = git.Repo(search_parent_directories=True)
        commit_hash = repo.head.object.hexsha

        #log the hyperparameters
        wandb.config.update({
            'git_commit': commit_hash,
            'layer count': args.llm_layers,
            'd_model': args.d_model,
            'train epochs': args.train_epochs,
            'model id': args.model_id,
            'model' : args.model,
            'LLM used': args.llm_model+"_LLM",
            'dsampfactor': args.dsampfactor,
            'percent': args.percent,
            'col_percent': args.col_percent,
            'train_percent': args.train_percent,
            'rand_init': args.rand_init,
            'seed': args.seed,
            'init_seed': args.init_seed,
            'pred_len': args.pred_len,
            'seq_len': args.seq_len, 
            'pretrain': args.pretrain,
            'finetune_llm': args.finetune_llm,
            'split_type': args.split_type,
            'source': args.source,
            'univar': args.univar
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
        args.des, 
        args.seed,
        args.pretrain,
        args.finetune_llm)


    vali_data, vali_loader = data_provider(args, 'val')
    test_data, test_loader = data_provider(args, 'test')
    args.percent = int(args.percent*args.train_percent/(100))
    train_data, train_loader = data_provider(args, 'train')

    
    criterion = nn.MSELoss()
    mae_metric = nn.L1Loss()

    def adapt_state_dict_keys(old_state_dict):
        new_state_dict = {}

        for key in old_state_dict.keys():
            if "linear_layer" in key:
                # Extract the layer index
                layer_idx = key.split('.')[1]
                
                # Replace "linear_layer" with "stage_2_layer.0"
                new_key = key.replace(f"linear_layer", f"stage_2_layer.0")
                
                # Add the updated key to the new state dict
                new_state_dict[new_key] = old_state_dict[key]
            else:
                # Keep other keys unchanged
                new_state_dict[key] = old_state_dict[key]

        return new_state_dict

    def visualize_example(args, accelerator, model, test_loader):
        if not accelerator.is_local_main_process:
            return

        model.eval()
        with torch.no_grad():
            for batch_x, batch_y, batch_x_mark, batch_y_mark in test_loader:
                batch_x = batch_x.float().to(accelerator.device)
                batch_y = batch_y.float().to(accelerator.device)
                batch_x_mark = batch_x_mark.float().to(accelerator.device)
                batch_y_mark = batch_y_mark.float().to(accelerator.device)

                dec_inp = torch.zeros_like(batch_y[:, -args.pred_len:, :]).to(accelerator.device)
                dec_inp = torch.cat([batch_y[:, :args.label_len, :], dec_inp], dim=1)

                if args.output_attention:
                    outputs = model(batch_x, batch_x_mark, dec_inp, batch_y_mark)[0]
                else:
                    outputs = model(batch_x, batch_x_mark, dec_inp, batch_y_mark)

                seq_len, pred_len = args.seq_len, args.pred_len
                f = outputs.shape[-1]

                ctx = batch_x[0, :seq_len, :f].cpu().numpy()
                gt  = batch_y[0, -pred_len:, :f].cpu().numpy()
                pred= outputs[0, -pred_len:, :f].cpu().numpy()

                T = seq_len + pred_len
                actual = np.zeros((T, f))
                actual[:seq_len] = ctx
                actual[seq_len:] = gt

                predicted = np.full((T, f), np.nan)
                predicted[seq_len:] = pred

                if args.source == "None":
                    feature_names = ['coal', 'nat_gas', 'nuclear', 'oil', 'hydro', 'solar', 'wind', 'other']
                else:
                    feature_names = [f'{args.source}']
                data = {}
                print("actual in visualize: ", actual)
                for i, name in enumerate(feature_names):
                    data[f'{name}_actual'] = actual[:, i]
                    data[f'{name}_pred']   = predicted[:, i]

                df = pd.DataFrame(data, index=np.arange(T))
                csv_path = f'visuals/visualize_{args.model_id}_{args.model}_{args.source}Source_randinit{args.rand_init}_seed{args.seed}_initseed{args.init_seed}.csv'
                df.to_csv(csv_path, index_label='time_step')
                break

    print("Using Framework: ", args.model)
    if args.model == 'TimeLLM':
        model = TimeLLM.Model(args).float()
    elif args.model == 'Autoformer':
        model = Autoformer.Model(args).float()
    elif args.model == 'DLinear':
        model = DLinear.Model(args).float()
    elif args.model == 'PatchTST':  # NEW
        model = PatchTST.Model(args).float()
    elif args.model == "Mamba4Cast":
        #device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        ssm_config = {
            "bidirectional":False,
            "enc_conv" : True,
            "init_dil_conv" : True,
            "enc_conv_kernel" : 5,
            "init_conv_kernel" : 5,
            "init_conv_max_dilation" : 3,
            "global_residual":False,
            "in_proj_norm":False,
            "initial_gelu_flag":True,
            "linear_seq":15,
            "mamba2":True,
            "norm":True,
            "norm_type":"layernorm",
            "num_encoder_layers":2,
            "d_state":128,
            "residual":False,
            "token_embed_len":1024,
        } #probably don't mess with this because we have a state dict to load
        model = SSMModelMulti(scaler='min_max', sub_day=True, **ssm_config).to(accelerator.device)
        new_state_dict = adapt_state_dict_keys(torch.load('/home/nesl/oliver/timeSeriesMamba/Mamba4Cast/models/model.pth', map_location=accelerator.device)['model_state_dict'])
        model.load_state_dict(new_state_dict)
        model.eval()     
        
        test_loss, test_mae_loss = vali(args, accelerator, model, test_data, test_loader, criterion, mae_metric)
        print("mse loss: ", test_loss)
        print("mae loss: ", test_mae_loss)
        if args.use_wandb:
            wandb.log({f"MSE loss": test_loss, f"MAE loss": test_mae_loss})
        

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
        #early_stopping(test_loss, model, path)
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
    if args.visualize:
        visualize_example(args, accelerator, model, test_loader)

    num_params = sum(p.numel() for p in unwrapped_model.parameters())
    print(f'Total number of parameters: {num_params}')
    if args.use_wandb:
        wandb.config.update({'num_params':num_params})
        wandb.config.update({'gpu time':elapsed_time})

    accelerator.wait_for_everyone()
    if accelerator.is_local_main_process:
        path = './checkpoints'  # unique checkpoint saving path

        if args.save_checkpoints == 0:
            # Delete the checkpoint file
            os.remove(best_model_path)
            
            # Try to remove the folder that contained the checkpoint file
            try:
                os.rmdir(os.path.dirname(best_model_path))
                accelerator.print(f'Successfully deleted checkpoint file and empty folder: {os.path.dirname(best_model_path)}')
            except OSError:
                # Folder not empty or another error occurred
                accelerator.print(f'Successfully deleted checkpoint file: {best_model_path}')
        else:
            accelerator.print('Checkpoints are set to be saved; no deletion performed.')
    accelerator.print('done!')
