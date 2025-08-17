import os
cuda_visible_devices = os.environ.get('CUDA_VISIBLE_DEVICES')
print(f'CUDA_VISIBLE_DEVICES: {cuda_visible_devices}')
import git, gc, argparse, torch, time, random
from accelerate import Accelerator, DeepSpeedPlugin
from accelerate import DistributedDataParallelKwargs
from torch import nn, optim
from torch.optim import lr_scheduler
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import wandb
import torch.nn.functional as F

from tqdm import tqdm

from models import Autoformer, DLinear, TimeMamba, TimeLLM
from data_provider.data_factory import data_provider
from utils.metrics import metric

import sys
sys.path.insert(0, "/home/nesl/oliver/timeSeriesMamba/Mamba4Cast/src_torch")
from training.models import SSMModel, SSMModelMulti

os.environ['CURL_CA_BUNDLE'] = ''
os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "max_split_size_mb:64"

# --------------------------------------------------------------------------------
# KEEP YOUR EXISTING METRIC/HELPER BLOCK HERE UNCHANGED
# - spectral_predictability / largest_lyapunov_rosenstein / batch_forecastability
# - spectral_entropy_1d / seasonality_strength_1d / batch_spectral_metrics
# - quick_stats / MASE / MASE_batched / sMAPE_batched / WAPE
# - vali(...) and visualize_example(...) exactly as in your file
# --------------------------------------------------------------------------------

# -------- Small baseline trainer for light models (DLinear/Ridge) --------
def _forward_model(args, model, batch_x, batch_x_mark, batch_y, batch_y_mark):
    """Unified forward compatible with your existing evaluate path."""
    dec_inp = torch.zeros_like(batch_y[:, -args.pred_len:, :], device=batch_y.device)
    dec_inp = torch.cat([batch_y[:, :args.label_len, :], dec_inp], dim=1)
    if args.output_attention:
        outputs = model(batch_x, batch_x_mark, dec_inp, batch_y_mark)[0]
    else:
        outputs = model(batch_x, batch_x_mark, dec_inp, batch_y_mark)
    f_dim = -1 if args.features == 'MS' else 0
    outputs = outputs[:, -args.pred_len:, f_dim:]
    gt = batch_y[:, -args.pred_len:, f_dim:]
    return outputs, gt

def train_quick_baseline(args, accelerator, model, train_loader, val_loader):
    """Fast supervised fit for DLinear in bf16."""
    criterion = nn.MSELoss()
    opt = optim.Adam(model.parameters(),
                     lr=args.baseline_lr,
                     weight_decay=getattr(args, "baseline_weight_decay", 0.0))

    # IMPORTANT: let Accelerate/DS control dtype casting
    train_loader, val_loader, model, opt = accelerator.prepare(
        train_loader, val_loader, model, opt
    )

    use_bf16 = (getattr(accelerator, "mixed_precision", None) == "bf16")
    best_val = float('inf'); best_state = None
    patience_left = int(getattr(args, "baseline_patience", 2))
    epochs = int(getattr(args, "baseline_epochs", 5))

    for epoch in range(epochs):
        model.train()
        epoch_loss = 0.0

        for batch_x, batch_y, batch_x_mark, batch_y_mark in train_loader:
            # move to device; DO NOT force dtype here
            batch_x      = batch_x.to(accelerator.device, non_blocking=True)
            batch_y      = batch_y.to(accelerator.device, non_blocking=True)
            batch_x_mark = batch_x_mark.to(accelerator.device, non_blocking=True)
            batch_y_mark = batch_y_mark.to(accelerator.device, non_blocking=True)

            opt.zero_grad(set_to_none=True)

            # bf16 autocast for forward+loss
            with accelerator.autocast():
                outputs, gt = _forward_model(args, model, batch_x, batch_x_mark, batch_y, batch_y_mark)
                loss = criterion(outputs, gt)

            # make loss bf16 for DS when requested
            if use_bf16:
                loss = loss.to(torch.bfloat16)

            accelerator.backward(loss)
            opt.step()

            epoch_loss += float(loss.detach().to(torch.float32).item())

        # ---- validation (no backward) ----
        model.eval()
        val_losses = []
        with torch.no_grad():
            for batch_x, batch_y, batch_x_mark, batch_y_mark in val_loader:
                batch_x      = batch_x.to(accelerator.device, non_blocking=True)
                batch_y      = batch_y.to(accelerator.device, non_blocking=True)
                batch_x_mark = batch_x_mark.to(accelerator.device, non_blocking=True)
                batch_y_mark = batch_y_mark.to(accelerator.device, non_blocking=True)

                with accelerator.autocast():
                    outputs, gt = _forward_model(args, model, batch_x, batch_x_mark, batch_y, batch_y_mark)
                    vloss = criterion(outputs, gt)

                val_losses.append(float(vloss.detach().to(torch.float32).item()))

        val_mse = float(np.mean(val_losses)) if val_losses else float('inf')

        if accelerator.is_local_main_process and getattr(args, "use_wandb", 0):
            wandb.log({
                "baseline_train_mse": epoch_loss / max(1, len(train_loader)),
                "baseline_val_mse": val_mse,
                "baseline_epoch": epoch,
            })

        # Early stopping
        if val_mse < best_val - 1e-8:
            best_val = val_mse
            best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
            patience_left = int(getattr(args, "baseline_patience", 2))
        else:
            patience_left -= 1
            if patience_left <= 0:
                break

    if best_state is not None:
        model.load_state_dict(best_state)

    return model


class TorchRidge(nn.Module):
    def __init__(self, in_dim, out_dim, alpha=1.0):
        super().__init__()
        self.linear = nn.Linear(in_dim, out_dim)
        self.alpha = alpha
    def forward(self, x):
        return self.linear(x)
    def ridge_loss(self, pred, target):
        mse = F.mse_loss(pred, target)
        l2 = self.alpha * torch.sum(self.linear.weight ** 2)
        return mse + l2

class TorchRidge(nn.Module):
    def __init__(self, in_dim, out_dim, alpha=1.0):
        super().__init__()
        self.linear = nn.Linear(in_dim, out_dim)
        self.alpha = alpha
    def forward(self, x):
        return self.linear(x)
    def ridge_loss(self, pred, target):
        mse = F.mse_loss(pred, target)
        l2 = self.alpha * torch.sum(self.linear.weight ** 2)
        return mse + l2

def fit_ridge_on_windows(args, accelerator, train_loader, val_loader):
    in_dim  = args.seq_len * args.enc_in
    out_dim = args.pred_len * args.dec_in
    model = TorchRidge(in_dim, out_dim, alpha=1.0)

    opt = torch.optim.Adam(model.parameters(),
                           lr=args.baseline_lr,
                           weight_decay=getattr(args, "baseline_weight_decay", 0.0))
    # Prepare after creating model/opt
    train_loader, val_loader, model, opt = accelerator.prepare(train_loader, val_loader, model, opt)

    use_bf16 = (getattr(accelerator, "mixed_precision", None) == "bf16")
    best_val = float('inf'); best_state = None
    patience_left = int(getattr(args, "baseline_patience", 2))
    epochs = int(getattr(args, "baseline_epochs", 5))

    for epoch in range(epochs):
        model.train()
        tr_losses = []
        for batch_x, batch_y, _, _ in train_loader:
            X = batch_x.view(batch_x.size(0), -1).to(accelerator.device, non_blocking=True)
            Y = batch_y[:, -args.pred_len:, :].reshape(batch_y.size(0), -1).to(accelerator.device, non_blocking=True)

            opt.zero_grad(set_to_none=True)
            with accelerator.autocast():
                pred = model(X)
                loss = model.ridge_loss(pred, Y)

            if use_bf16:
                loss = loss.to(torch.bfloat16)

            accelerator.backward(loss)
            opt.step()
            tr_losses.append(float(loss.detach().to(torch.float32).item()))

        # val
        model.eval()
        val_losses = []
        with torch.no_grad():
            for batch_x, batch_y, _, _ in val_loader:
                X = batch_x.view(batch_x.size(0), -1).to(accelerator.device, non_blocking=True)
                Y = batch_y[:, -args.pred_len:, :].reshape(batch_y.size(0), -1).to(accelerator.device, non_blocking=True)
                with accelerator.autocast():
                    pred = model(X)
                    vloss = F.mse_loss(pred, Y)
                val_losses.append(float(vloss.detach().to(torch.float32).item()))

        val_mse = float(np.mean(val_losses)) if val_losses else float('inf')
        if accelerator.is_local_main_process and getattr(args, "use_wandb", 0):
            wandb.log({"ridge_train_loss": np.mean(tr_losses) if tr_losses else np.nan,
                       "ridge_val_mse": val_mse,
                       "ridge_epoch": epoch})

        if val_mse < best_val - 1e-8:
            best_val = val_mse
            best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
            patience_left = int(getattr(args, "baseline_patience", 2))
        else:
            patience_left -= 1
            if patience_left <= 0:
                break

    if best_state is not None:
        model.load_state_dict(best_state)
    return model

# ------------------------------ main ------------------------------
if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Time-LLM')

    # basic
    parser.add_argument('--task_name', type=str, required=True, default='long_term_forecast')
    parser.add_argument('--model_id', type=str, required=True, default='test')
    parser.add_argument('--model', type=str, required=True, default='Autoformer',
                        help='[Autoformer, DLinear, TimeLLM, Ridge]')
    parser.add_argument('--seed', type=int, default=2021)

    # data / paths
    parser.add_argument('--checkpoint_path', type=str, default=None, help='ignored for DLinear/Ridge')
    parser.add_argument('--data', type=str, required=True, default='ETTm1')
    parser.add_argument('--root_path', type=str, default='./dataset')
    parser.add_argument('--data_path', type=str, default='ETTh1.csv')
    parser.add_argument('--data_path_test', type=str, default='None')
    parser.add_argument('--data_path_val', type=str, default='None')

    parser.add_argument('--features', type=str, default='M')
    parser.add_argument('--target', type=str, default='OT')
    parser.add_argument('--loader', type=str, default='modal')
    parser.add_argument('--freq', type=str, default='h')
    parser.add_argument('--checkpoints', type=str, default='./checkpoints/')
    parser.add_argument('--pretrain', type=int, default=0)

    # forecasting task
    parser.add_argument('--seq_len', type=int, default=96)
    parser.add_argument('--label_len', type=int, default=48)
    parser.add_argument('--pred_len', type=int, default=96)
    parser.add_argument('--seasonal_patterns', type=str, default='Monthly')
    parser.add_argument('--dsampfactor', type=int, default=1)

    # model define
    parser.add_argument('--enc_in', type=int, default=7)
    parser.add_argument('--dec_in', type=int, default=7)
    parser.add_argument('--c_out', type=int, default=7)
    parser.add_argument('--d_model', type=int, default=16)
    parser.add_argument('--n_heads', type=int, default=8)
    parser.add_argument('--e_layers', type=int, default=2)
    parser.add_argument('--d_layers', type=int, default=1)
    parser.add_argument('--d_ff', type=int, default=32)
    parser.add_argument('--moving_avg', type=int, default=25)
    parser.add_argument('--factor', type=int, default=1)
    parser.add_argument('--dropout', type=float, default=0.1)
    parser.add_argument('--embed', type=str, default='timeF')
    parser.add_argument('--activation', type=str, default='gelu')
    parser.add_argument('--output_attention', action='store_true')
    parser.add_argument('--patch_len', type=int, default=16)
    parser.add_argument('--stride', type=int, default=8)
    parser.add_argument('--prompt_domain', type=int, default=0)
    parser.add_argument('--llm_model', type=str, default='Mamba')
    parser.add_argument('--llm_dim', type=int, default=768)
    parser.add_argument('--num_params', type=str, default='130m')
    parser.add_argument('--rand_init', type=int, default=0)
    parser.add_argument('--init_seed', type=int, default=0)
    parser.add_argument('--finetune_llm', type=int, default=0)
    parser.add_argument('--boundary_file', type=str, default=None)

    # optimization / runtime
    parser.add_argument('--num_workers', type=int, default=10)
    parser.add_argument('--itr', type=int, default=1)
    parser.add_argument('--train_epochs', type=int, default=10)
    parser.add_argument('--align_epochs', type=int, default=10)
    parser.add_argument('--batch_size', type=int, default=32)
    parser.add_argument('--eval_batch_size', type=int, default=8)
    parser.add_argument('--patience', type=int, default=3)
    parser.add_argument('--learning_rate', type=float, default=1e-4)
    parser.add_argument('--des', type=str, default='test')
    parser.add_argument('--loss', type=str, default='MSE')
    parser.add_argument('--lradj', type=str, default='type1')
    parser.add_argument('--pct_start', type=float, default=0.2)
    parser.add_argument('--use_amp', action='store_true', default=False)
    parser.add_argument('--llm_layers', type=int, default=6)
    parser.add_argument('--percent', type=int, default=100)
    parser.add_argument('--col_percent', type=int, default=100)
    parser.add_argument('--train_percent', type=int, default=100)
    parser.add_argument('--split_type', type=str, default="temporal")
    parser.add_argument('--source', type=str, default="None")
    parser.add_argument('--heldout', type=str, default="None")
    parser.add_argument('--visualize', action='store_true')
    parser.add_argument('--use_wandb', type=int, default=1)
    parser.add_argument('--verbose', type=int, default=1)
    parser.add_argument('--early_break', type=int, default=0)
    parser.add_argument('--save_checkpoints', type=int, default=0)
    parser.add_argument('--use_classical_model', action='store_true')

    # NEW: fast-baseline knobs
    parser.add_argument('--train_baseline', type=int, default=1, help='if 1, train DLinear/Ridge quickly on train split')
    parser.add_argument('--baseline_epochs', type=int, default=5)
    parser.add_argument('--baseline_lr', type=float, default=5e-4)
    parser.add_argument('--baseline_weight_decay', type=float, default=0.0)
    parser.add_argument('--baseline_patience', type=int, default=2)

    args = parser.parse_args()

    # Accelerator
    ddp_kwargs = DistributedDataParallelKwargs(find_unused_parameters=True)
    if args.llm_model == "Moirai":
        deepspeed_plugin = DeepSpeedPlugin(hf_ds_config='./float_ds_config.json')
    else:
        deepspeed_plugin = DeepSpeedPlugin(hf_ds_config='./ds_config_zero2.json')
    accelerator = Accelerator(kwargs_handlers=[ddp_kwargs], deepspeed_plugin=deepspeed_plugin)
    print("accelerator device: ", accelerator.device)

    # seeds
    random.seed(args.seed); torch.manual_seed(args.seed); np.random.seed(args.seed)

    # wandb
    if args.use_wandb:
        try:
            repo = git.Repo(search_parent_directories=True)
            commit_hash = repo.head.object.hexsha
            wandb.init(project='TimeMamba')
            wandb.config.update({k: getattr(args, k) for k in vars(args)})
            wandb.config.update({'git_commit': commit_hash})
        except Exception as e:
            print(f"Failed to initialize wandb: {e}")
            args.use_wandb = 0

    # -------- data --------
    train_data, train_loader = data_provider(args, 'train')
    val_data,   val_loader   = data_provider(args, 'val')
    test_data,  test_loader  = data_provider(args, 'test')

    # -------- model init --------
    if args.model == 'TimeLLM':
        model = TimeLLM.Model(args).float()
    elif args.model == 'Autoformer':
        model = Autoformer.Model(args).float()
    elif args.model == 'DLinear':
        model = DLinear.Model(args).float()
    elif args.model == 'Ridge':
        # placeholder; we will build TorchRidge separately
        model = None
    else:
        raise ValueError(f"Unknown model: {args.model}")

    # -------- training/eval logic --------
    criterion = nn.MSELoss()
    mae_metric = nn.L1Loss()

    if args.model == 'Ridge':
        # Train small Ridge on train, early stop on val
        ridge_model = fit_ridge_on_windows(args, accelerator, train_loader, val_loader)
        # Evaluate on test
        ridge_model.eval()
        with torch.no_grad():
            # rebuild test X,Y as in fit
            X_list, Y_list = [], []
            for batch_x, batch_y, _, _ in test_loader:
                X_list.append(batch_x.view(batch_x.size(0), -1))
                Y_list.append(batch_y[:, -args.pred_len:, :].reshape(batch_y.size(0), -1))
            X = torch.cat(X_list, dim=0).to(accelerator.device)
            Y = torch.cat(Y_list, dim=0).to(accelerator.device)
            pred = ridge_model(X)
            mse_loss = F.mse_loss(pred, Y).item()
            mae_loss = F.l1_loss(pred, Y).item()
        print(f"[Ridge] Test MSE: {mse_loss} | MAE: {mae_loss}")
        if args.use_wandb:
            wandb.log({"MSE loss": mse_loss, "MAE loss": mae_loss})
            wandb.finish()
        # Optional: simple CSV viz (same idea as your visualize_example)
        if args.visualize and X.size(0) > 0:
            b0 = 0
            y_pred_seq = pred[b0].reshape(args.pred_len, args.dec_in).detach().cpu().numpy()
            y_true_seq = Y[b0].reshape(args.pred_len, args.dec_in).detach().cpu().numpy()
            # Build a minimal CSV with just horizon (no context here since Ridge input is flattened)
            df = pd.DataFrame({f'pred_{i}': y_pred_seq[:, i] for i in range(args.dec_in)})
            for i in range(args.dec_in):
                df[f'true_{i}'] = y_true_seq[:, i]
            os.makedirs('visuals', exist_ok=True)
            csv_path = f'visuals/visualize_{args.model_id}_ridge_seed{args.seed}.csv'
            df.to_csv(csv_path, index_label='t')
            print(f"[Ridge] Visualization saved to {csv_path}")
        exit(0)

    # DLinear quick baseline train (if enabled)
    if args.model == 'DLinear' and args.train_baseline:
        model = train_quick_baseline(args, accelerator, model, train_loader, val_loader)
        # After training, eval on test using your vali()
        # Need to prepare test loader + model again for eval consistency
        # (Accelerate allows reusing prepared modules; safest is to re-prepare fresh for test)
        test_loader_prep, model_prep = accelerator.prepare(test_loader, model)
        # Use your existing vali (imports from your file); it expects optimizer too,
        # but we can pass a dummy optimizer since vali() doesn’t use it.
        # To avoid touching vali(), wrap a no-op optimizer:
        dummy_optim = optim.SGD(model_prep.parameters(), lr=1.0)
        test_loss, test_mae_loss, test_mase_loss, test_smape_loss = vali(args, accelerator, model_prep, test_data, test_loader_prep, criterion, mae_metric)
        print(f"[DLinear] Test MSE: {test_loss} | MAE: {test_mae_loss} | MASE: {test_mase_loss} | sMAPE: {test_smape_loss}")
        if args.use_wandb:
            wandb.log({"MSE loss": test_loss, "MAE loss": test_mae_loss, "MASE loss": test_mase_loss, "sMAPE loss": test_smape_loss})
            wandb.finish()
        if args.visualize:
            visualize_example(args, accelerator, model_prep, test_loader_prep)
        exit(0)

    # For non-baseline (TimeLLM/Autoformer) or DLinear eval-only path:
    # Optional checkpoint loading (skipped for DLinear)
    if args.model != 'DLinear':
        if args.checkpoint_path and os.path.isfile(args.checkpoint_path):
            state = torch.load(args.checkpoint_path, map_location=lambda storage, loc: storage)
            model.load_state_dict(state)
        else:
            print("[INFO] No checkpoint supplied/found; proceeding without loading weights.")

    # Prepare for plain eval
    test_loader, model = accelerator.prepare(test_loader, model)

    # Inspect params
    earlyUnwrap = accelerator.unwrap_model(model)
    num_params = sum(p.numel() for p in earlyUnwrap.parameters())
    print(f'Total number of parameters: {num_params}')
    if args.use_wandb:
        wandb.config.update({'num_params': num_params})

    # Evaluate
    test_loss, test_mae_loss, test_mase_loss, test_smape_loss = vali(args, accelerator, model, test_data, test_loader, nn.MSELoss(), nn.L1Loss())
    print(f"MSE loss: {test_loss}")
    print(f"MAE loss: {test_mae_loss}")
    print(f"MASE loss: {test_mase_loss}")
    print(f"sMAPE loss: {test_smape_loss}")

    if args.visualize:
        visualize_example(args, accelerator, model, test_loader)
    if args.use_wandb:
        wandb.log({"MSE loss": test_loss, "MAE loss": test_mae_loss, "MASE loss": test_mase_loss, "sMAPE loss": test_smape_loss})
        wandb.finish()
