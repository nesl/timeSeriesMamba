
# Code to generate the plots and MAE statistics in Jupyter notebook
import torch
import matplotlib.pyplot as plt
import pandas as pd
from huggingface_hub import hf_hub_download
from gluonts.dataset.pandas import PandasDataset
from gluonts.dataset.split import split
from uni2ts.model.moirai import MoiraiForecast, MoiraiModule
import numpy as np
from tqdm import tqdm
from IPython.display import display, HTML

# Create synthetic dataset
syn_len = 252*10
train_len = 252*9
i1 = np.array([[1]*1 + [0]*1]*5000).flatten()[-syn_len:] - 0.5
syn_df = pd.DataFrame(
    np.c_[
        i1,
        i1
    ],
    columns=["i1", "y"]
)
syn_df.index = pd.date_range(end="2024-03-31", freq="D", periods=syn_len)
syn_df.index.name="date"

# Set MOIRAI parameters
SIZE = "small"  # model size: choose from {'small', 'base', 'large'}
PDT = 1  # prediction length: any positive integer
CTX = 252  # context length: any positive integer
PSZ = "auto"  # patch size: choose from {"auto", 8, 16, 32, 64, 128}
BSZ = 32  # batch size: any positive integer
TEST = 252  # test set length: any positive integer
window_distance = 1 # PDT for non-overlapping windows
n_windows = (TEST - PDT) // window_distance # TEST // PDT if window

# Run inference using two setups
#    - with 1 covariate
#    - without any covariates

target_cols = ["y"]

for CTX in [252]:
    display(HTML(f"<h3>CTX={CTX}</h3>"))
    for feature_cols in [["i1"], []]:
        # Convert into GluonTS dataset
        ds = PandasDataset(syn_df, target="y", feat_dynamic_real=feature_cols)

        # Split into train/test set
        train, test_template = split(
            ds,
            offset=-TEST
        )  # assign last TEST time steps as test set

        # Construct rolling window evaluation
        test_data = test_template.generate_instances(
            prediction_length=PDT,  # number of time steps for each prediction
            windows=n_windows,  # number of windows in rolling window evaluation
            distance=window_distance,  # number of time steps between each window - distance=PDT for non-overlapping windows
        )

        moirai_syn_preds = {}
        print("ds: ", ds)
        exit()
        for MOIRAI_SIZE in ["base"]:
                model = MoiraiForecast(
                module=MoiraiModule.from_pretrained(f"Salesforce/moirai-1.0-R-{SIZE}"),
                prediction_length=PDT,
                context_length=CTX,
                patch_size=PSZ,
                num_samples=100,
                target_dim=1,
                feat_dynamic_real_dim=ds.num_feat_dynamic_real,
                past_feat_dynamic_real_dim=ds.num_past_feat_dynamic_real,
                )

                predictor = model.create_predictor(batch_size=BSZ)
                forecasts = predictor.predict(test_data.input)

                input_it = iter(test_data.input)
                label_it = iter(test_data.label)
                forecast_it = iter(forecasts)

                forecast_out = []
                forecast_vals = []
                forecast_dates = []
                for _ in tqdm(range(test_data.windows)):
                        # Make predictions
                        inp = next(input_it)
                        label = next(label_it)
                        forecast = next(forecast_it)
                        forecast_out.append(forecast)
                tmp_moirai_preds = pd.DataFrame([[x.quantile(0.5)[0], x.start_date.start_time.date()] for x in forecast_out], 
                                                columns=["y_pred_moirai", "date"]).set_index("date")
                moirai_syn_preds[MOIRAI_SIZE] = tmp_moirai_preds.copy()


        plot_df = syn_df.copy()

        for MOIRAI_SIZE in ["base"]:
            tag = "_" + MOIRAI_SIZE[0].upper()
            plot_df = pd.merge(plot_df, moirai_syn_preds[MOIRAI_SIZE].add_suffix(tag), left_index=True, right_index=True, how="left")
            plot_df[f"err_moirai{tag}"] = plot_df[f"y_pred_moirai{tag}"] - plot_df["y"]
            plot_df[f"err_moirai{tag}_abs"] = plot_df[f"err_moirai{tag}"].abs()

        #print("plot_df", plot_df)

        display(HTML(f"<h3>feature_cols={feature_cols}</h3>"))
        plot_df[["y", "y_pred_moirai_B"]].dropna().plot(figsize=(14, 4), lw=1.0)
        plt.show()

        display(plot_df.dropna(subset=["y_pred_moirai_B"], how="any")[["err_moirai_B_abs"]].describe())