import fire
import pickle
import json
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from ray import tune
from pytagi import metric
from pytagi import Normalizer as normalizer
from canari import (
    DataProcess,
    Model,
    Optimizer,
    SKF,
    plot_data,
    plot_prediction,
    plot_skf_states,
    plot_states,
)
from canari.component import LocalTrend, LocalAcceleration, LstmNetwork, WhiteNoise

with open("examples/benchmark/BM_metadata_global.json", "r") as f:
    metadata = json.load(f)


def main(
    num_trial_optim_model: int = 1,
    param_optimization: bool = False,
    benchmark_no: str = ["6"],
):
    for benchmark in benchmark_no:

        # Load configuration from metadata for a specific benchmark
        config = metadata[benchmark]
        print("----------------------------")
        print(f"Benchmark being analyzed: #{benchmark}")
        print("----------------------------")

        ######### Data processing #########
        # Read data
        data_file = config["data_path"]
        df = pd.read_csv(data_file, skiprows=0, delimiter=",")
        date_time = pd.to_datetime(df["date"])
        df = df.drop("date", axis=1)
        df = df.iloc[:, [0]]
        # df = df.interpolate(method="linear")
        df.index = date_time
        df.index.name = "date_time"
        # Data pre-processing
        df = DataProcess.add_lagged_columns(df, config["lag_vector"])
        output_col = config["output_col"]
        data_processor = DataProcess(
            data=df,
            time_covariates=config["time_covariates"],
            train_split=config["train_split"],
            validation_split=config["validation_split"],
            output_col=output_col,
        )
        train_data, validation_data, _, all_data = data_processor.get_splits()

        ######### Define model with parameters #########
        look_back_len = 36
        lstm = Model(
            LstmNetwork(
                    look_back_len=look_back_len,
                    num_features=config["num_feature"],
                    num_layer=3,
                    infer_len=config["infer_len"],
                    num_hidden_unit=40,
                    smoother=False,
                    load_lstm_net="saved_params/hq_global_seq_36.bin",
                )
        )
        lstm_dict = lstm.lstm_net.state_dict()

        # lstm_dict["SLSTM.0"] = lstm_dict.pop("LSTM.0")
        # lstm_dict["SLSTM.1"] = lstm_dict.pop("LSTM.1")
        # lstm_dict["SLSTM.2"] = lstm_dict.pop("LSTM.2")
        # lstm_dict["SLinear.3"] = lstm_dict.pop("Linear.3")
        
        std_residual = 1e-1

        def model_with_parameters(param):
            model = Model(
                LocalTrend(),
                LstmNetwork(
                    look_back_len=look_back_len,
                    num_features=config["num_feature"],
                    num_layer=3,
                    infer_len=52*3,
                    num_hidden_unit=40,
                    smoother=False,
                ),
                WhiteNoise(std_error=std_residual),
            )

            model.lstm_net.load_state_dict(lstm_dict)

            model.auto_initialize_baseline_states(
                train_data["y"][
                    config["init_period_states"][0] : config["init_period_states"][1]
                ]
            )
            # model.var_states[0,0] = 1e-3
            model.var_states[1,1] = 1e-5

            num_epoch = 50
            for epoch in range(num_epoch):
                mu_validation_preds, std_validation_preds, states = model.lstm_train(
                    train_data=train_data,
                    validation_data=validation_data,
                )

                mu_validation_preds_unnorm = normalizer.unstandardize(
                    mu_validation_preds,
                    data_processor.scale_const_mean[data_processor.output_col],
                    data_processor.scale_const_std[data_processor.output_col],
                )

                std_validation_preds_unnorm = normalizer.unstandardize_std(
                    std_validation_preds,
                    data_processor.scale_const_std[data_processor.output_col],
                )

                validation_obs = data_processor.get_data("validation").flatten()
                validation_log_lik = metric.log_likelihood(
                    prediction=mu_validation_preds_unnorm,
                    observation=validation_obs,
                    std=std_validation_preds_unnorm,
                )

                model.early_stopping(
                    evaluate_metric=-validation_log_lik,
                    current_epoch=epoch,
                    max_epoch=num_epoch,
                    skip_epoch=0,
                )
                model.metric_optim = model.early_stop_metric

                if model.stop_training:
                    break

            #### Define SKF model with parameters #########
            abnorm_model = Model(
                LocalAcceleration(),
                LstmNetwork(),
                WhiteNoise()
            )
            skf = SKF(
                norm_model=model,
                abnorm_model=abnorm_model,
                std_transition_error=1e-4,
                norm_to_abnorm_prob=1e-5,
                abnorm_to_norm_prob = 0.14,
            )

            skf.save_initial_states()

            skf.filter(data=all_data)
            log_lik_all = np.nanmean(skf.ll_history)
            skf.metric_optim = -log_lik_all

            skf.load_initial_states()
            return skf
        
        param = {}
        skf_optim = model_with_parameters(param)

        skf_optim_dict = skf_optim.get_dict()
        skf_optim_dict["model_param"] = param
        skf_optim_dict["cov_names"] = train_data["cov_names"]
        with open(f"{config['saved_model_path']}_global_seq_{look_back_len}.pkl", "wb") as f:
            pickle.dump(skf_optim_dict, f)

        ######### Detect anomaly #########
        filter_marginal_abnorm_prob, states = skf_optim.filter(data=all_data)

        mean_std = states.get_std(states_name="lstm")
        mean_std = np.nanmean(mean_std)
        print(f"average std lstm: {mean_std:0.4f} ")
        print(f"ratsumio std lstm/residual: {mean_std+std_residual:0.4f}")

        fig, ax = plot_skf_states(
            data_processor=data_processor,
            states=states,
            model_prob=filter_marginal_abnorm_prob,
            standardization=True,
        )
        fig.suptitle("SKF hidden states", fontsize=10, y=1)
        plt.savefig(f"{config['saved_result_path']}_LL_obj_agvi_1.png")
        plt.show()


if __name__ == "__main__":
    fire.Fire(main)
