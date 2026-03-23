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
from canari.component import LocalTrend, LocalAcceleration, LstmNetwork, WhiteNoise, Autoregression

with open("examples/benchmark/BM_metadata_global.json", "r") as f:
    metadata = json.load(f)


def main(
    num_trial_optim_model: int = 50,
    param_optimization: bool = False,
    benchmark_no: str = ["2"],
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
        look_back_len = 52
        lstm = Model(
            LstmNetwork(
                    look_back_len=look_back_len,
                    num_features=config["num_feature"],
                    num_layer=5,
                    infer_len=config["infer_len"],
                    num_hidden_unit=40,
                    smoother=False,
                    load_lstm_net="saved_params/hq_g_seq_52_5layer.bin",
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
                    num_layer=5,
                    infer_len=52*1,
                    num_hidden_unit=40,
                    smoother=False,
                    # manual_seed=1,
                ),
                WhiteNoise(std_error=std_residual),
            )
            
            init_lstm_dict = model.lstm_net.state_dict()
            for key, value in init_lstm_dict.items():
                tmp = list(lstm_dict[key])
                factor = 1
                tmp[1] = np.array(value[1])*factor
                tmp[3] = np.array(value[3])*factor
                lstm_dict[key] = tuple(tmp)
            model.lstm_net.load_state_dict(lstm_dict)

            # lstm_states = model.lstm_net.get_lstm_states()
            # for key in lstm_states:
            #     values = 1*np.ones((40))
            #     lstm_states[key] = (values, values, values, values)
            # model.lstm_net.set_lstm_states(lstm_states)

            model.auto_initialize_baseline_states(
                train_data["y"][
                    config["init_period_states"][0] : config["init_period_states"][1]
                ]
            )

            num_epoch = 50
            for epoch in range(num_epoch):
                mu_validation_preds, std_validation_preds, states = model.lstm_train(
                    train_data=train_data,
                    validation_data=validation_data,
                    white_noise_decay=True,
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
                )
                model.metric_optim = model.early_stop_metric

                if model.stop_training:
                    break

            print(f"Optimal epoch: #{model.optimal_epoch}")
            #### Define SKF model with parameters #########
            abnorm_model = Model(
                LocalAcceleration(),
                LstmNetwork(),
                WhiteNoise(),
            )
            skf = SKF(
                norm_model=model,
                abnorm_model=abnorm_model,
                std_transition_error=1e-4,
                norm_to_abnorm_prob=1e-5,
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
        print(f"sum std lstm/residual: {mean_std+std_residual:0.4f}")

        fig, ax = plot_skf_states(
            data_processor=data_processor,
            states=states,
            model_prob=filter_marginal_abnorm_prob,
            standardization=True,
            states_to_plot=["level","trend","acceleration","lstm", "autorgression"],
        )
        fig.suptitle("SKF hidden states", fontsize=10, y=1)
        plt.savefig(f"{config['saved_result_path']}_LL_obj_agvi_1.png")
        plt.show()


if __name__ == "__main__":
    fire.Fire(main)
