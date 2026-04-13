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
    benchmark_no: str = ["4"],
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
                    num_layer=3,
                    infer_len=config["infer_len"],
                    num_hidden_unit=40,
                    smoother=False,
                    load_lstm_net="saved_params/hq_g_seq_52.bin",
                )
        )
        lstm_dict = lstm.lstm_net.state_dict()
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
                    manual_seed=1,
                ),
                WhiteNoise(std_error=std_residual),
            )
            
            model.lstm_net.load_state_dict(lstm_dict)

            model.auto_initialize_baseline_states(
                train_data["y"][
                    config["init_period_states"][0] : config["init_period_states"][1]
                ]
            )

            model.var_states[0,0] = 1e-3
            model.var_states[1,1] = 1e-7

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

            return (
                model
            )


        def skf_with_parameters(skf_param_space, skf_input):
            norm_model = Model.load_dict(skf_input["model_optim_dict"])

            abnorm_model = Model(
                LocalAcceleration(),
                LstmNetwork(),
                WhiteNoise(),
            )
            skf = SKF(
                norm_model=norm_model,
                abnorm_model=abnorm_model,
                std_transition_error=skf_param_space["std_transition_error"],
                norm_to_abnorm_prob=skf_param_space["norm_to_abnorm_prob"],
                abnorm_to_norm_prob = skf_param_space["abnorm_to_norm_prob"],
            )

            # CDF
            # skf.save_initial_states()

            # num_anomaly = 50
            # detection_rate, false_rate, _ = skf.detect_synthetic_anomaly(
            #     data=train_data,
            #     num_anomaly=num_anomaly,
            #     slope_anomaly=skf_param_space["slope"] / 52,
            # )

            # data_len_year = (
            #     data_processor.data.index[data_processor.train_end]
            #     - data_processor.data.index[data_processor.train_start]
            # ).days / 365.25

            # false_rate_yearly = false_rate / data_len_year
            # metric_optim = skf.objective(
            #     detection_rate, false_rate_yearly, skf_param_space["slope"]
            # )

            # skf.load_initial_states()
            # skf.metric_optim = metric_optim.copy()
            # print_metric = {}
            # print_metric["detection_rate"] = detection_rate
            # print_metric["yearly_false_rate"] = false_rate_yearly
            # skf.print_metric = print_metric

            # Log-likelihood
            skf.save_initial_states()
            skf.filter(data=all_data)
            log_lik_all = np.nanmean(skf.ll_history)
            skf.metric_optim = -log_lik_all
            skf.load_initial_states()

            return skf


        if param_optimization:
            # Define parameter search space
            param = {}
            model_optim = (
                model_with_parameters(param)
            )

            # Save best model for SKF analysis later
            model_optim_dict = model_optim.get_dict(time_step=0)

            # # Optimize for skf
            skf_param_space = {
                "std_transition_error": config["std_transition_error"],
                "norm_to_abnorm_prob": config["norm_to_abnorm_prob"],
                # "slope": config["slope"],
                "abnorm_to_norm_prob": [0.1, 0.2],
            }

            skf_input = {}
            skf_input["model_optim_dict"] = model_optim_dict
            skf_optimizer = Optimizer(
                model=skf_with_parameters,
                param=skf_param_space,
                model_input=skf_input,
                num_optimization_trial=60,
                num_startup_trials=30,
                mode="min",
            )
            skf_optimizer.optimize()
            # Get parameters
            skf_param = skf_optimizer.get_best_param()

            skf_optim = skf_with_parameters(skf_param, skf_input)
            skf_optim_dict = skf_optim.get_dict()
            skf_optim_dict["model_param"] = param
            skf_optim_dict["skf_param"] = skf_param
            skf_optim_dict["cov_names"] = train_data["cov_names"]
            with open(f"{config['saved_model_path']}_global.pkl", "wb") as f:
                pickle.dump(skf_optim_dict, f)
        else:
            # # Load saved skf model
            with open(f"{config['saved_model_path']}_global.pkl", "rb") as f:
                skf_optim_dict = pickle.load(f)
            skf_optim = SKF.load_dict(skf_optim_dict)
        

        ######### Detect anomaly #########
        filter_marginal_abnorm_prob, states = skf_optim.filter(data=all_data)

        mean_std = states.get_std(states_name="lstm")
        mean_std = np.nanmean(mean_std)
        print(f"average std lstm: {mean_std:0.4f} ")
        # print(f"sum std lstm/residual: {mean_std+std_residual:0.4f}")

        fig, ax = plot_skf_states(
            data_processor=data_processor,
            states=states,
            model_prob=filter_marginal_abnorm_prob,
            standardization=True,
        )
        fig.suptitle("SKF hidden states", fontsize=10, y=1)
        plt.savefig(f"{config['saved_result_path']}_global.png")
        plt.show()


if __name__ == "__main__":
    fire.Fire(main)
