import fire
import pickle
import json
import pandas as pd
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
from scipy.stats import norm, lognorm

with open("examples/benchmark/BM_metadata.json", "r") as f:
    metadata = json.load(f)


def main(
    num_trial_optim_model: int = 200,
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
        def model_with_parameters(param):
            model = Model(
                LocalTrend(),
                LstmNetwork(
                    look_back_len=param["look_back_len"],
                    num_features=config["num_feature"],
                    num_layer=1,
                    infer_len=config["infer_len"],
                    num_hidden_unit=50,
                    manual_seed=1,
                    smoother=config["smoother"],
                ),
                WhiteNoise(std_error=param["sigma_v"]),
            )

            model.auto_initialize_baseline_states(
                train_data["y"][
                    config["init_period_states"][0] : config["init_period_states"][1]
                ]
            )
            num_epoch = config["num_epoch"]
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
                )
                model.metric_optim = model.early_stop_metric

                if model.stop_training:
                    break

            #### Define SKF model with parameters #########
            abnorm_model = Model(
                LocalAcceleration(),
                LstmNetwork(),
                WhiteNoise(),
            )
            skf = SKF(
                norm_model=model,
                abnorm_model=abnorm_model,
                std_transition_error=param["std_transition_error"],
                norm_to_abnorm_prob=param["norm_to_abnorm_prob"],
            )

            skf.save_initial_states()

            num_anomaly = 50
            detection_rate, false_rate, false_alarm_train = (
                skf.detect_synthetic_anomaly(
                    data=train_data,
                    num_anomaly=num_anomaly,
                    slope_anomaly=param["slope"] / 52,
                )
            )

            data_len_year = (
                data_processor.data.index[data_processor.train_end]
                - data_processor.data.index[data_processor.train_start]
            ).days / 365.25

            false_rate_yearly = false_rate / data_len_year
            metric_optim = skf.objective(
                detection_rate, false_rate_yearly, param["slope"]
            )

            skf.load_initial_states()
            skf.metric_optim = metric_optim.copy()
            print_metric = {}
            print_metric["detection_rate"] = detection_rate
            print_metric["yearly_false_rate"] = false_rate_yearly
            skf.print_metric = print_metric

            return skf

        ######### Parameter optimization #########
        if param_optimization:
            param_space = {
                "look_back_len": config["look_back_len"],
                "sigma_v": config["sigma_v"],
                "std_transition_error": config["std_transition_error"],
                "norm_to_abnorm_prob": config["norm_to_abnorm_prob"],
                "slope": config["slope"],
            }
            # Define optimizer
            model_optimizer = Optimizer(
                model=model_with_parameters,
                param=param_space,
                num_optimization_trial=num_trial_optim_model,
                num_startup_trials=50,
                mode="max",
            )
            model_optimizer.optimize()
            # Get best model
            param = model_optimizer.get_best_param()
            skf_optim = model_with_parameters(param)

            skf_optim_dict = skf_optim.get_dict()
            skf_optim_dict["model_param"] = param
            skf_optim_dict["cov_names"] = train_data["cov_names"]
            with open(f"{config['saved_model_path']}_cdf_obj.pkl", "wb") as f:
                pickle.dump(skf_optim_dict, f)
        else:
            # # Load saved skf model
            with open(f"{config['saved_model_path']}_cdf_obj.pkl", "rb") as f:
                skf_optim_dict = pickle.load(f)
            skf_optim = SKF.load_dict(skf_optim_dict)

        ######### Detect anomaly #########
        print("Model parameters used:", skf_optim_dict["model_param"])

        filter_marginal_abnorm_prob, states = skf_optim.filter(data=all_data)

        fig, ax = plot_skf_states(
            data_processor=data_processor,
            states=states,
            model_prob=filter_marginal_abnorm_prob,
        )
        fig.suptitle("SKF hidden states", fontsize=10, y=1)
        plt.savefig(f"{config['saved_result_path']}_cdf_obj.png")
        plt.show()

        # Plot a sample of anomaly with optimal magnitude
        synthetic_anomaly_data = DataProcess.add_synthetic_anomaly(
            train_data,
            num_samples=1,
            slope=[skf_optim_dict["model_param"]["slope"] / 52],
        )

        train_time = data_processor.get_time("train")
        plt.plot(train_time, synthetic_anomaly_data[0]["y"])
        plot_data(
            data_processor=data_processor,
            standardization=True,
            plot_validation_data=False,
            plot_test_data=False,
            plot_column=output_col,
            train_label="data without anomaly",
        )
        plt.legend(
            [
                "data with optimal anomaly slope",
                "data without anomaly",
            ]
        )
        plt.title("Train data with added synthetic anomalies")
        plt.show()


if __name__ == "__main__":
    fire.Fire(main)
