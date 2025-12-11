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

with open("examples/benchmark/BM_metadata.json", "r") as f:
    metadata = json.load(f)


def main(
    num_trial_optim_model: int = 10,
    param_optimization: bool = True,
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

            mu_preds, std_preds, _, _ = skf.filter(data=all_data)

            mu_preds_unnorm = normalizer.unstandardize(
                mu_preds,
                data_processor.scale_const_mean[data_processor.output_col],
                data_processor.scale_const_std[data_processor.output_col],
            )

            std_preds_unnorm = normalizer.unstandardize_std(
                std_preds,
                data_processor.scale_const_std[data_processor.output_col],
            )

            obs_all = data_processor.get_data("all").flatten()
            log_lik_all = metric.log_likelihood(
                prediction=mu_preds_unnorm,
                observation=obs_all,
                std=std_preds_unnorm,
            )
            skf.metric_optim = -log_lik_all

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
            with open(config["saved_model_path"], "wb") as f:
                pickle.dump(skf_optim_dict, f)
        else:
            # # Load saved skf model
            with open(config["saved_model_path"], "rb") as f:
                skf_optim_dict = pickle.load(f)
            skf_optim = SKF.load_dict(skf_optim_dict)

        ######### Detect anomaly #########
        print("Model parameters used:", skf_optim_dict["model_param"])
        print("SKF model parameters used:", skf_optim_dict["skf_param"])

        _, _, states, filter_marginal_abnorm_prob = skf_optim.filter(data=all_data)

        fig, ax = plot_skf_states(
            data_processor=data_processor,
            states=states,
            model_prob=filter_marginal_abnorm_prob,
        )
        fig.suptitle("SKF hidden states", fontsize=10, y=1)
        plt.savefig(config["saved_result_path"])
        plt.show()


if __name__ == "__main__":
    fire.Fire(main)
