from ray import tune, train
from ray.tune.schedulers import PopulationBasedTraining, ASHAScheduler, AsyncHyperBandScheduler, HyperBandScheduler
from ray.tune.search.optuna import OptunaSearch


def optimize_random_search(
        objective,
        config: dict,
        metric: str,
        mode: str,
        n_samples: int,
        epochs: int,
        cpus_per_trial: float,
        gpus_per_trial: float):
    """
    随机/网格搜索超参数优化
    :param objective: 目标函数
    :param config: 搜索空间以及传递的的参数
    :param metric: 优化目标量,需要与目标函数report的变量名一致
    :param mode: 优化模式,可取max、min
    :param n_samples: 采样点个数
    :param epochs: 训练轮数
    :param cpus_per_trial: 每个实验的CPU资源
    :param gpus_per_trial: 每个实验的GPU资源
    :return: 最优参数
    """
    results = tune.run(
        objective,
        resources_per_trial={"cpu": cpus_per_trial, "gpu": gpus_per_trial},
        num_samples=n_samples,
        stop={"training_iteration": epochs},
        config=config,
        metric=metric,
        mode=mode,
    )
    return results


def optimize_ASHA_search(
        objective,
        config: dict,
        metric: str,
        mode: str,
        n_samples: int,
        epochs: int,
        grace_period: int,
        reduce_factor: int,
        cpus_per_trial: float,
        gpus_per_trial: float,
):
    """
    加入早停策略的随机/网格搜索超参数优化，
    ASHA调度器将会在初期将评价不良的函数淘汰
    :param objective: 目标函数
    :param config: 搜索空间以及传递的的参数
    :param metric: 优化目标量,需要与目标函数report的变量名一致
    :param mode: 优化模式,可取max、min
    :param n_samples: 采样点个数
    :param epochs: 每次实验的最大训练轮数
    :param reduce_factor: 减少实验因子Eta
    :param grace_period: 每个实验最少的训练轮数，为了防止某些优质实验过早停止
    :param cpus_per_trial: 每个实验的CPU资源
    :param gpus_per_trial: 每个实验的GPU资源
    :return: 最优参数
    """

    # ASHA调度器，会自动淘汰一些在早期训练较差的实验
    # In ASHA, you can decide how many trials are early terminated.
    # reduction_factor=4 means that only 25% of all trials are kept each time they are reduced.
    # With grace_period=n you can force ASHA to train each trial at least for n epochs.
    scheduler = ASHAScheduler(
        time_attr='training_iteration',
        max_t=epochs,
        metric=metric,
        mode=mode,
        reduction_factor=reduce_factor,
        grace_period=grace_period,
    )

    results = tune.run(
        objective,
        config=config,
        resources_per_trial={"cpu": cpus_per_trial, "gpu": gpus_per_trial},
        num_samples=n_samples,
        scheduler=scheduler
    )
    return results


def optimize_hyperBand_search(
        objective,
        config: dict,
        metric: str,
        mode: str,
        n_samples: int,
        epochs: int,
        grace_period: int,
        brackets: int,
        cpus_per_trial: float,
        gpus_per_trial: float,
):
    """
    基于HyperBand调度的随机/网格搜索超参数优化，
    :param objective: 目标函数
    :param config: 搜索空间以及传递的的参数
    :param metric: 优化目标量,需要与目标函数report的变量名一致
    :param mode: 优化模式,可取max、min
    :param n_samples: 采样点个数
    :param epochs: 每次实验的最大训练轮数
    :param brackets: HyperBand 算法所涉及的组合的概念。论文推荐值为3或4
    :param grace_period: 每个实验最少的训练轮数，为了防止某些优质实验过早停止
    :param cpus_per_trial: 每个实验的CPU资源
    :param gpus_per_trial: 每个实验的GPU资源
    :return: 最优参数
    """

    scheduler = AsyncHyperBandScheduler(
        time_attr='training_iteration',
        max_t=epochs,
        metric=metric,
        mode=mode,
        grace_period=grace_period,
        brackets=brackets
    )

    results = tune.run(
        objective,
        config=config,
        resources_per_trial={"cpu": cpus_per_trial, "gpu": gpus_per_trial},
        num_samples=n_samples,
        scheduler=scheduler
    )
    return results


def optimize_PBT_search(objective,
                        config: dict,
                        hyperparameters: dict,
                        metric: str,
                        mode: str,
                        n_samples: int,
                        cpus_per_trial: float,
                        gpus_per_trial: float,
                        perturbation_interval: int):
    """
    随机搜索超参数优化
    :param hyperparameters:
    :param objective: 目标函数
    :param config: 搜索空间以及传递的的参数
    :param metric: 优化目标量,需要与目标函数report的变量名一致
    :param mode: 优化模式,可取max、min
    :param n_samples: 采样点个数。
    :param cpus_per_trial: 每个实验的CPU资源
    :param gpus_per_trial: 每个实验的GPU资源
    :return: 最优参数
    """
    trainable_with_resources = tune.with_resources(
        trainable=objective,
        resources={
            "cpu": cpus_per_trial,
            "gpu": gpus_per_trial
        }
    )
    scheduler = PopulationBasedTraining(
        time_attr="training_iteration",
        perturbation_interval=perturbation_interval,
        metric=metric,
        mode=mode,
        hyperparam_mutations=hyperparameters,
    )

    tuner = tune.Tuner(
        trainable=trainable_with_resources,
        run_config=train.RunConfig(
            # Stop when we've reached a threshold accuracy, or a maximum
            # training_iteration, whichever comes first
            stop={"training_iteration": 10},
            checkpoint_config=train.CheckpointConfig(
                checkpoint_score_attribute=metric,
                num_to_keep=4,
            ),
        ),
        tune_config=tune.TuneConfig(
            scheduler=scheduler,
            num_samples=n_samples,
        ),
        param_space=config,
    )

    results = tuner.fit()
    return results


def optimize_optuna_search(
        objective,
        config: dict,
        metric: str,
        mode: str,
        n_samples: int,
        cpus_per_trial: float,
        gpus_per_trial: float):
    """
    基于Optuna的贝叶斯超参数优化
    :param objective: 目标函数
    :param metric: 优化目标量,需要与目标函数report的变量名一致
    :param mode: 优化模式,可取max、min
    :param config: 搜索空间以及传递的参数
    :param n_samples: 采样点个数
    :param cpus_per_trial: 每个实验的CPU资源
    :param gpus_per_trial: 每个实验的GPU资源
    :return:最优参数
    """
    results = tune.run(
        objective,
        resources_per_trial={"cpu": cpus_per_trial, "gpu": gpus_per_trial},
        num_samples=n_samples,
        config=config,
        metric=metric,
        mode=mode,
        search_alg=OptunaSearch(),
    )
    return results
