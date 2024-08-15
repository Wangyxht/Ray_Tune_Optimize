from typing import Union

from ray import tune, train
from ray.tune.schedulers import *
from ray.tune.search.optuna import OptunaSearch
from optuna.samplers import TPESampler, GPSampler


def optimize_random_search(
        objective=None,
        config: dict = None,
        metric: str = None,
        mode: str = None,
        n_samples: int = 10,
        max_iter: int = 10,
        scheduler: str = None,
        grace_period: int = None,
        reduce_factor: float = None,
        cpus_per_trial: float = 1,
        gpus_per_trial: float = 1,
):
    """
    随机/网格搜索超参数优化
    :param scheduler: 用于实现Hyperband或者ASHA算法的调度器
    :param objective: 目标函数
    :param config: 搜索空间以及传递的的参数
    :param metric: 优化目标量,需要与目标函数report的变量名一致
    :param mode: 优化模式,可取max、min
    :param n_samples: 采样点个数
    :param max_iter: 训练轮数
    :param cpus_per_trial: 每个实验的CPU资源
    :param gpus_per_trial: 每个实验的GPU资源
    :return: 最优参数
    """
    if objective is None:
        raise ValueError("优化目标objective未指定")
    if config is None:
        raise ValueError("超参数搜索空间config未指定")
    if metric is None or mode is None:
        raise ValueError("优化目标metric或优化模式mode未指定")
    if mode not in ["max", "min"]:
        raise ValueError("优化模式mode取值范围为min/max")
    if scheduler is None and grace_period is not None and reduce_factor is not None:
        raise ValueError("指定grace_period reduce_factor参数需要指定调度器scheduler")

    trainable_with_resources = tune.with_resources(
        trainable=objective,
        resources={
            "cpu": cpus_per_trial,
            "gpu": gpus_per_trial
        }
    )

    _scheduler = _create_scheduler(
        schedule_type=scheduler,
        metric=metric,
        mode=mode,
        max_iter=max_iter,
        reduce_factor=reduce_factor if reduce_factor is not None else 3,
        grace_period=grace_period if grace_period is not None else 1,
    )

    tuner = tune.Tuner(
        trainable=trainable_with_resources,
        run_config=train.RunConfig(
            stop={"training_iteration": max_iter},
        ),
        tune_config=tune.TuneConfig(
            scheduler=_scheduler,
            num_samples=n_samples,
        ),
        param_space=config,
    )

    results = tuner.fit()
    best_result = results.get_best_result(metric=metric, mode=mode)
    print("best hyperparameter information:\n %s" % best_result.config)

    return results, best_result


def optimize_pbt_search(
        objective,
        config: dict,
        hyperparameters: dict,
        metric: str,
        mode: str,
        n_samples: int,
        max_iter: int,
        perturbation_interval: int,
        checkpoint_keep_num: int,
        cpus_per_trial: float,
        gpus_per_trial: float,
):
    """
    PBT 种群优化搜索，一般用于迭代次数相关模型的超参数优化
    :param objective: 目标函数
    :param hyperparameters:超参数空间
    :param config: 搜索空间以及传递的的参数
    :param metric: 优化目标量,需要与目标函数report的变量名一致
    :param mode: 优化模式,可取max、min
    :param n_samples: 采样点个数。
    :param max_iter: 最大迭代次数
    :param checkpoint_keep_num: 同时保存检查点的最大值
    :param perturbation_interval: 进行保存模型并且交换模型的间隔轮数
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

            stop={"training_iteration": max_iter},
            checkpoint_config=train.CheckpointConfig(
                checkpoint_score_attribute=metric,
                num_to_keep=checkpoint_keep_num,
            ),
        ),
        tune_config=tune.TuneConfig(
            scheduler=scheduler,
            num_samples=n_samples,
        ),
        param_space=config,
    )

    results = tuner.fit()
    best_result = results.get_best_result(metric=metric, mode=mode)
    print("best hyperparameter information:\n %s" % best_result.config)
    return results, best_result


def optimize_optuna_search(
        objective,
        config: dict,
        metric: str,
        mode: str,
        n_samples: int,
        max_iter: int,
        points_to_evaluate: Union[dict, list, None],
        cpus_per_trial: float,
        gpus_per_trial: float,
        optuna_sampler: str = 'TPE',
        scheduler: str = None,
):
    """
    基于Optuna的贝叶斯超参数优化,Optuna库默认使用基于TPE的贝叶斯优化
    :param scheduler:
    :param objective: 目标函数
    :param metric: 优化目标量,需要与目标函数report的变量名一致
    :param mode: 优化模式,可取max、min
    :param config: 搜索空间以及传递的参数
    :param n_samples: 采样点个数
    :param max_iter: 最大迭代次数
    :param points_to_evaluate: 设定初始值的点
    :param cpus_per_trial: 每个实验的CPU资源
    :param gpus_per_trial: 每个实验的GPU资源
    :param optuna_sampler: optuna库采样器，默认为TPE,可选GP
    :return:最优参数
    """
    if optuna_sampler is None or optuna_sampler == 'TPE':
        optuna_sampler = TPESampler()
    if optuna_sampler == 'GP':
        optuna_sampler = GPSampler()

    _scheduler = _create_scheduler(
        schedule_type=scheduler,
        metric=metric,
        mode=mode,
        max_iter=max_iter
    )

    trainable_with_resources = tune.with_resources(
        trainable=objective,
        resources={
            "cpu": cpus_per_trial,
            "gpu": gpus_per_trial
        }
    )

    optuna_search = OptunaSearch(
        points_to_evaluate=points_to_evaluate,
        metric=metric,
        mode=mode,
        sampler=optuna_sampler,
    )

    tuner = tune.Tuner(
        trainable=trainable_with_resources,
        run_config=train.RunConfig(
            # Stop when we've reached a threshold accuracy, or a maximum
            # training_iteration, whichever comes first
            stop={"training_iteration": max_iter},
        ),
        tune_config=tune.TuneConfig(
            search_alg=optuna_search,
            num_samples=n_samples,
            scheduler=_scheduler,
        ),
        param_space=config,
    )

    results = tuner.fit()
    best_result = results.get_best_result(metric=metric, mode=mode)
    print("best hyperparameter information:\n %s" % best_result.config)
    return results, best_result.config


def _create_scheduler(
        schedule_type: str = None,
        metric: str = None,
        mode: str = None,
        max_iter: int = None,
        grace_period: int = 1,
        reduce_factor: float = 3,
):
    if schedule_type is None:
        return None
    elif schedule_type == 'ASHA':
        return ASHAScheduler(
            max_t=max_iter,
            metric=metric,
            mode=mode,
            reduction_factor=reduce_factor,
            grace_period=grace_period,
        )

    elif schedule_type == 'HyperBand':
        return HyperBandScheduler(
            max_t=max_iter,
            metric=metric,
            mode=mode,
            grace_period=grace_period,
            reduction_factor=reduce_factor,
        )
    else:
        raise ValueError('schedule type error')
