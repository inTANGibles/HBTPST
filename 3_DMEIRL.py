import argparse
import json
import os

from grid_world.grid_world import GridWorld
from DMEIRL.DeepMEIRL_FC import DMEIRL
from utils_tool import utils


def parse_args():
    p = argparse.ArgumentParser(
        description="默认运行 Optuna 超参搜索；加 --single-run 则只跑单次 DMEIRL 训练。"
    )
    p.add_argument(
        "--single-run",
        action="store_true",
        help="不进行 Optuna，仅按 --lr / --weight-decay 等参数训练一次。",
    )
    p.add_argument(
        "--optuna",
        action="store_true",
        help="兼容旧用法；当前默认已是 Optuna，无需再写。",
    )
    p.add_argument("--n-trials", type=int, default=20)
    p.add_argument("--optuna-epochs", type=int, default=300, help="Epochs per trial (use a smaller value for screening).")
    p.add_argument("--epochs", type=int, default=1000, help="Epochs for a normal single run.")
    p.add_argument("--study-name", type=str, default="dmeirl_svf")
    p.add_argument(
        "--storage",
        type=str,
        default=None,
        help="Optuna storage URL, e.g. sqlite:///optuna_dmeirl.db (enables resume).",
    )
    p.add_argument("--lr", type=float, default=5e-5)
    p.add_argument("--weight-decay", type=float, default=0.25)
    p.add_argument("--clip-norm", type=float, default=-1.0, help="Gradient clip max norm; -1 disables clipping.")
    p.add_argument("--traj-len-bias", type=int, default=None, help="If set, call experts.ChangeTrajLenBias before training.")
    return p.parse_args()


def make_world():
    env_folder_path = "wifi_track_data/dacang/grid_data/env_imgs/40_30"
    expert_traj_path = "wifi_track_data/dacang/track_data/trajs_sliced_513_40x30.csv"

    if not os.path.isfile(expert_traj_path):
        raise FileNotFoundError(
            f"专家轨迹文件不存在: {expert_traj_path}\n"
            "（若文件在 cluster_data 下，请改路径或先生成该 CSV。）"
        )

    world = GridWorld(
        expert_traj_filePath=expert_traj_path,
        environments_img_folderPath=env_folder_path,
        width=40,
        height=30,
        discount=0.95,
        trans_prob=0.8,
    )
    print("GridWorld initialized")
    if world.experts is None:
        raise RuntimeError(
            "world.experts 未初始化（GridWorld 在加载专家轨迹时失败）。请检查 expert_traj_path 与 CSV 列 trajs/m。"
        )
    return world


def default_layers():
    return (64, 128, 128, 64)


def _params_jsonable(params):
    out = {}
    for k, v in params.items():
        out[k] = list(v) if isinstance(v, tuple) else v
    return out


def run_optuna(world, args):
    import optuna

    layer_candidates = [
        (64, 128, 128, 64),
        (60, 120, 60),
        (60, 240, 60),
        (120, 240, 120),
        (60, 120, 240, 120, 60),
    ]

    def objective(trial) -> float:
        lr = trial.suggest_float("lr", 1e-5, 3e-3, log=True)
        weight_decay = trial.suggest_float("weight_decay", 0.05, 0.6)
        layers = trial.suggest_categorical("layers", layer_candidates)
        clip_norm = trial.suggest_categorical("clip_norm", [-1, 0.5, 1.0])
        bias = trial.suggest_int("traj_len_bias", 0, 20, step=5)

        world.experts.ChangeTrajLenBias(int(bias))
        log = f"{utils.date}sliced_bias{bias}_v0.001_tp{world.trans_prob}_dis{world.discount}_opt{trial.number}"
        dme = DMEIRL(
            world,
            layers=layers,
            lr=lr,
            weight_decay=weight_decay,
            clip_norm=clip_norm,
            log=log,
            log_dir="run_sliced",
        )
        _br, _bi, _rh, best_mse = dme.train(
            n_epochs=args.optuna_epochs,
            demo=True,
            save=False,
        )
        return float(best_mse)

    study_kwargs = {"direction": "minimize", "study_name": args.study_name}
    if args.storage:
        study_kwargs["storage"] = args.storage
        study_kwargs["load_if_exists"] = True
    study = optuna.create_study(**study_kwargs)
    study.optimize(objective, n_trials=args.n_trials, show_progress_bar=True)

    print("Best trial:")
    t = study.best_trial
    print("  value (min SVF MSE):", t.value)
    print("  params:", t.params)

    out_dir = os.path.join("train", utils.date)
    os.makedirs(out_dir, exist_ok=True)
    best_path = os.path.join(out_dir, f"optuna_best_{args.study_name}.json")
    payload = {
        "study_name": args.study_name,
        "n_trials": args.n_trials,
        "optuna_epochs": args.optuna_epochs,
        "storage": args.storage,
        "best_value": float(t.value),
        "best_params": _params_jsonable(t.params),
        "best_trial_number": t.number,
    }
    with open(best_path, "w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2, ensure_ascii=False)
    print(f"Best trial 已写入: {best_path}")
    return study


def main():
    args = parse_args()
    world = make_world()

    # Apply cluster (optional)
    # df_cluster = pd.read_csv('wifi_track_data/dacang/cluster_data/cluster_result.csv')
    # world.experts.ReadCluster(df_cluster)
    # world.experts.ApplyCluster([0])

    if not args.single_run:
        run_optuna(world, args)
        return

    if args.traj_len_bias is not None:
        world.experts.ChangeTrajLenBias(int(args.traj_len_bias))

    layers = default_layers()
    clip_norm = -1 if args.clip_norm == -1.0 else args.clip_norm

    dme = DMEIRL(
        world,
        layers=layers,
        lr=args.lr,
        weight_decay=args.weight_decay,
        clip_norm=clip_norm,
        log=f"{utils.date}sliced_bias{world.traj_len_bias}_v0.001_tp{world.trans_prob}_dis{world.discount}",
        log_dir="run_sliced",
    )
    dme.train(n_epochs=args.epochs)


if __name__ == "__main__":
    main()
