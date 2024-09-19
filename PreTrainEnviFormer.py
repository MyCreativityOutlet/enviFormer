import os
from argparse import ArgumentParser, Namespace
import pytorch_lightning as pl
import subprocess
from pytorch_lightning.loggers import TensorBoardLogger
from pytorch_lightning.callbacks import LearningRateMonitor, ModelCheckpoint, EarlyStopping
import json
import torch
from utils.FormatData import get_dataset
from models.BaseModel import BaseModel
from models.EnviFormerModel import EnviFormerModel


def build_trainer(args: Namespace, config: dict, monitor_value="val_seq_acc") -> pl.Trainer:
    logger = TensorBoardLogger(f"results/", name=args.model_name,
                               version=args.data_name + "_" + args.tokenizer)
    lr_monitor = LearningRateMonitor(logging_interval="epoch")
    patience = 10
    monitor_mode = "max" if "acc" in monitor_value else "min"
    early_stopping = EarlyStopping(monitor=monitor_value, mode=monitor_mode, patience=patience, check_on_train_epoch_end=False)
    checkpoint_cb = ModelCheckpoint(monitor=monitor_value, mode=monitor_mode)
    callbacks = [lr_monitor, checkpoint_cb, early_stopping]
    acc_grad_batch = max(1, int(args.batch_size / config["batch_size"]))
    return pl.Trainer(accelerator="auto", logger=logger, strategy="auto", devices=args.gpus, callbacks=callbacks,
                      default_root_dir=f"results/{args.model_name}/",
                      max_epochs=180 if args.debug else args.epochs, gradient_clip_val=1.0,
                      accumulate_grad_batches=1, log_every_n_steps=10,
                      num_sanity_val_steps=2 if args.debug else 0, deterministic="warn")


def setup_model(args: Namespace) -> tuple[BaseModel, dict]:
    try:
        with open(f"models/{args.model_name}_config.json") as config_file:
            config = json.load(config_file)
    except FileNotFoundError:
        raise FileNotFoundError(f"Could not find a config file for {args.model_name}")
    model_class = globals()[args.model_name]
    print(config)
    return model_class, config


def find_latest_ckpt(f):
    epoch = f.split("=")[1]
    epoch = epoch.split("-")[0]
    return int(epoch)


def main(args: Namespace) -> None:
    print(f"Debug: {args.debug}\nModel: {args.model_name}\nDataset: {args.data_name}\nTokenizer: {args.tokenizer}")
    print("Setting up")
    pl.seed_everything(2)
    torch.set_float32_matmul_precision('high')
    model_class, config = setup_model(args)
    data, tokenizer = get_dataset(args)
    model = model_class(config, vocab=tokenizer, p_args=args)
    trainer = build_trainer(args, config)
    train_loader, val_loader, test_loader = model.encode_data(data)

    print("Beginning training")
    ckpt_path = f"results/{args.model_name}/{args.data_name}_{args.tokenizer}/checkpoints"
    if os.path.exists(ckpt_path):
        ckpt_file = [f for f in os.listdir(ckpt_path) if ".ckpt" in f]
        ckpt_file = max(ckpt_file, key=find_latest_ckpt)
        ckpt_path = os.path.join(ckpt_path, ckpt_file) if len(ckpt_path) > 0 else None
    else:
        ckpt_path = None
    trainer.fit(model, train_loader, val_loader, ckpt_path=ckpt_path)

    print("Evaluating on test set")
    trainer.test(model, test_loader, ckpt_path=trainer.checkpoint_callback.best_model_path)

    with open(f"results/{args.model_name}/{args.data_name}_{args.tokenizer}/tokens.json", "w") as token_file:
        json.dump(tokenizer, token_file, indent=4)
    with open(f"results/{args.model_name}/{args.data_name}_{args.tokenizer}/config.json", "w") as c_file:
        json.dump(config, c_file, indent=4)
    return


if __name__ == "__main__":
    proc = subprocess.Popen(["java", "-jar", "java/envirule-2.6.0-jar-with-dependencies.jar"])
    parser = ArgumentParser()
    parser.add_argument("--model_name", type=str, default="EnviFormerModel", help="Valid models include: EnviFormerModel")
    parser.add_argument("--data_name", type=str, default="uspto", help="Which dataset to use, uspto, envipath")
    parser.add_argument("--tokenizer", type=str, default="regex", help="Style of tokenizer, regex")
    parser.add_argument("--max-len", type=int, default=256, help="Maximum encoded length to consider")
    parser.add_argument("--min-len", type=int, default=0, help="Minimum encoded length to consider")
    parser.add_argument("--test-mapping", action="store_true", help="Whether to remove atom mapping before calculating accuracy")
    parser.add_argument("--score-all", action="store_true", help="Whether to group same reactants together")
    parser.add_argument("--augment-count", type=int, default=-1, help="How much SMILES augmentation to do, -1 disables")
    parser.add_argument("--debug", action="store_true", default=False, help="Whether to set parameters for debugging")
    parser.add_argument("--gpus", type=int, default=1, help="Number of GPUs to use")
    parser.add_argument("--epochs", type=int, default=250, help="Number of training epochs")
    parser.add_argument("--batch-size", type=int, default=128, help="Batch size to target")
    parser.add_argument("--weights-dataset", type=str, default="", help="Pretrained weights based on what dataset")
    try:
        main(parser.parse_args())
    except Exception as e:
        proc.kill()
        raise e
    proc.kill()
