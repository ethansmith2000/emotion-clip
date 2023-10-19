import logging
import math
import os
import random
import itertools

import datasets
import torch
from accelerate import Accelerator
from accelerate.logging import get_logger
from accelerate.utils import set_seed
from torch.utils.data import DataLoader
from tqdm.auto import tqdm

import transformers
from transformers import (
    AutoConfig,
    AutoModelForSequenceClassification,
    AutoTokenizer,
    DataCollatorWithPadding,
    default_data_collator,
    get_scheduler,
)
from torch import nn

from types import SimpleNamespace

logger = get_logger(__name__)

from network import Network

from dataset import ImageDataset

from transformers import AutoModel, CLIPVisionModelWithProjection
import pandas as pd

default_args = {
    "train_file": "train.csv",
    "validation_file": "test.csv",
    "model_name_or_path": "openai/clip-vit-large-patch14",
    "per_device_train_batch_size": 64,
    "per_device_eval_batch_size": 64,
    "learning_rate": 2.0e-4,
    "eval_steps": 25,
    "weight_decay": 0.01,
    "num_train_epochs": 22,
    "max_train_steps": None,
    "gradient_accumulation_steps": 1,
    "lr_scheduler_type": "linear",
    "num_warmup_steps": 75,
    "output_dir": "emotion-clip",
    "seed": 123,
    "checkpointing_steps": "2000",
    "resume_from_checkpoint": None,
    "with_tracking": True,
    "report_to": "wandb",
    "ignore_mismatched_sizes": False,
    "hidden_dim": 384,
    "pre_layernorm": True,
    "gradient_checkpointing": True,
    "mixed_precision": "bf16",
    "max_grad_norm": 1.0,
    "do_validation": True,
    "disable_wandb": False,
    "num_transformer_layers": 1,
    "act_fn": "none",
    "save_every":150,
    "fit_method":["pad", "resize", "center"],
    "num_cutouts":0,
    "image_folder": "./iaps_images",
    "num_workers":8,
    "color_jitter": 0.15,
}


# def step_forward(model, pixel_values, accelerator):
#     hidden_states = model.embeddings(pixel_values)
#     hidden_states = model.pre_layrnorm(hidden_states)
#
#     causal_attention_mask = None
#     attention_mask=None
#     output_attentions=False
#
#     for idx, encoder_layer in enumerate(model.vision_mode.encoder.layers):
#
#         if idx == len(model.encoder.layers) - args.num_vision_model_layers_trainable:
#             hidden_states = hidden_states.to(torch.float32).requires_grad_(True)
#
#         if model.gradient_checkpointing and model.training:
#
#             def create_custom_forward(module):
#                 def custom_forward(*inputs):
#                     return module(*inputs, output_attentions)
#
#                 return custom_forward
#
#             layer_outputs = torch.utils.checkpoint.checkpoint(
#                 create_custom_forward(encoder_layer),
#                 hidden_states,
#                 attention_mask,
#                 causal_attention_mask,
#             )
#         else:
#             layer_outputs = encoder_layer(
#                 hidden_states,
#                 attention_mask,
#                 causal_attention_mask,
#             )
#
#         hidden_states = layer_outputs[0]
#
#
#     return hidden_states


def main(args):
    accelerator = Accelerator(log_with=args.report_to if not args.disable_wandb else None,
                    project_dir=args.output_dir,
                    mixed_precision=args.mixed_precision)
    # Make one log on every process with the configuration for debugging.
    logging.basicConfig(
        format="%(asctime)s - %(levelname)s - %(name)s - %(message)s",
        datefmt="%m/%d/%Y %H:%M:%S",
        level=logging.INFO,
    )
    logger.info(accelerator.state, main_process_only=False)
    if accelerator.is_local_main_process:
        datasets.utils.logging.set_verbosity_warning()
        transformers.utils.logging.set_verbosity_info()
    else:
        datasets.utils.logging.set_verbosity_error()
        transformers.utils.logging.set_verbosity_error()

    # If passed along, set the training seed now.
    if args.seed is not None:
        set_seed(args.seed)

    accelerator.wait_for_everyone()

    weight_dtype = torch.float32
    if accelerator.mixed_precision == "fp16":
        weight_dtype = torch.float16
    elif accelerator.mixed_precision == "bf16":
        weight_dtype = torch.bfloat16

    seed = args.seed if args.seed is not None else torch.randint(10_000, size=(1,)).item()
    set_seed(seed)

    train_df = pd.read_csv(args.train_file)
    eval_df = pd.read_csv(args.validation_file)

    train_dataset = ImageDataset(args.image_folder, train_df, fit_method=args.fit_method, num_cutouts=args.num_cutouts, is_clip=True)
    eval_dataset = ImageDataset(args.image_folder, eval_df, fit_method=args.fit_method, num_cutouts=args.num_cutouts, is_clip=True)

    # Load pretrained model
    model = CLIPVisionModelWithProjection.from_pretrained(args.model_name_or_path).to(accelerator.device).to(weight_dtype)

    classifier = Network(input_dim=1024, hidden_dim=args.hidden_dim, act_fn=args.act_fn, num_transformer_layers=args.num_transformer_layers)
    criterion = nn.MSELoss()

    def collate_fn(examples):
        keys = examples[0].keys()
        collated = {k: torch.stack([example[k] for example in examples]) for k in keys}
        return collated

    train_dataloader = DataLoader(
        train_dataset, shuffle=True, collate_fn=collate_fn, batch_size=args.per_device_train_batch_size,
        num_workers=args.num_workers
    )
    eval_dataloader = DataLoader(eval_dataset, shuffle=True, collate_fn=collate_fn,
                                 num_workers=args.num_workers,
                                 batch_size=args.per_device_eval_batch_size)

    # Optimizer
    # Split weights in two groups, one with weight decay and the other not.

    # no_decay = ["bias", "LayerNorm.weight"]
    # optimizer_grouped_parameters = [
    #     {
    #         "params": [p for n, p in classifier.named_parameters() if not any(nd in n for nd in no_decay)],
    #         "weight_decay": args.weight_decay,
    #     },
    #     {
    #         "params": [p for n, p in classifier.named_parameters() if any(nd in n for nd in no_decay)],
    #         "weight_decay": 0.0,
    #     },
    # ]

    trainable_params = [*classifier.parameters()]

    model.requires_grad_(False)
    classifier.requires_grad_(True)

    optimizer = torch.optim.AdamW(trainable_params, lr=args.learning_rate)

    # Scheduler and math around the number of training steps.
    overrode_max_train_steps = False
    num_update_steps_per_epoch = math.ceil(len(train_dataloader) / args.gradient_accumulation_steps)
    if args.max_train_steps is None:
        args.max_train_steps = args.num_train_epochs * num_update_steps_per_epoch
        overrode_max_train_steps = True

    lr_scheduler = get_scheduler(
        name=args.lr_scheduler_type,
        optimizer=optimizer,
        num_warmup_steps=args.num_warmup_steps,
        num_training_steps=args.max_train_steps,
    )

    if args.gradient_checkpointing:
        classifier.enable_gradient_checkpointing()

    # Prepare everything with our `accelerator`.
    model, optimizer, classifier, train_dataloader, eval_dataloader, lr_scheduler = accelerator.prepare(
        model, optimizer, classifier, train_dataloader, eval_dataloader, lr_scheduler
    )

    # We need to recalculate our total training steps as the size of the training dataloader may have changed
    num_update_steps_per_epoch = math.ceil(len(train_dataloader) / args.gradient_accumulation_steps)
    if overrode_max_train_steps:
        args.max_train_steps = args.num_train_epochs * num_update_steps_per_epoch
    # Afterwards we recalculate our number of training epochs
    args.num_train_epochs = math.ceil(args.max_train_steps / num_update_steps_per_epoch)

    # Figure out how many steps we should save the Accelerator states
    checkpointing_steps = args.checkpointing_steps
    if checkpointing_steps is not None and checkpointing_steps.isdigit():
        checkpointing_steps = int(checkpointing_steps)

    # We need to initialize the trackers we use, and also store our configuration.
    # The trackers initializes automatically on the main process.
    if args.with_tracking and not args.disable_wandb:
        experiment_config = vars(args)
        # TensorBoard cannot log Enums, need the raw value
        accelerator.init_trackers("emotion-clip", experiment_config)

    # Train!
    total_batch_size = args.per_device_train_batch_size * accelerator.num_processes * args.gradient_accumulation_steps

    logger.info("***** Running training *****")
    logger.info(f"  Num examples = {len(train_dataset)}")
    logger.info(f"  Num Epochs = {args.num_train_epochs}")
    logger.info(f"  Instantaneous batch size per device = {args.per_device_train_batch_size}")
    logger.info(f"  Total train batch size (w. parallel, distributed & accumulation) = {total_batch_size}")
    logger.info(f"  Gradient Accumulation steps = {args.gradient_accumulation_steps}")
    logger.info(f"  Total optimization steps = {args.max_train_steps}")
    # Only show the progress bar once on each machine.
    progress_bar = tqdm(range(args.max_train_steps), disable=not accelerator.is_local_main_process)
    completed_steps = 0
    starting_epoch = 0
    evals_done_so_far = 0
    grad_norm = 0
    # Potentially load in the weights and states from a previous save
    if args.resume_from_checkpoint:
        if args.resume_from_checkpoint is not None or args.resume_from_checkpoint != "":
            accelerator.print(f"Resumed from checkpoint: {args.resume_from_checkpoint}")
            accelerator.load_state(args.resume_from_checkpoint)
            path = os.path.basename(args.resume_from_checkpoint)
        else:
            # Get the most recent checkpoint
            dirs = [f.name for f in os.scandir(os.getcwd()) if f.is_dir()]
            dirs.sort(key=os.path.getctime)
            path = dirs[-1]  # Sorts folders by date modified, most recent checkpoint is the last
        # Extract `epoch_{i}` or `step_{i}`
        training_difference = os.path.splitext(path)[0]

        if "epoch" in training_difference:
            starting_epoch = int(training_difference.replace("epoch_", "")) + 1
            resume_step = None
        else:
            resume_step = int(training_difference.replace("step_", ""))
            starting_epoch = resume_step // len(train_dataloader)
            resume_step -= starting_epoch * len(train_dataloader)

    for epoch in range(starting_epoch, args.num_train_epochs):
        model.eval()
        classifier.train()
        total_loss = 0
        for step, batch in enumerate(train_dataloader):
            model.eval()
            classifier.train()
            # We need to skip steps until we reach the resumed step
            if args.resume_from_checkpoint and epoch == starting_epoch:
                if resume_step is not None and step < resume_step:
                    completed_steps += 1
                    continue

            pixel_values = batch['pixel_values'].cuda()
            score = batch['score'].cuda()

            hidden_states = model(pixel_values, output_hidden_states=True).hidden_states[-2]

            #valence_pred, arousal_pred, dominance_pred, dominance_2_pred = classifier(hidden_states).squeeze()
            valence_pred, arousal_pred = classifier(hidden_states)

            valence_loss = criterion(valence_pred.float(), score[:, 0:1].float())
            arousal_loss = criterion(arousal_pred.float(), score[:, 1:2].float())
            # dominance_loss = criterion(dominance_pred.float(), score[:, 2:3].float())
            # dominance_2_loss = criterion(dominance_2_pred.float(), score[:, 3:4].float())

            #loss = (valence_loss + arousal_loss + dominance_loss + dominance_2_loss) / 4
            loss = (valence_loss + arousal_loss) / 2

            loss = loss.mean()

            # We keep track of the loss at each epoch
            if args.with_tracking:
                total_loss += loss.detach().float()
            loss = loss / args.gradient_accumulation_steps
            accelerator.backward(loss)
            if accelerator.sync_gradients:
                grad_norm = accelerator.clip_grad_norm_(classifier.parameters(), args.max_grad_norm)
            if step % args.gradient_accumulation_steps == 0 or step == len(train_dataloader) - 1:

                optimizer.step()
                lr_scheduler.step()
                optimizer.zero_grad()
                progress_bar.update(1)
                completed_steps += 1

                accelerator.log(
                    {"train_loss": loss.detach().float().item(),
                     "epoch": epoch,
                     "total_norm": grad_norm,
                     "valence_loss": valence_loss.mean().item(),
                        "arousal_loss": arousal_loss.mean().item(),
                        # "dominance_loss": dominance_loss.mean().item(),
                        # "dominance_2_loss": dominance_2_loss.mean().item(),
                     },
                    step=completed_steps, )
                accelerator.log({"lr": lr_scheduler.get_last_lr()[0]}, step=completed_steps)

            if completed_steps % args.eval_steps == 0:
                model.eval()
                for evl_step, evl_batch in enumerate(eval_dataloader):
                    batch = evl_batch
                    if evl_step == evals_done_so_far:
                        break
                with torch.no_grad():

                    pixel_values = batch['pixel_values'].cuda()
                    score = batch['score'].cuda()

                    hidden_states = model(pixel_values, output_hidden_states=True).hidden_states[-2]

                    # valence_pred, arousal_pred, dominance_pred, dominance_2_pred = classifier(hidden_states).squeeze()
                    valence_pred, arousal_pred = classifier(hidden_states)

                    valence_loss = criterion(valence_pred.float(), score[:, 0:1].float())
                    arousal_loss = criterion(arousal_pred.float(), score[:, 1:2].float())
                    # dominance_loss = criterion(dominance_pred.float(), score[:, 2:3].float())
                    # dominance_2_loss = criterion(dominance_2_pred.float(), score[:, 3:4].float())

                    # loss = (valence_loss + arousal_loss + dominance_loss + dominance_2_loss) / 4
                    eval_loss = (valence_loss + arousal_loss) / 2

                    eval_loss = eval_loss.mean().item()

                    #binary_log_pr_and_roc(batch["labels"].long().detach().cpu().numpy(),logits.float().detach().cpu().numpy(), accelerator, completed_steps)
                    evals_done_so_far += 1

                if args.with_tracking and not args.disable_wandb:
                    accelerator.log(
                        {
                            "eval_loss": eval_loss,
                            "eval_valence_loss": valence_loss.mean().item(),
                            "eval_arousal_loss": arousal_loss.mean().item(),
                            # "eval_dominance_loss": dominance_loss.mean().item(),
                            # "eval_dominance_2_loss": dominance_2_loss.mean().item(),
                        },
                        step=completed_steps,
                    )
                model.train()

            if completed_steps % args.save_every == 0 and completed_steps > 1:
                state_dict = {}
                classifier = accelerator.unwrap_model(classifier)
                state_dict["classifier"] = classifier.state_dict()
                state_dict["act_fn"] = args.act_fn
                state_dict["num_cutouts"] = args.num_cutouts
                state_dict["fit_method"] = args.fit_method
                state_dict["base_model_name"] = args.model_name_or_path
                os.makedirs(args.output_dir, exist_ok=True)
                torch.save(state_dict, os.path.join(args.output_dir, f"classifier_{completed_steps}.pt"))
                print("saved classifier, completed steps:", completed_steps)


            if completed_steps >= args.max_train_steps:
                break

    if args.with_tracking:
        accelerator.end_training()

    accelerator.wait_for_everyone()
    state_dict = {}
    classifier = accelerator.unwrap_model(classifier)
    state_dict["classifier"] = classifier.state_dict()
    state_dict["act_fn"] = args.act_fn
    state_dict["num_cutouts"] = args.num_cutouts
    state_dict["fit_method"] = args.fit_method
    state_dict["base_model_name"] = args.model_name_or_path
    os.makedirs(args.output_dir, exist_ok=True)
    torch.save(state_dict, os.path.join(args.output_dir, f"classifier_{completed_steps}.pt"))


if __name__ == "__main__":
    args = SimpleNamespace(**default_args)
    main(args)