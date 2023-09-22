import bitsandbytes as bnb
from torch import nn
from transformers.trainer_pt_utils import get_parameter_names

def create_optimizer(training_args, model):
    """
    Create an optimizer based on the specified training arguments and model parameters.

    Args:
        training_args (TrainingArguments): The training arguments object containing hyperparameters for training.
        model (nn.Module): The neural network model to be optimized.

    Returns:
        An optimizer instance that can be used for training the model.
    """
    # Get the names of all layer normalization parameters
    decay_parameters = get_parameter_names(model, [nn.LayerNorm])
    decay_parameters = [name for name in decay_parameters if "bias" not in name]

    # Define the optimizer parameters
    optimizer_grouped_parameters = [
        {
            "params": [p for n, p in model.named_parameters() if n in decay_parameters],
            "weight_decay": training_args.weight_decay,
        },
        {
            "params": [p for n, p in model.named_parameters() if n not in decay_parameters],
            "weight_decay": 0.0,
        },
    ]

    # Create the optimizer using the AdamW algorithm with 8-bit precision
    optimizer_kwargs = {
        "betas": (training_args.adam_beta1, training_args.adam_beta2),  
        "eps": training_args.adam_epsilon,
    }
    optimizer_kwargs["lr"] = training_args.learning_rate
    adam_bnb_optim = bnb.optim.Adam8bit(
        optimizer_grouped_parameters,
        betas=(training_args.adam_beta1, training_args.adam_beta2),
        eps=training_args.adam_epsilon,
        lr=training_args.learning_rate,
    )

    return adam_bnb_optim
