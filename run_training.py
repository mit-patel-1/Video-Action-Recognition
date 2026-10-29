"""Run training with the same model, splits, and logging as run.py."""

from run import args_parser as parse_args, main


def args_parser():
    """Read training arguments without requiring a mode option."""
    return parse_args(training_only=True)


def trainer(args):
    """Use the shared training entry point."""
    args.mode = 'train'
    return main(args)


if __name__ == '__main__':
    trainer(args_parser())
