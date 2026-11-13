"""Train the VIIRS FRP model and estimate intensity for the bundled FIRMS data."""
from train_model import train
from run_prediction import run_inference


def main():
    train()
    run_inference()


if __name__ == "__main__":
    main()
