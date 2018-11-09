from sacred import Experiment
from schnetpack.datasets import dataset_ingredient


ex = Experiment("schnetpack", ingredients=[dataset_ingredient])


@ex.config
def cfg():
    model = "schnet"


@ex.command
def download():
    print("Load data")


@ex.command
def train():
    print("Train")


@ex.command
def evaluate():
    print("Evaluate")


@ex.automain
def main():
    print(ex.config)
