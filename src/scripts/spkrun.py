from sacred import Experiment

from schnetpack.config import data_ingredient

ex = Experiment("schnetpack", ingredients=[
    data_ingredient
])


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
