from sacred import Experiment

import schnetpack.experiment as exp

ex = Experiment("schnetpack", ingredients=[
    exp.data_ingredient
])


@ex.config
def cfg():
    model = "schnet"


@ex.command
def download():
    print("Load data")


@ex.command
def train(_log):
    _log.info("Load data")
    #exp.load()



@ex.command
def evaluate():
    print("Evaluate")


@ex.automain
def main():
    print(ex.config)
