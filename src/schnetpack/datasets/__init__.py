r"""
Classes wrapping various standard benchmark datasets.
"""
import os
from sacred import Ingredient

from .ani1 import *
from .iso17 import *
from .matproj import *
from .md17 import *
from .qm9 import *
from ..data import AtomsData, AtomsDataError

dataset_ingredient = Ingredient('dataset')


@dataset_ingredient.config
def cfg():
    name = None  # name of the dataset
    path = None  # path to ASE db (destination)


@dataset_ingredient.named_config
def qm9():
    name = 'qm9'
    remove_uncharacterized = True


@dataset_ingredient.named_config
def md17():
    name = 'md17'
    molecule = None  # molecule of the MD17 collection to be loaded


@dataset_ingredient.command
def download(name):
    if name == 'qm9':
        download_qm9()
    elif name == 'md17':
        download_md17()
    else:
        raise AtomsDataError('Unknown dataset: ' + name)


@dataset_ingredient.command
def load(path):
    if not os.path.exists(path):
        download()

    # TODO
    collect_triples = False
    required_properties = []
    environment_provider = None
    data = AtomsData(path, required_properties=required_properties,
                     collect_triples=collect_triples)
    print(data)
    return data


@dataset_ingredient.capture
def download_qm9(name, path, remove_uncharacterized):
    data = QM9(path, download=True,
               remove_uncharacterized=remove_uncharacterized)


@dataset_ingredient.capture
def download_md17(path, molecule):
    data = MD17(path, molecule, download=True)
