import os

from sacred import Ingredient

import schnetpack.data as dat
import schnetpack.datasets as dset

dataset_ingredient = Ingredient('dataset')


@dataset_ingredient.config
def cfg():
    name = None  # name of the dataset
    path = None  # path to ASE db (destination)
    property_map = {}


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
        raise dat.AtomsDataError('Unknown dataset: ' + name)


@dataset_ingredient.capture
def load_data(path, environment_provider, collect_triples,
              required_properties):
    if not os.path.exists(path):
        download()

    data = dat.AtomsData(path, required_properties=required_properties,
                         collect_triples=collect_triples,
                         environment_provider=environment_provider)
    print(data)
    return data


@dataset_ingredient.capture
def download_qm9(name, path, remove_uncharacterized):
    dset.QM9(path, download=True, properties=dset.QM9.properties,
             remove_uncharacterized=remove_uncharacterized)


@dataset_ingredient.capture
def download_md17(path, molecule):
    dset.MD17(path, molecule, download=True,
              properties=dset.MD17.required_properties)
