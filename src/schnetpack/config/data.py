import numpy as np
from sacred import Ingredient

import schnetpack.environment as env
import schnetpack.data as dat

from .datasets import dataset_ingredient, load_data

data_ingredient = Ingredient('data', ingredients=[dataset_ingredient])


@data_ingredient.config
def cfg():
    num_train = 0.8
    num_val = 0.1
    env_provider = 'simple'
    collect_triples = False
    env_cutoff = None
    use_atomref = True


@data_ingredient.capture
def get_environment_provider(env_provider, env_cutoff):
    if env_provider == 'simple':
        return env.SimpleEnvironmentProvider()
    if env_provider == 'ase':
        return env.ASEEnvironmentProvider(env_cutoff)
    else:
        raise ValueError(
            'Unknown environment provider: ' + env_provider)


@data_ingredient.capture
def create_split(_seed, num_train, num_val, data):
    np.random.seed(_seed)

    if num_train + num_val < 1:
        num_train = int(num_train * len(data))
        num_val = int(num_val * len(data))

    return data.create_splits(num_train, num_val)


@data_ingredient.capture
def load(collect_triples):
    required_properties = ['energy_U0', 'energy_U']

    data = load_data(
        environment_provider=get_environment_provider(),
        collect_triples=collect_triples,
        required_properties=required_properties
    )

    return data


@data_ingredient.capture
def load_splits():
    data = load()
    train, val, test = create_split(data=data)
    return train, val, test


@data_ingredient.capture
def stats(properties, use_atomref, data=None, per_atom=True):
    if data is None:
        data = load()

    if use_atomref:
        atomrefs = [data.get_atomref(p) for p in prop]
    else:
        atomrefs = None

    data_loader = dat.AtomsLoader(data)
    mean, stddev = data_loader.get_statistics(properties, per_atom, atomrefs)
    return mean, stddev


