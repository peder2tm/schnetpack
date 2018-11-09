from sacred import Ingredient

import schnetpack.environment as env
from .datasets import dataset_ingredient, load_data

data_ingredient = Ingredient('data', ingredients=[dataset_ingredient])


@data_ingredient.config
def cfg():
    data_splits = {}
    split_file = None
    environment_provider = 'simple'
    collect_triples = False
    cutoff = None


@data_ingredient.capture
def get_environment_provider(environment_provider, cutoff):
    if environment_provider == 'simple':
        return env.SimpleEnvironmentProvider()
    if environment_provider == 'ase':
        return env.ASEEnvironmentProvider(cutoff)
    else:
        raise ValueError(
            'Unknown environment provider: ' + environment_provider)


@data_ingredient.command
def load(collect_triples):
    required_properties = []

    data = load_data(
        environment_provider=get_environment_provider(),
        collect_triples=collect_triples,
        required_properties=required_properties
    )
    return data
