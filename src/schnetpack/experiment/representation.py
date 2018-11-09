from sacred import Ingredient

import schnetpack.nn
from schnetpack.representation import SchNet, BehlerSFBlock

representation_ingredient = Ingredient('representation')


class RepresentationError(Exception):
    pass


@representation_ingredient.config
def cfg():
    name = None  # Name of representation


@representation_ingredient.named_config
def schnet():
    name = 'schnet'

    n_atom_basis = 128
    n_filters = 128
    n_interactions = 6
    cutoff = 5.0
    n_gaussians = 25
    normalize_filter = False
    coupled_interactions = False
    return_intermediate = False
    max_z = 100
    cutoff_network = 'hard'
    trainable_gaussians = False
    distance_expansion = None


@representation_ingredient.capture
def construct_schnet(name, n_atom_basis, n_filters, n_interactions, cutoff, n_gaussians, normalize_filter,
                     coupled_interactions, return_intermediate, max_z, cutoff_network, trainable_gaussians,
                     distance_expansion):
    if cutoff_network == 'hard':
        cutoff_function = schnetpack.nn.HardCutoff
    elif cutoff_network == 'cosine':
        cutoff_function = schnetpack.nn.CosineCutoff
    elif cutoff_network == 'mollifier':
        cutoff_function = schnetpack.nn.MollifierCutoff
    else:
        raise RepresentationError('Unrecognized cutoff {:s}'.format(cutoff_network))

    return SchNet(
        n_atom_basis=n_atom_basis,
        n_filters=n_filters,
        n_interactions=n_interactions,
        cutoff=cutoff,
        n_gaussians=n_gaussians,
        normalize_filter=normalize_filter,
        coupled_interactions=coupled_interactions,
        return_intermediate=return_intermediate,
        max_z=max_z,
        cutoff_network=cutoff_function,
        trainable_gaussians=trainable_gaussians,
        distance_expansion=distance_expansion
    )


@representation_ingredient.named_config
def acsf():
    name = 'acsf'

    n_radial = 22
    n_angular = 5
    zetas = [1]
    cutoff_radius = 5.0
    elements = [1, 6, 7, 8, 9]
    centered = False
    crossterms = False
    mode = 'weighted'


@representation_ingredient.capture
def construct_acsf(name, n_radial, n_angular, zetas, cutoff_radius, elements, centered, crossterms, mode):
    return BehlerSFBlock(
        n_radial=n_radial,
        n_angular=n_angular,
        zetas=zetas,
        cutoff_radius=cutoff_radius,
        elements=elements,
        centered=centered,
        crossterms=crossterms,
        mode=mode
    )


@representation_ingredient.command
def construct(name):
    if name == 'schnet':
        return construct_schnet()
    elif name == 'acsf':
        return construct_acsf()
    else:
        raise RepresentationError('Unknown representation {:s}'.format(name))