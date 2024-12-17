import torch
import torch.nn as nn
import numpy as np

from sine import SineLayer
from abs_layer import AbsLayer

class Siren(nn.Module):
    def __init__(self, in_features, hidden_features, hidden_layers, out_features, outermost_linear=False, outermost_activation=AbsLayer(),
                 first_omega_0=30, hidden_omega_0=30.):

        super().__init__()

        self.in_features = in_features
        self.net = []
        self.net.append(SineLayer(in_features, hidden_features,
                                  is_first=True, omega_0=first_omega_0))

        for i in range(hidden_layers):
            self.net.append(SineLayer(hidden_features, hidden_features,
                                      is_first=False, omega_0=hidden_omega_0))

        if outermost_linear:
            final_linear = nn.Linear(hidden_features, out_features)

            with torch.no_grad():
                final_linear.weight.uniform_(-np.sqrt(6 / hidden_features) / hidden_omega_0,
                                             np.sqrt(6 / hidden_features) / hidden_omega_0)

            self.net.append(final_linear)
            self.net.append(outermost_activation)
        else:
            self.net.append(SineLayer(hidden_features, out_features,
                                      is_first=False, omega_0=hidden_omega_0))

        self.net = nn.Sequential(*self.net)

    def forward(self, coords):
        coords = coords.clone().detach().requires_grad_(
            True)  # allows to take derivative w.r.t. input
        # We must force the putput to be positive as it represents a density.
        output = self.net(coords)
        return output

    def forward_with_activations(self, coords, retain_grad=False):
        '''Returns not only model output, but also intermediate activations.
        Only used for visualizing activations later!'''
        activations = OrderedDict()

        activation_count = 0
        x = coords.clone().detach().requires_grad_(True)
        activations['input'] = x
        for i, layer in enumerate(self.net):
            if isinstance(layer, SineLayer):
                x, intermed = layer.forward_with_intermediate(x)

                if retain_grad:
                    x.retain_grad()
                    intermed.retain_grad()

                activations['_'.join(
                    (str(layer.__class__), "%d" % activation_count))] = intermed
                activation_count += 1
            else:
                x = layer(x)

                if retain_grad:
                    x.retain_grad()

            activations['_'.join(
                (str(layer.__class__), "%d" % activation_count))] = x
            activation_count += 1

        return activations


if __name__ == '__main__':
    encoding_dim = 3
    n_neurons = 100
    hidden_layers = 9
    activation = AbsLayer()
    siren_omega = 30.0


    model = Siren(in_features=encoding_dim, out_features=1, hidden_features=n_neurons,
                     hidden_layers=hidden_layers, outermost_linear=True, outermost_activation=activation,
                     first_omega_0=siren_omega, hidden_omega_0=siren_omega)
    qq = 0