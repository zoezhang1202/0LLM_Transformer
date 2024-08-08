"""
Quaternion Library for PyTorch
Some utils for Quaternion functions and layers.

Code based on https://github.com/vanzytay/QuaternionTransformers/blob/master/layers/qlib.py

References:
https://arxiv.org/pdf/1906.04393
https://arxiv.org/pdf/1806.04418.pdf
https://arxiv.org/pdf/1806.07789.pdf
https://github.com/Orkis-Research/light-Recurrent-Neural-Networks
https://github.com/Orkis-Research/light-Convolutional-Neural-Networks-for-End-to-End-Automatic-Speech-Recognition

"""

import torch
import torch.nn as nn


class QuaternionFFN(nn.Module):
    def __init__(self, input_dim, output_dim, activation=None):
        super(QuaternionFFN, self).__init__()
        # Set input and output dimensions
        self.input_dim = (
            input_dim  # x is [bsz x features] tensor -- input dimension  == features
        )
        self.output_dim = output_dim
        self.activation = activation

        # Initialize quaternion kernel weights using Xavier initialization
        self.kernel = nn.Parameter(torch.Tensor(input_dim // 4, output_dim))
        nn.init.xavier_uniform_(self.kernel)

    def forward(self, x):
        # Compute Hamilton product
        output = hamilton_product(x, self.kernel)

        # Apply activation function if provided
        if self.activation:
            output = self.activation(output)

        return output


class QuaternionFFN3D(nn.Module):
    def __init__(self, input_dim, output_dim, activation=None):
        super(QuaternionFFN3D, self).__init__()
        self.input_dim = input_dim
        self.output_dim = output_dim
        self.activation = activation

        # Define quaternion feed-forward layer
        self.quaternion_ffn = QuaternionFFN(input_dim, output_dim, activation)

    def forward(self, x):
        # Get the input shape dimensions
        bsz, seq_len, _d = x.size()

        # Reshape input tensor to [bsz*seq_len, _d]
        x = x.view(-1, _d)

        # Apply quaternion feed-forward layer
        x = self.quaternion_ffn(x)

        # Reshape output tensor to [bsz, seq_len, dim]
        x = x.view(bsz, seq_len, self.output_dim)

        return x


def make_quaternion_mul(kernel, concat_dim=0):
    r, i, j, k = torch.chunk(kernel, 4, dim=-1)
    r2 = torch.cat([r, -i, -j, -k], dim=-1)  # 0, 1, 2, 3
    i2 = torch.cat([i, r, -k, j], dim=-1)  # 1, 0, 3, 2
    j2 = torch.cat([j, k, r, -i], dim=-1)  # 2, 3, 0, 1
    k2 = torch.cat([k, -j, i, r], dim=-1)  # 3, 2, 1, 0
    hamilton = torch.cat([r2, i2, j2, k2], dim=concat_dim)
    return hamilton


def hamilton_product(x, kernel):
    h = make_quaternion_mul(kernel)
    output = torch.matmul(x, h)
    return output


def get_r(x, a=1):
    return torch.chunk(x, 4, dim=a)[0]


def get_i(x, a=1):
    return torch.chunk(x, 4, dim=a)[1]


def get_j(x, a=1):
    return torch.chunk(x, 4, dim=a)[2]


def get_k(x, a=1):
    return torch.chunk(x, 4, dim=a)[3]


def quaternion_attention(a, b):
    """
    Performs dot product attention between two quaternion sequences.

    a: bsz x al x dim
    b: bsz x bl x dim

    The output is one attention matrix for each component (r, i, j, k).
    """
    ar, ax, ay, az = torch.chunk(a, 4, dim=-1)
    br, bx, by, bz = torch.chunk(b, 4, dim=-1)

    r = (
        torch.matmul(ar, br.transpose(1, 2))
        - torch.matmul(ax, bx.transpose(1, 2))
        - torch.matmul(ay, by.transpose(1, 2))
        - torch.matmul(az, bz.transpose(1, 2))
    )
    i = (
        torch.matmul(ar, bx.transpose(1, 2))
        + torch.matmul(ax, br.transpose(1, 2))
        + torch.matmul(ay, bz.transpose(1, 2))
        - torch.matmul(az, by.transpose(1, 2))
    )
    j = (
        torch.matmul(ar, by.transpose(1, 2))
        - torch.matmul(ax, bz.transpose(1, 2))
        + torch.matmul(ay, br.transpose(1, 2))
        + torch.matmul(az, bx.transpose(1, 2))
    )
    k = (
        torch.matmul(ar, bz.transpose(1, 2))
        + torch.matmul(ax, by.transpose(1, 2))
        - torch.matmul(ay, bx.transpose(1, 2))
        + torch.matmul(az, br.transpose(1, 2))
    )

    return [r, i, j, k]


def quaternion_dot(q0, q1):
    """Quaternion product between 2 quaternions

    Returns the same shape and acts like element-wise quaternion multiplication.
    """
    q1_r = get_r(q1)
    q1_i = get_i(q1)
    q1_j = get_j(q1)
    q1_k = get_k(q1)

    r_base = torch.mul(q0, q1)
    r = get_r(r_base) - get_i(r_base) - get_j(r_base) - get_k(r_base)

    i_base = torch.mul(q0, torch.cat([q1_i, q1_r, q1_k, q1_j], dim=1))
    i = get_r(i_base) + get_i(i_base) + get_j(i_base) - get_k(i_base)

    j_base = torch.mul(q0, torch.cat([q1_j, q1_k, q1_r, q1_i], dim=1))
    j = get_r(j_base) - get_i(j_base) + get_j(j_base) + get_k(j_base)

    k_base = torch.mul(q0, torch.cat([q1_k, q1_j, q1_i, q1_r], dim=1))
    k = get_r(k_base) + get_i(k_base) - get_j(k_base) + get_k(k_base)

    return torch.cat([r, i, j, k], dim=1)


def quarternion_dot_product_att(a, b):
    """Wrapper for two sequences"""
    bsz, al, d = a.shape
    bl = b.shape[1]

    # Reshape and tile tensors
    a = a.view(-1, d).repeat(bl, 1)
    b = b.view(-1, d).repeat(al, 1)

    # Compute quaternion dot product
    att = quaternion_dot(a, b)

    # Reshape the result
    att = att.view(bsz, -1, al * bl)

    # Sum along the second dimension
    att = torch.sum(att, dim=1)

    return att.view(-1, al * bl)


def quaternion_dot_3d(q0, q1):
    _, sq, d = q0.shape
    q0 = q0.view(-1, d)
    q1 = q1.view(-1, d)
    out = quaternion_dot(q0, q1)
    return out.view(-1, sq, d)


def quaternion_concat(x, axis):
    """Helpful if we have 2 quaternions in [r,i,j,k].
    We can't simply concat them as it would mess the components.
    So in this case, we extract each component and concat them individually.
    """
    output = [[] for _ in range(4)]
    for _x in x:
        sp = torch.chunk(_x, 4, dim=axis)
        for i in range(4):
            output[i].append(sp[i])

    final = []
    for o in output:
        o = torch.cat(o, dim=axis)
        final.append(o)

    return torch.cat(final, dim=axis)
