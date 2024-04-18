import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2' 
from tensorflow import keras
import numpy as np
import sympy as sp
import tensorflow as tf
import time
import itertools
import MLGeometry as mlg
import sympy as sp
import argparse

parser = argparse.ArgumentParser()

parser.add_argument('--manifold', default='cicy1')
parser.add_argument('--max_epochs', type=int, default=8000)
parser.add_argument('--layers', default='64_256_1024_15')
parser.add_argument('--batch_size', type=int)
parser.add_argument('--load_path')
parser.add_argument('--save_path')
args = parser.parse_args()

np.random.seed(1024)
tf.random.set_seed(1024)

cicy = args.manifold
max_epochs = args.max_epochs
layer = args.layers
batch_size = args.batch_size
load_path = args.load_path
model_save_path = args.save_path

train_set_path = 'dataset/dg_' + cicy + '_100000_train'
test_set_path = 'dataset/dg_' + cicy + '_10000_train'

if model_save_path is None:
    model_save_path = 'trained_models_one_form/' + cicy + '/' + layer

CY_model = tf.keras.models.load_model('trained_models/' + cicy + '/64_256_1024_1', compile=False)

Z = sp.var('z0:6')

if cicy == 'cicy1':
    f1 = -z0**4 + z1**4 + z2**4 + z3**4 + 1/4*z4**4
    f2 = -z0**2  + 1/4*z1**2 + 1/4 * z2**2 + z4**2 + z5**2
    f = [f1, f2]

elif cicy == 'cicy2':
    f1 = -z0**4 + z1**4 + z2**4 + z3**4 + 1/10*z4**4 + 1/10*z5**4
    f2 = -z0**2 + z4**2 + z5**2 + 1/100*z1**2 + 1/100*z2**2 + 1/100*z3**2

    f = [f1, f2]

def df_tf(z):
    if cicy == 'cicy1':
        df_tf = [-4*z[0]**3, 4*z[1]**3, 4*z[2]**3, 4*z[3]**3, z[4]**3, 0]
    elif cicy == 'cicy2':
        df_tf = [-4*z[0]**3, 4*z[1]**3, 4*z[2]**3, 4*z[3]**3, 0.4*z[4]**3, 0.4*z[5]**3]
    return df_tf

def dg_tf(z):
    if cicy == 'cicy1':
        dg_tf = [-2*z[0], 0.5*z[1], 0, 0.5*z[2], 2*z[4], 2*z[5]]
    elif cicy == 'cicy2':
        dg_tf = [-2*z[0], 0.02*z[1], 0.02*z[2], 0.02*z[3], 2*z[4], 2*z[5]]
    return dg_tf

'''
# Deprecated
def dfg_tf(z):
    if cicy == 'cicy1':
        dfg_row0 = [0, -2.0*z[0]**3*z[1] + 8*z[0]*z[1]**3, -2.0*z[0]**3*z[2] + 8*z[0]*z[2]**3, 8*z[0]*z[3]**3, -8*z[0]**3*z[4] + 2.0*z[0]*z[4]**3, -8*z[0]**3*z[5]]
        dfg_row1 = [2.0*z[0]**3*z[1] - 8*z[0]*z[1]**3, 0, 2.0*z[1]**3*z[2] - 2.0*z[1]*z[2]**3, -2.0*z[1]*z[3]**3, 8*z[1]**3*z[4] - 0.5*z[1]*z[4]**3, 8*z[1]**3*z[5]] 
        dfg_row2 = [2.0*z[0]**3*z[2] - 8*z[0]*z[2]**3, -2.0*z[1]**3*z[2] + 2.0*z[1]*z[2]**3, 0, -2.0*z[2]*z[3]**3, 8*z[2]**3*z[4] - 0.5*z[2]*z[4]**3, 8*z[2]**3*z[5]]
        dfg_row3 = [-8*z[0]*z[3]**3, 2.0*z[1]*z[3]**3, 2.0*z[2]*z[3]**3, 0, 8*z[3]**3*z[4], 8*z[3]**3*z[5]]
        dfg_row4 = [8*z[0]**3*z[4] - 2.0*z[0]*z[4]**3, -8*z[1]**3*z[4] + 0.5*z[1]*z[4]**3, -8*z[2]**3*z[4] + 0.5*z[2]*z[4]**3, -8*z[3]**3*z[4], 0, 2.0*z[4]**3*z[5]]
        dfg_row5 = [8*z[0]**3*z[5], -8*z[1]**3*z[5], -8*z[2]**3*z[5], -8*z[3]**3*z[5], -2.0*z[4]**3*z[5], 0]

    elif cicy == 'cicy2':
        dfg_row0 = [0, -0.08*z[0]**3*z[1] + 8*z[0]*z[1]**3, -0.08*z[0]**3*z[2] + 8*z[0]*z[2]**3, -0.08*z[0]**3*z[3] + 8*z[0]*z[3]**3, -8*z[0]**3*z[4] + 0.8*z[0]*z[4]**3, -8*z[0]**3*z[5] + 0.8*z[0]*z[5]**3]
        dfg_row1 = [0.08*z[0]**3*z[1] - 8*z[0]*z[1]**3, 0, 0.08*z[1]**3*z[2] - 0.08*z[1]*z[2]**3, 0.08*z[1]**3*z[3] - 0.08*z[1]*z[3]**3, 8*z[1]**3*z[4] - 0.008*z[1]*z[4]**3, 8*z[1]**3*z[5] - 0.008*z[1]*z[5]**3]
        dfg_row2 = [0.08*z[0]**3*z[2] - 8*z[0]*z[2]**3, -0.08*z[1]**3*z[2] + 0.08*z[1]*z[2]**3, 0, 0.08*z[2]**3*z[3] - 0.08*z[2]*z[3]**3, 8*z[2]**3*z[4] - 0.008*z[2]*z[4]**3, 8*z[2]**3*z[5] - 0.008*z[2]*z[5]**3]
        dfg_row3 = [0.08*z[0]**3*z[3] - 8*z[0]*z[3]**3, -0.08*z[1]**3*z[3] + 0.08*z[1]*z[3]**3, -0.08*z[2]**3*z[3] + 0.08*z[2]*z[3]**3, 0, 8*z[3]**3*z[4] - 0.008*z[3]*z[4]**3, 8*z[3]**3*z[5] - 0.008*z[3]*z[5]**3]
        dfg_row4 = [8*z[0]**3*z[4] - 0.8*z[0]*z[4]**3, -8*z[1]**3*z[4] + 0.008*z[1]*z[4]**3, -8*z[2]**3*z[4] + 0.008*z[2]*z[4]**3, -8*z[3]**3*z[4] + 0.008*z[3]*z[4]**3, 0, 0.8*z[4]**3*z[5] - 0.8*z[4]*z[5]**3]
        dfg_row5 = [8*z[0]**3*z[5] - 0.8*z[0]*z[5]**3, -8*z[1]**3*z[5] + 0.008*z[1]*z[5]**3, -8*z[2]**3*z[5] + 0.008*z[2]*z[5]**3, -8*z[3]**3*z[5] + 0.008*z[3]*z[5]**3, -0.8*z[4]**3*z[5] + 0.8*z[4]*z[5]**3, 0]

    return tf.stack([dfg_row0, dfg_row1, dfg_row2, dfg_row3, dfg_row4, dfg_row5], axis=0)
'''

def get_basis(point):
    n = tf.shape(point)[0]
    mask = tf.eye(n, dtype=point.dtype)
    basis = tf.einsum('i,jk->ijk', point, mask)
    basis_antisym = (basis - tf.transpose(basis, perm=[1, 0, 2])) / (2*tf.linalg.norm(point)**2)

    return basis_antisym

def fill_strict_upper_tri(polys):
    # Fill the outputs of the NN to a strictly upper triangular matrix
    n = 5
    # For a more general case, solve for n^2 - n = 2m:
    # m = tf.shape(polys)[-1]
    # n = tf.cast(tf.sqrt(0.25 + tf.cast(2 * m, dtype=tf.float32)), dtype=tf.int32)
    polys = tf.concat([polys, tf.reverse(polys[n:], axis=[0])],axis=0)
    polys = tf.reshape(polys, (n,n))
    upper_triangular = tf.linalg.band_part(polys, 0, -1)
    # Add one row of zeros at the end, and one column of zeros at the beginning
    upper_triangular = tf.pad(upper_triangular, ((0, 1), (1, 0)))

    return upper_triangular

def delete_columns(tensor, i, j, axis):
    mask = tf.ones(tensor.shape[-1], dtype=tf.bool)
    mask = tf.tensor_scatter_nd_update(mask, [[i], [j[0]], [j[1]]], [False]*3)
    result = tf.boolean_mask(tensor, mask, axis=axis)
    return result

'''
def get_restriction(point, const_coord, ignored_coord):

    dfg = dfg_tf(point)
    n = tf.size(point)
    restriction= tf.eye(n, dtype=point.dtype)
    indices = tf.range(n, dtype=tf.int64)[:, tf.newaxis]

    # Didn't work
    mask = tf.cast(indices == ignored_coord[0], dtype=point.dtype)
    restriction = (1 - mask) * restriction + mask * (-dfg[:,ignored_coord[1]] / dfg[ignored_coord])
    mask = tf.cast(indices == ignored_coord[1], dtype=point.dtype)
    restriction = (1 - mask) * restriction + mask * (-dfg[ignored_coord[0],:] / dfg[ignored_coord])

    # Didn't work
    replace_axis = 0
    for k in range(ignored_coord.shape[0]):
        dfg_row = dfg
        for i in range(ignored_coord.shape[0]):
            if ignored_coord[i] == ignored_coord[k]:
                replace_axis += 1 
                continue
            dfg_row = tf.gather(dfg_row, ignored_coord[i], axis=axis)
        mask = tf.cast(indices == ignored_coord[k], dtype=tf.float64)
        restriction = (1 - mask) * restriction + mask * (-dfg_row / dfg[ignored_coord])

    # Didn't work
    for i in range(ignored_coord.shape[-1]):
        ignored_coord_mut = ignored_coord[:].numpy().tolist()
        ignored_coord_mut[i] = slice(None)
        # Replace the 0-th row with -df[:,j,k,...] / df[i,j,k,...]
        # The 1-th row with -df[i,:,k,...] / df[i,j,k,...]
        # The 2-th row with -df[i,j,:,...] / df[i,j,k,...], etc
        mask = tf.cast(indices == ignored_coord[i], dtype=tf.float64)
        restriction = (1 - mask) * restriction + mask * (-df[ignored_coord_mut] / df[ignored_coord])

    restriction = delete_columns(restriction, const_coord, ignored_coord, axis=1)

    return restriction
'''
# A backup version
def get_restriction(point, const_coord, ignored_coord):

    df = df_tf(point)
    dg = dg_tf(point)
    diffs = [df, dg]

    n = tf.size(point)
    restriction= tf.eye(n, dtype=point.dtype)
    indices = tf.range(n, dtype=ignored_coord.dtype)[:, tf.newaxis]

    replace_axis = 0
    for k in range(ignored_coord.shape[0]):

        mask = tf.cast(indices == ignored_coord[k], dtype=point.dtype)
        restriction = (1 - mask) * restriction + mask * diffs[k]

    restriction = tf.linalg.inv(restriction)
    restriction = delete_columns(restriction, const_coord, ignored_coord, axis=1)

    return restriction

def get_CY_metrics(args):
    point, const_coord, ignored_coord = args
    with tf.GradientTape(persistent=True) as tape:
        tape.watch(point)
        point_c = tf.cast(point, dtype=tf.complex64)
        #dummy = tf.constant([complex(np.random.uniform(-1, 1), np.random.uniform(-1, 1)) for _ in range(6)], dtype=tf.complex64)
        dummy = tf.constant([1.0+0.0j, 0.8 + 0.7j, 0.2 - 0.33j, 0.34 + 0.5j, 0.4 + 0.15j , -0.55 + 0.87j], dtype=tf.complex64)
        point_c = tf.stack([point_c, dummy], axis=0)
        g = tf.math.real(mlg.complex_math.complex_hessian(tf.math.real(CY_model(point_c))[0], point_c)[0])
        restriction = get_restriction(point, const_coord, ignored_coord) # (5, 3)
        g = tf.einsum('ij, jk, kl', tf.transpose(restriction), g, tf.math.conj(restriction))
        #s, u, v = tf.linalg.svd(tf.reshape(g, [3,3]))
        #g_inv = tf.matmul(v, tf.matmul(tf.linalg.pinv(tf.linalg.diag(s)), u, adjoint_b=True))
        g_inv = tf.linalg.inv(g)
        sqrt_det_g = tf.sqrt(tf.linalg.det(g))
    d_g_inv = tape.jacobian(g_inv, point)
    d_sqrt_det_g = tape.jacobian(sqrt_det_g, point)
    d_g_inv = tf.einsum('ijk, kl', d_g_inv, restriction)
    d_sqrt_det_g = tf.einsum('k, kl', d_sqrt_det_g, restriction)
    return g, g_inv, d_g_inv, sqrt_det_g, d_sqrt_det_g


def loss_func(args):
    point, g, g_inv, d_g_inv, sqrt_det_g, d_sqrt_det_g, const_coord, ignored_coord = args
    # ∂(P_L*ω^k )/∂x^i 
    with tf.GradientTape() as tape:
        tape.watch(point)
        restriction = get_restriction(point, const_coord, ignored_coord) # (5, 3)
        basis = get_basis(point) # (5, 5, 5)
        omega_comp = model(tf.expand_dims(point / tf.norm(point), 0))[0]
        omega_comp = fill_strict_upper_tri(omega_comp) # (5, 5)
        # Multiply Omega_{ij} with e^{ijk}. 
        # To make the broadcasting works properly, tranpose k to the first coordinate
        # Then transpose it back after tf.multiply
        basis = tf.transpose(basis, [2, 0, 1])
        omega  = tf.multiply(omega_comp, basis)
        omega  = tf.transpose(omega, [1, 2, 0])

        omega = tf.reduce_sum(tf.einsum('ijk, kl', omega, restriction), axis=[0, 1]) # (5, 5, 3) -> (3)
        # star_omega = sqrt(tf.linalg.det(g)) * tf.einsum('i, ij, jkl', omega, g_inv, eps) / 2 # (3, 3)

    # ∂(ω^k)/∂x^i  * dx
    # Note that the derivatives appear as the last index here, 
    # which is why later on d_star_Omega[1,2,0] is the first term in d_star_omega_square 
    # Alternatively one can also transpose the tensors and move it to the first index as in the formulas.
    d_Omega = tape.jacobian(omega, point) # (3, 5)

    #d_star_Omega = (tf.einsum('m, i, ij, jkl -> klm', d_sqrt_det_g, omega, g_inv, eps) +
    #                sqrt_det_g *(tf.einsum('im, ij, jkl -> klm', d_Omega, g_inv, eps) + 
    #                             tf.einsum('i, ijm ,jkl -> klm', omega, d_g_inv, eps))) / 2 #(3, 3, 5)

    d_Omega = tf.einsum('ik, kl', d_Omega, restriction) # (3, 5) -> (3, 3)

    d_omega = d_Omega - tf.transpose(d_Omega)
    # The 1/2 factor comes from overcounting the upper / lower triangular
    # Multiplied by sqrt_det_g for the integration
    d_omega_square = (0.5*tf.einsum('ij, ik, jl, kl', d_omega, g_inv, g_inv, d_omega)) * sqrt_det_g
    d_star_Omega = (tf.einsum('m, i, im', d_sqrt_det_g, omega, g_inv) +
                    sqrt_det_g *(tf.einsum('im, im', d_Omega, g_inv) + 
                                 tf.einsum('i, imm ', omega, d_g_inv))) / 2 #(1）

    #d_star_omega_square = ((d_star_Omega[1,2,0] -
    #                        d_star_Omega[0,2,1] +
    #                        d_star_Omega[0,1,2]))**2
    # One can multiple a eps to d_star_Omega and then it can be simplied to the current form

    d_star_omega_square = 1 / sqrt_det_g * d_star_Omega**2
    loss = d_omega_square + d_star_omega_square
    omega_norm_square = tf.einsum('i, ij, j',omega, g_inv, omega) * sqrt_det_g
    return loss,  omega_norm_square, d_omega_square, d_star_omega_square

def generate_dataset(HS):
    dataset = None
    for i in range(len(HS.patches)):
        for j in range(len(HS.patches[i].patches)):
            points = np.array(np.real(HS.patches[i].patches[j].points), dtype=np.float32)
            ignored_coords = HS.patches[i].patches[j].max_grad_coordinate
            for k in range(len(ignored_coords)):
                if ignored_coords[k] >= i:
                    ignored_coords[k] += 1
            points_batch = np.array_split(points, int(len(points)/3000+1))
            for batch in points_batch:
                st = time.time()
                print("processing patch: ", i, " ", j)
                const_coords_arr = np.full(len(batch), i, dtype=np.int32)
                ignored_coords_arr = np.array([ignored_coords]*len(batch), dtype=np.int32)
                gs, g_invs, d_g_invs, sqrt_det_gs, d_sqrt_det_gs = tf.vectorized_map(get_CY_metrics, (batch, const_coords_arr, ignored_coords_arr))
                new_dataset = tf.data.Dataset.from_tensor_slices((batch, gs, g_invs, d_g_invs, sqrt_det_gs, d_sqrt_det_gs, const_coords_arr, ignored_coords_arr))
                if dataset is None:
                    dataset = new_dataset
                else:
                    dataset = dataset.concatenate(new_dataset)
                print('Time for this batch: ', time.time() - st)
    return dataset

#gs, g_invs, d_g_invs, sqrt_det_gs, d_sqrt_det_gs = tf.vectorized_map(get_CY_metrics, (points, const_coords, ignored_coords))

try:
    train_set = tf.data.Dataset.load(train_set_path)
    test_set = tf.data.Dataset.load(test_set_path)
    print('Loaded train sets at ' + train_set_path)
    print('Loaded test sets at ' + test_set_path)
except:
    HS_train = mlg.cicyhypersurface.RealCICYHypersurface(Z, f, 100000)
    HS_test = mlg.cicyhypersurface.RealCICYHypersurface(Z, f, 10000)
    train_set = generate_dataset(HS_train)
    test_set = generate_dataset(HS_test)
    if train_set_path is not None:
        tf.data.Dataset.save(train_set, train_set_path)
        print('Datasets saved at ' + train_set_path)

    if test_set_path is not None:
        tf.data.Dataset.save(test_set, test_set_path)
        print('Datasets saved at ' + test_set_path)

try:
    model = tf.keras.models.load_model(load_path, compile=False)
    print('Loaded model from ', load_path)
except:
    print('Creating a new model')
    n_units = layer.split('_')
    n_units = [int(n_unit) for n_unit in n_units]
    model = tf.keras.Sequential([
      tf.keras.layers.Dense(n_units[0], activation=tf.square, input_dim=6),
      tf.keras.layers.Dense(n_units[1], activation=tf.square),
      tf.keras.layers.Dense(n_units[2], activation=tf.square),
      tf.keras.layers.Dense(n_units[3])])

print('start optimizing')
n_points_train = tf.data.Dataset.cardinality(train_set).numpy()
n_points_test = tf.data.Dataset.cardinality(test_set).numpy()
train_set = train_set.shuffle(n_points_train)
test_set = test_set.shuffle(n_points_test).batch(n_points_test)

if batch_size is None:
    batch_size = int(n_points_train/2)+1
train_set_batched = train_set.batch(batch_size)

# rescale the metrics
g_factor = 0.2

optimizer = tf.keras.optimizers.Adam()

checkpoint_directory = model_save_path + '_checkpoints'
checkpoint_prefix = os.path.join(checkpoint_directory, "ckpt")
checkpoint = tf.train.Checkpoint(optimizer=optimizer, model=model)
status = checkpoint.restore(tf.train.latest_checkpoint(checkpoint_directory))

epoch = 0
while epoch < max_epochs:
    epoch = epoch + 1
    for step, (points, gs, g_invs, d_g_invs, sqrt_det_gs, d_sqrt_det_gs, const_coords, ignored_coords) in enumerate(train_set_batched):
        # print('points :', points[10])
        # print('gs :', gs[10])
        # print('g_invs :', g_invs[10])
        # print('d_g_invs :', d_g_invs[10])
        # print('sqrt_det_gs :', sqrt_det_gs[10])
        # print('d_sqrt_det_gs :', d_sqrt_det_gs[10])
        # print('const_coords :', const_coords[10])
        # print('ignored_coords :', ignored_coords[10])
    #for step, entries in enumerate(train_set_batched):
        st = time.time()
        with tf.GradientTape() as tape:
            loss, norm_square, d_omega_square, d_star_omega_square = tf.vectorized_map(loss_func, (points, g_factor*gs, 1/g_factor*g_invs, 1/g_factor*d_g_invs, g_factor**(3/2)*sqrt_det_gs, g_factor**(3/2)*d_sqrt_det_gs, const_coords, ignored_coords))
            loss = tf.reduce_mean(loss)
            # It's actually the avg_norm squared
            avg_norm_square = tf.reduce_mean(norm_square)
            loss = loss / avg_norm_square
            grads = tape.gradient(loss, model.trainable_weights)
        optimizer.apply_gradients(zip(grads, model.trainable_weights))

        #print('time: ', time.time() - st)

    if epoch % 10 == 0:

        print('loss: ', loss)
        print('d_omega_square: ', tf.reduce_mean(d_omega_square))
        print('d_star_omega_square: ', tf.reduce_mean(d_star_omega_square))
        print('avg_norm_square: ', avg_norm_square)
        print('max_norm_square: ', tf.math.reduce_max(norm_square))
        print('min_norm_square: ', tf.math.reduce_min(norm_square))

        for step, (points, gs, g_invs, d_g_invs, sqrt_det_gs, d_sqrt_det_gs, const_coords, ignored_coords) in enumerate(test_set):
            loss, norm_square, d_omega_square, d_star_omega_square = tf.vectorized_map(loss_func, (points, g_factor*gs, 1/g_factor*g_invs, 1/g_factor*d_g_invs, g_factor**(3/2)*sqrt_det_gs, g_factor**(3/2)*d_sqrt_det_gs, const_coords, ignored_coords))
            loss = tf.reduce_mean(loss)
            avg_norm_square = tf.reduce_mean(norm_square)
            loss = loss / avg_norm_square
            print('test loss: ', loss)
            print('test d_omega_square: ', tf.reduce_mean(d_omega_square))
            print('test d_star_omega_square: ', tf.reduce_mean(d_star_omega_square))
            print('test avg_norm_square: ', avg_norm_square)
            print('test max_norm_square: ', tf.math.reduce_max(norm_square))
            print('test min_norm_square: ', tf.math.reduce_min(norm_square))
        #model.save(model_save_path + '_tmp/epoch_{}'.format(epoch))
        checkpoint.save(file_prefix=checkpoint_prefix)

model.save(model_save_path)

