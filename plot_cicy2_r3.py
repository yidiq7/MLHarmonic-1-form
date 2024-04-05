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
import matplotlib.pyplot as plt

parser = argparse.ArgumentParser()

parser.add_argument('--manifold', default='cicy2')
parser.add_argument('--layers', default='128_256_1024_15')
#parser.add_argument('--load_path')
parser.add_argument('--save_path')
args = parser.parse_args()

np.random.seed(1024)
tf.random.set_seed(1024)

cicy = args.manifold
layer = args.layers
#load_path = args.load_path
model_save_path = args.save_path

coord_set = 'z345'
#coord_set = 'z235'

if coord_set == 'z345':
    # R3 (really should be R3) is the surface without the zero coordinates, used
    # in points sampling
    points_set_path = 'dataset/cicy2_C_R3.npy'
    train_set_path = 'dataset/cicy2_C_R6'
elif coord_set == 'z235': 
    points_set_path = 'dataset/cicy2_C2_R3v2.npy'
    train_set_path = 'dataset/cicy2_C2_R6v2'
elif coord_set == 'z123':
    points_set_path = 'dataset/cicy2_C3_R3.npy'
    train_set_path = 'dataset/cicy2_C3_R6'
#train_set_path = 'dataset/dg_' + cicy + '_100000_train'
#test_set_path = 'dataset/dg_' + cicy + '_10000_train'

if model_save_path is None:
    model_save_path = 'trained_models_one_form_fixed/' + cicy + '/' + layer

CY_model = tf.keras.models.load_model('trained_models/' + cicy + '/64_256_1024_1', compile=False)

Z = sp.var('z0:6')

if coord_set == 'z345':
    # Generate points of the curve C:
    Z_C = [z0, z3, z4, z5]

    f1 = -z0**4 + z3**4 + 1/10*z4**4 + 1/10*z5**4
    f2 = -z0**2 + z4**2 + z5**2 + 1/100*z3**2
elif coord_set == 'z235':
    #C2
    Z_C = [z0, z2, z3, z5]

    f1 = -z0**4 + z2**4 + z3**4 + 1/10*z5**4
    f2 = -z0**2 + z5**2 + 1/100*z2**2 + 1/100*z3**2
elif coord_set == 'z123':
    # This gives no solutions
    Z_C = [z0, z1, z2, z3]

    f1 = -z0**4 + z1**4 + z2**4 + z3**4  + 1/10
    f2 = -z0**2 + 1 + 1/100*z1**2 + 1/100*z2**2 + 1/100*z3**2

f = [f1, f2]

try:
    points_C = np.load(points_set_path)
except:
    HS_C = mlg.cicyhypersurface.RealCICYHypersurface(Z_C, f, 10000)
    HS_C.list_patches()
    points_C = np.array(HS_C.patches[0].points)
    np.save(points_set_path, points_C)

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
        df_tf = [-4*z[0]**3, -4*z[0]**3, 4*z[1]**3, 4*z[2]**3, 4*z[3]**3, z[4]**3, 0]
    elif cicy == 'cicy2':
        df_tf = [-4*z[0]**3, 4*z[1]**3, 4*z[2]**3, 4*z[3]**3, 0.4*z[4]**3, 0.4*z[5]**3]
    return df_tf

def dg_tf(z):
    if cicy == 'cicy1':
        dg_tf = [-2*z[0], 0.5*z[1], 0, 0.5*z[2], 2*z[4], 2*z[5]]
    elif cicy == 'cicy2':
        dg_tf = [-2*z[0], 0.02*z[1], 0.02*z[2], 0.02*z[3], 2*z[4], 2*z[5]]
    return dg_tf

def df_arr(z):
    df_arr = np.column_stack((-4*z[:,0]**3, 4*z[:,1]**3, 4*z[:,2]**3, 4*z[:,3]**3, 0.4*z[:,4]**3, 0.4*z[:,5]**3))
    return df_arr

def dg_arr(z):
    dg_arr = np.column_stack((-2*z[:,0], 0.02*z[:,1], 0.02*z[:,2], 0.02*z[:,3], 2*z[:,4], 2*z[:,5]))
    return dg_arr

def project_to_surface(w, v1, v2):
    return w - np.sum(w*v1, axis=1)[:, np.newaxis] * v1 - np.sum(w*v2, axis=1)[:, np.newaxis] * v2

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

#def get_one_from(args):

def loss_func(args):
    point, g, g_inv, d_g_inv, sqrt_det_g, d_sqrt_det_g, const_coord, ignored_coord = args
    # ∂(P_L*ω^k )/∂x^i 
    with tf.GradientTape(persistent=True) as tape:
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

        omega_5d = tf.reduce_sum(omega, axis=[0, 1]) # (5, 5, 5) -> (5)
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
    omega_norm = tf.einsum('i, ij, j',omega, g_inv, omega) * sqrt_det_g
    return loss, omega_norm, d_omega_square, d_star_omega_square, omega_5d

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
    #test_set = tf.data.Dataset.load(test_set_path)
    print('Loaded datasets at ' + train_set_path)
    #print('Loaded test sets at ' + test_set_path)
except:
    #z1z2 = np.zeros((points_C.shape[0], 2))
    #points_C_R6 = np.insert(points_C, [1, 1], z1z2, axis=1)

# Concatenate the columns horizontally to form the 2D array
    
    zero_column = np.zeros(points_C.shape[0])
    if coord_set == 'z345':
        z1z2 = np.column_stack((zero_column, zero_column))
        points_C_R6 = np.insert(points_C, [1, 1], z1z2, axis=1)
    elif coord_set == 'z235':
        z1z4 = np.column_stack((zero_column, zero_column))
        points_C_R6 = np.insert(points_C, [1, 3], z1z4, axis=1)
    elif coord_set == 'z123':
        one_column = np.ones(points_C.shape[0])
        z4z5 = np.column_stack((zero_column, one_column))
        points_C_R6 = np.insert(points_C, [4, 4], z4z5, axis=1)

    HS_C_R6 = mlg.cicyhypersurface.RealCICYHypersurface(Z, f, n_pairs=10000, points=points_C_R6, auto_patch=True)
    train_set = generate_dataset(HS_C_R6)
    if train_set_path is not None:
        tf.data.Dataset.save(train_set, train_set_path)
        print('Datasets saved at ' + train_set_path)

#try:
#    model = tf.keras.models.load_model(load_path, compile=False)
#    print('Loaded model from ', load_path)
#except:
    #print('Creating a new model')
n_units = layer.split('_')
n_units = [int(n_unit) for n_unit in n_units]
model = tf.keras.Sequential([
  tf.keras.layers.Dense(n_units[0], activation=tf.square, input_dim=6),
  tf.keras.layers.Dense(n_units[1], activation=tf.square),
  tf.keras.layers.Dense(n_units[2], activation=tf.square),
  tf.keras.layers.Dense(n_units[3])])

n_points_train = tf.data.Dataset.cardinality(train_set).numpy()
#n_points_test = tf.data.Dataset.cardinality(test_set).numpy()
#train_set = train_set.shuffle(n_points_train)
#test_set = test_set.shuffle(n_points_test).batch(n_points_test)

batch_size = n_points_train
train_set_batched = train_set.batch(batch_size)

# rescale the metrics
g_factor = 0.2

checkpoint_directory = model_save_path + '_checkpoints'
checkpoint_prefix = os.path.join(checkpoint_directory, "ckpt")
checkpoint = tf.train.Checkpoint(model=model)
status = checkpoint.restore(tf.train.latest_checkpoint(checkpoint_directory))

for step, (points, gs, g_invs, d_g_invs, sqrt_det_gs, d_sqrt_det_gs, const_coords, ignored_coords) in enumerate(train_set_batched):
    loss, norm, d_omega_square, d_star_omega_square, omega_5d = tf.vectorized_map(loss_func, (points, g_factor*gs, 1/g_factor*g_invs, 1/g_factor*d_g_invs, g_factor**(3/2)*sqrt_det_gs, g_factor**(3/2)*d_sqrt_det_gs, const_coords, ignored_coords))
    loss = tf.reduce_mean(loss)
    avg_norm = tf.reduce_mean(norm)
    loss = loss / avg_norm
    print('test loss: ', loss)

    omega_renorm = (omega_5d/tf.math.sqrt(avg_norm)).numpy()

    random_indices = np.random.choice(points.shape[0], size=90, replace=False)
    points_r = points.numpy()[random_indices]
    omega_renorm_r = omega_renorm[random_indices]

    # Projection
    grad_f = df_arr(points_r)
    grad_g = dg_arr(points_r)

    grad_f = grad_f / np.linalg.norm(grad_f, axis=1)[:, np.newaxis]
    grad_g = grad_g / np.linalg.norm(grad_g, axis=1)[:, np.newaxis]

    vec_1 = grad_f + grad_g
    vec_2 = grad_f - grad_g

    vec_1 = vec_1 / np.linalg.norm(vec_1, axis=1)[:, np.newaxis]
    vec_2 = vec_2 / np.linalg.norm(vec_2, axis=1)[:, np.newaxis]

    print("Before projecting to the surface: ", omega_renorm_r)
    omega_renorm_r = project_to_surface(omega_renorm_r, vec_1, vec_2)
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')
    print(points_C)
    print(points)
    print(omega_renorm_r)
    ax.scatter(points_C[:,1], points_C[:,2], points_C[:,3], marker='.', s=1.0)
    # C1
    if coord_set == 'z345':
        ax.quiver(points_r[:,3], points_r[:,4], points_r[:,5], omega_renorm_r[:,3], omega_renorm_r[:,4], omega_renorm_r[:,5], length=0.5, color='orange')
        ax.set_title('1-form on CICY2 with z1 = 0, z2 = 0')
        plt.savefig('Curve_C.pdf')
    elif coord_set == 'z235':
    # C2
        ax.quiver(points_r[:,2], points_r[:,3], points_r[:,5], omega_renorm_r[:,2], omega_renorm_r[:,3], omega_renorm_r[:,5], length=1000, color='orange')
        ax.set_title('1-form on CICY2 with z1 = 0, z4 = 0')
        plt.savefig('Curve_C2.pdf')
