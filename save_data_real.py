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

np.random.seed(42)
tf.random.set_seed(42)

CY_model = tf.keras.models.load_model('trained_models/64_256_1024_1_p5', compile=False)
z0, z1, z2, z3, z4 = sp.symbols('z0, z1, z2, z3, z4')
Z = [z0,z1,z2,z3,z4]

f = 0.00195503*((119*z0**5)/47 + (409*z0**4*z1)/775 + (18*z0**3*z1**2)/41 + (316*z0**2*z1**3)/941 + (611*z0*z1**4)/877 + (379*z1**5)/991 + (34*z0**4 *z2)/43 + 145/334 *z0**3 *z1 *z2 + 867/866 *z0**2*z1**2 *z2 + 257/909*z0*z1**3*z2 + (267*z1**4*z2)/490 + (601*z0**3*z2**2)/61 + 299/708*z0**2 *z1*z2**2 + 981/850*z0*z1**2*z2**2 + (143*z1**3*z2**2)/641 + (538*z0**2*z2**3)/247 + 46/127*z0*z1*z2**3 + (22*z1**2 *z2**3)/91 + (81*z0*z2**4)/343 + (138*z1*z2**4)/77 + (61*z2**5)/73 + (476*z0**4*z3)/655 + 694/413*z0**3*z1*z3 + 179/776*z0**2*z1**2*z3 + 878/853*z0 *z1**3*z3 + (253 *z1**4 *z3)/456 + 952/573 *z0**3 *z2 *z3 + 167/184 *z0**2 *z1 *z2 *z3 + 981/388 *z0 *z1**2 *z2 *z3 + 273/59 *z1**3 *z2 *z3 + 915/514 *z0**2 *z2**2 *z3 + 288/59 *z0 *z1 *z2**2 *z3 + 637/745 *z1**2 *z2**2 *z3 + 335/383 *z0 *z2**3 *z3 + 690/403 *z1 *z2**3 *z3 + (43 *z2**4 *z3)/244 + (73 *z0**3 *z3**2)/34 + 539/326 *z0**2 *z1 *z3**2 +131/799 *z0 *z1**2 *z3**2 + (801 *z1**3 *z3**2)/698 + 113/163 *z0**2 *z2 *z3**2 +584/643 *z0 *z1 *z2 *z3**2 + 76/319 *z1**2 *z2 *z3**2 + 973/3 *z0 *z2**2 *z3**2 + 317/857 *z1 *z2**2 *z3**2 + (379 *z2**3 *z3**2)/587 + (775 *z0**2 *z3**3)/719 +388/89 *z0 *z1 *z3**3 + (389 *z1**2 *z3**3)/461 + 27/34 *z0 *z2 *z3**3 +488/177 *z1 *z2 *z3**3 + (49 *z2**2 *z3**3)/383 + (964 *z0 *z3**4)/355 +227 *z1 *z3**4 + (150 *z2 *z3**4)/19 + (77 *z3**5)/85 + (239 *z0**4 *z4)/377 +391/133 *z0**3 *z1 *z4 + 373/846 *z0**2 *z1**2 *z4 + 12/49 *z0 *z1**3 *z4 + (17 *z1**4 *z4)/23 +73/105 *z0**3 *z2 *z4 + 665/607 *z0**2 *z1 *z2 *z4 + 832/361 *z0 *z1**2 *z2 *z4 +288/551 *z1**3 *z2 *z4 + 841/202 *z0**2 *z2**2 *z4 + 271/822 *z0 *z1 *z2**2 *z4 +379/290 *z1**2 *z2**2 *z4 + 457/93 *z0 *z2**3 *z4 + 973/759 *z1 *z2**3 *z4 + (24 *z2**4 *z4)/55 + 71/56 *z0**3 *z3 *z4 + 74/327 *z0**2 *z1 *z3 *z4 + 701/222 *z0 *z1**2 *z3 *z4 +187/304 *z1**3 *z3 *z4 + 205/262 *z0**2 *z2 *z3 *z4 + 1/3 *z0 *z1 *z2 *z3 *z4 +271/101 *z1**2 *z2 *z3 *z4 + 995/167 *z0 *z2**2 *z3 *z4 + 244/189 *z1 *z2**2 *z3 *z4 +964/539 *z2**3 *z3 *z4 + 378/691 *z0**2 *z3**2 *z4 + 203/317 *z0 *z1 *z3**2 *z4 +9/691 *z1**2 *z3**2 *z4 + 363/161 *z0 *z2 *z3**2 *z4 + 646/65 *z1 *z2 *z3**2 *z4 +175/326 *z2**2 *z3**2 *z4 + 65/478 *z0 *z3**3 *z4 + 154/117 *z1 *z3**3 *z4 +522/71 *z2 *z3**3 *z4 + (77 *z3**4 *z4)/115 + (118 *z0**3 *z4**2)/119 +261/455 *z0**2 *z1 *z4**2 + 107/19 *z0 *z1**2 *z4**2 + (287 *z1**3 *z4**2)/284 + 147/694 *z0**2 *z2 *z4**2 + 841/15 *z0 *z1 *z2 *z4**2 + 537/914 *z1**2 *z2 *z4**2 +237/410 *z0 *z2**2 *z4**2 + 347/848 *z1 *z2**2 *z4**2 + (401 *z2**3 *z4**2)/352 +3/500 *z0**2 *z3 *z4**2 + 302/653 *z0 *z1 *z3 *z4**2 + 563/443 *z1**2 *z3 *z4**2 +983/765 *z0 *z2 *z3 *z4**2 + 47/15 *z1 *z2 *z3 *z4**2 + 22/179 *z2**2 *z3 *z4**2 +370/423 *z0 *z3**2 *z4**2 + 989/465 *z1 *z3**2 *z4**2 + 203/225 *z2 *z3**2 *z4**2 + (759 *z3**3 *z4**2)/919 + (996 *z0**2 *z4**3)/347 + 36/35 *z0 *z1 *z4**3 + (656 *z1**2 *z4**3)/361 + 449/180 *z0 *z2 *z4**3 + 479/205 *z1 *z2 *z4**3 + (573 *z2**2 *z4**3)/991 + 72/37 *z0 *z3 *z4**3 + 587/201 *z1 *z3 *z4**3 + 346/61*z2 *z3 *z4**3 + (350 *z3**2 *z4**3)/867 + (27 *z0 *z4**4)/49 + (464 *z1 *z4**4)/ 165 + (11 *z2 *z4**4)/108 + (353 *z3 *z4**4)/138 + (319 *z4**5)/93) + (z0**2 + z1**2 + z2**2 + z3**2 + z4**2)*(-z0**3 + z1**3 / 2 - z2**3 / 4 - z3**3 - z4**3 + z2*(z0**2 - z1**2 + z3**2 + z4**2))

def df_tf(z):
    dfdx0 = (-3 *z[0]**2 + 2 *z[0] *z[2])* (z[0]**2 + z[1]**2 + z[2]**2 + z[3]**2 + z[4]**2) + 0.00195503*((595*z[0]**4)/47 + (1636 *z[0]**3 *z[1])/775 + (54 *z[0]**2 *z[1]**2)/41 + (632 *z[0]*z[1]**3)/941 + (611 *z[1]**4)/877 + (136 *z[0]**3 *z[2])/43 + 435/334 *z[0]**2 *z[1] *z[2] + 867/433 *z[0] *z[1]**2 *z[2] + (257 *z[1]**3 *z[2])/909 + (1803 *z[0]**2 *z[2]**2)/61 + 299/354 *z[0] *z[1] *z[2]**2 + (981 *z[1]**2 *z[2]**2)/850 + (1076 *z[0] *z[2]**3)/247 + (46 *z[1] *z[2]**3)/127 + (81 *z[2]**4)/343 + (1904 *z[0]**3 *z[3])/655 + 2082/413 *z[0]**2 *z[1] *z[3] + 179/388 *z[0] *z[1]**2 *z[3] + (878 *z[1]**3 *z[3])/853 + 952/191 *z[0]**2 *z[2] *z[3] + 167/92 *z[0] *z[1] *z[2] *z[3] + 981/388 *z[1]**2 *z[2] *z[3] + 915/257 *z[0] *z[2]**2 *z[3] + 288/59 *z[1] *z[2]**2 *z[3] + (335 *z[2]**3 *z[3])/383 + (219 *z[0]**2 *z[3]**2)/34 + 539/163 *z[0] *z[1] *z[3]**2 + (131 *z[1]**2 *z[3]**2)/799 + 226/163 *z[0] *z[2] *z[3]**2 + 584/643 *z[1] *z[2] *z[3]**2 + (973 *z[2]**2 *z[3]**2)/3 + (1550 *z[0] *z[3]**3)/719 + (388 *z[1] *z[3]**3)/89 + (27 *z[2] *z[3]**3)/34 + (964 *z[3]**4)/355 + (956 *z[0]**3 *z[4])/377 + 1173/133 *z[0]**2 *z[1] *z[4] + 373/423 *z[0] *z[1]**2 *z[4] + (12 *z[1]**3 *z[4])/49 + 73/35 *z[0]**2 *z[2] *z[4] + 1330/607 *z[0] *z[1] *z[2] *z[4] + 832/361 *z[1]**2 *z[2] *z[4] + 841/101 *z[0] *z[2]**2 *z[4] + 271/822 *z[1] *z[2]**2 *z[4] + (457 *z[2]**3 *z[4])/93 + 213/56 *z[0]**2 *z[3] *z[4] + 148/327 *z[0] *z[1] *z[3] *z[4] + 701/222 *z[1]**2 *z[3] *z[4] + 205/131 *z[0] *z[2] *z[3] *z[4] + 1/3 *z[1] *z[2] *z[3] *z[4] + 995/167 *z[2]**2 *z[3] *z[4] + 756/691 *z[0] *z[3]**2 *z[4] + 203/317 *z[1] *z[3]**2 *z[4] + 363/161 *z[2] *z[3]**2 *z[4] + (65 *z[3]**3 *z[4])/478 + (354 *z[0]**2 *z[4]**2)/119 + 522/455 *z[0] *z[1] *z[4]**2 + (107 *z[1]**2 *z[4]**2)/19 + 147/347 *z[0] *z[2] *z[4]**2 + 841/15 *z[1] *z[2] *z[4]**2 + (237 *z[2]**2 *z[4]**2)/410 + 3/250 *z[0] *z[3] *z[4]**2 + 302/653 *z[1] *z[3] *z[4]**2 + 983/765 *z[2] *z[3] *z[4]**2 + (370 *z[3]**2 *z[4]**2)/ 423 + (1992 *z[0] *z[4]**3)/347 + (36 *z[1] *z[4]**3)/35 + (449 *z[2] *z[4]**3)/180 + (72 *z[3] *z[4]**3)/37 + (27 *z[4]**4)/49) + 2 *z[0] *(-z[0]**3 + z[1]**3/2 - z[2]**3/4 - z[3]**3 - z[4]**3 + z[2]*(z[0]**2 - z[1]**2 + z[3]**2 + z[4]**2))

    dfdx1 = ((3 *z[1]**2)/2 - 2 *z[1] *z[2])* (z[0]**2 + z[1]**2 + z[2]**2 + z[3]**2 + z[4]**2) + 0.00195503* ((409 *z[0]**4)/775 + (36 *z[0]**3 *z[1])/41 + (948 *z[0]**2 *z[1]**2)/941 + (2444 *z[0] *z[1]**3)/877 + (1895 *z[1]**4)/991 + (145 *z[0]**3 *z[2])/334 + 867/433 *z[0]**2 *z[1] *z[2] + 257/303 *z[0] *z[1]**2 *z[2] + (534 *z[1]**3 *z[2])/245 + (299 *z[0]**2 *z[2]**2)/708 + 981/425 *z[0] *z[1] *z[2]**2 + (429 *z[1]**2 *z[2]**2)/641 + (46 *z[0] *z[2]**3)/127 + (44 *z[1] *z[2]**3)/91 + (138 *z[2]**4)/77 + (694 *z[0]**3 *z[3])/413 + 179/388 *z[0]**2 *z[1] *z[3] + 2634/853 *z[0] *z[1]**2 *z[3] + (253 *z[1]**3 *z[3])/114 + 167/184 *z[0]**2 *z[2] *z[3] + 981/194 *z[0] *z[1] *z[2] *z[3] + 819/59 *z[1]**2 *z[2] *z[3] + 288/59 *z[0] *z[2]**2 *z[3] + 1274/745 *z[1] *z[2]**2 *z[3] + (690 *z[2]**3 *z[3])/403 + (539 *z[0]**2 *z[3]**2)/326 + 262/799 *z[0] *z[1] *z[3]**2 + (2403 *z[1]**2 *z[3]**2)/698 + 584/643 *z[0] *z[2] *z[3]**2 + 152/319 *z[1] *z[2] *z[3]**2 + (317 *z[2]**2 *z[3]**2)/857 + (388 *z[0] *z[3]**3)/89 + (778 *z[1] *z[3]**3)/461 + (488 *z[2] *z[3]**3)/177 + 227 *z[3]**4 + (391 *z[0]**3 *z[4])/133 + 373/423 *z[0]**2 *z[1] *z[4] + 36/49 *z[0] *z[1]**2 *z[4] + (68 *z[1]**3 *z[4])/23 + 665/607 *z[0]**2 *z[2] *z[4] + 1664/361 *z[0] *z[1] *z[2] *z[4] + 864/551 *z[1]**2 *z[2] *z[4] + 271/822 *z[0] *z[2]**2 *z[4] + 379/145 *z[1] *z[2]**2 *z[4] + (973 *z[2]**3 *z[4])/759 + 74/327 *z[0]**2 *z[3] *z[4] + 701/111 *z[0] *z[1] *z[3] *z[4] + 561/304 *z[1]**2 *z[3] *z[4] + 1/3 *z[0] *z[2] *z[3] *z[4] + 542/101 *z[1] *z[2] *z[3] *z[4] + 244/189 *z[2]**2 *z[3] *z[4] + 203/317 *z[0] *z[3]**2 *z[4] + 18/691 *z[1] *z[3]**2 *z[4] + 646/65 *z[2] *z[3]**2 *z[4] + (154 *z[3]**3 *z[4])/117 + (261 *z[0]**2 *z[4]**2)/455 + 214/19 *z[0] *z[1] *z[4]**2 + (861 *z[1]**2 *z[4]**2)/284 + 841/15 *z[0] *z[2] *z[4]**2 + 537/457 *z[1] *z[2] *z[4]**2 + (347 *z[2]**2 *z[4]**2)/848 + 302/653 *z[0] *z[3] *z[4]**2 + 1126/443 *z[1] *z[3] *z[4]**2 + 47/15 *z[2] *z[3] *z[4]**2 + (989 *z[3]**2 *z[4]**2)/465 + (36 *z[0] *z[4]**3)/35 + (1312 *z[1] *z[4]**3)/361 + (479 *z[2] *z[4]**3)/205 + (587 *z[3] *z[4]**3)/201 + (464 *z[4]**4)/165) +  2 *z[1] *(-z[0]**3 + z[1]**3/2 - z[2]**3/4 - z[3]**3 - z[4]**3 + z[2]* (z[0]**2 - z[1]**2 + z[3]**2 + z[4]**2))

    dfdx2 = (z[0]**2 - z[1]**2 - (3 *z[2]**2)/4 + z[3]**2 + z[4]**2)* (z[0]**2 + z[1]**2 + z[2]**2 + z[3]**2 + z[4]**2) +  0.00195503* ((34 *z[0]**4)/43 + (145 *z[0]**3 *z[1])/334 + (867 *z[0]**2 *z[1]**2)/866 + (257 *z[0] *z[1]**3)/909 + (267 *z[1]**4)/490 + (1202 *z[0]**3 *z[2])/61 +  299/354 *z[0]**2 *z[1] *z[2] + 981/425 *z[0] *z[1]**2 *z[2] + (286 *z[1]**3 *z[2])/641 + (1614 *z[0]**2 *z[2]**2)/247 + 138/127 *z[0] *z[1] *z[2]**2 + (66 *z[1]**2 *z[2]**2)/91 + (324 *z[0] *z[2]**3)/343 + (552 *z[1] *z[2]**3)/77 + (305 *z[2]**4)/73 + (952 *z[0]**3 *z[3])/573 + 167/184 *z[0]**2 *z[1] *z[3] + 981/388 *z[0] *z[1]**2 *z[3] + (273 *z[1]**3 *z[3])/59 + 915/257 *z[0]**2 *z[2] *z[3] + 576/59 *z[0] *z[1] *z[2] *z[3] + 1274/745 *z[1]**2 *z[2] *z[3] + 1005/383 *z[0] *z[2]**2 *z[3] + 2070/403 *z[1] *z[2]**2 *z[3] + (43 *z[2]**3 *z[3])/61 + (113 *z[0]**2 *z[3]**2)/163 + 584/643 *z[0] *z[1] *z[3]**2 + (76 *z[1]**2 *z[3]**2)/319 + 1946/3 *z[0] *z[2] *z[3]**2 + 634/857 *z[1] *z[2] *z[3]**2 + (1137 *z[2]**2 *z[3]**2)/587 + (  27 *z[0] *z[3]**3)/34 + (488 *z[1] *z[3]**3)/177 + (98 *z[2] *z[3]**3)/383 + (150 *z[3]**4)/19 + ( 73 *z[0]**3 *z[4])/105 + 665/607 *z[0]**2 *z[1] *z[4] + 832/361 *z[0] *z[1]**2 *z[4] + (288 *z[1]**3 *z[4])/ 551 + 841/101 *z[0]**2 *z[2] *z[4] + 271/411 *z[0] *z[1] *z[2] *z[4] + 379/145 *z[1]**2 *z[2] *z[4] +  457/31 *z[0] *z[2]**2 *z[4] + 973/253 *z[1] *z[2]**2 *z[4] + (96 *z[2]**3 *z[4])/55 + 205/262 *z[0]**2 *z[3] *z[4] + 1/3 *z[0] *z[1] *z[3] *z[4] + 271/101 *z[1]**2 *z[3] *z[4] + 1990/167 *z[0] *z[2] *z[3] *z[4] + 488/189 *z[1] *z[2] *z[3] *z[4] + 2892/539 *z[2]**2 *z[3] *z[4] + 363/161 *z[0] *z[3]**2 *z[4] + 646/65 *z[1] *z[3]**2 *z[4] + 175/163 *z[2] *z[3]**2 *z[4] + (522 *z[3]**3 *z[4])/71 + (147 *z[0]**2 *z[4]**2)/694 + 841/15 *z[0] *z[1] *z[4]**2 + (537 *z[1]**2 *z[4]**2)/914 + 237/205 *z[0] *z[2] *z[4]**2 + 347/424 *z[1] *z[2] *z[4]**2 + (1203 *z[2]**2 *z[4]**2)/352 + 983/765 *z[0] *z[3] *z[4]**2 + 47/15 *z[1] *z[3] *z[4]**2 + 44/179 *z[2] *z[3] *z[4]**2 + (203 *z[3]**2 *z[4]**2)/225 + (449 *z[0] *z[4]**3)/180 + (479 *z[1] *z[4]**3)/205 + (1146 *z[2] *z[4]**3)/991 + (346 *z[3] *z[4]**3)/61 + (11 *z[4]**4)/108) + 2 *z[2]* (-z[0]**3 + z[1]**3/2 - z[2]**3/4 - z[3]**3 - z[4]**3 + z[2]* (z[0]**2 - z[1]**2 + z[3]**2 + z[4]**2))

    dfdx3 = (2 *z[2] *z[3] - 3 *z[3]**2) *(z[0]**2 + z[1]**2 + z[2]**2 + z[3]**2 + z[4]**2) +  0.00195503 *((476 *z[0]**4)/655 + (694 *z[0]**3 *z[1])/413 + (179 *z[0]**2 *z[1]**2)/776 + (878 *z[0] *z[1]**3)/853 + (253 *z[1]**4)/456 + (952 *z[0]**3 *z[2])/573 + 167/184 *z[0]**2 *z[1] *z[2] + 981/388 *z[0] *z[1]**2 *z[2] + (273 *z[1]**3 *z[2])/59 + (915 *z[0]**2 *z[2]**2)/514 + 288/59 *z[0] *z[1] *z[2]**2 + (637 *z[1]**2 *z[2]**2)/745 + (335 *z[0] *z[2]**3)/383 + (690 *z[1] *z[2]**3)/403 + (43 *z[2]**4)/244 + (73 *z[0]**3 *z[3])/17 +  539/163 *z[0]**2 *z[1] *z[3] + 262/799 *z[0] *z[1]**2 *z[3] + (801 *z[1]**3 *z[3])/349 +  226/163 *z[0]**2 *z[2] *z[3] + 1168/643 *z[0] *z[1] *z[2] *z[3] + 152/319 *z[1]**2 *z[2] *z[3] + 1946/3 *z[0] *z[2]**2 *z[3] + 634/857 *z[1] *z[2]**2 *z[3] + (758 *z[2]**3 *z[3])/587 + ( 2325 *z[0]**2 *z[3]**2)/719 + 1164/89 *z[0] *z[1] *z[3]**2 + (1167 *z[1]**2 *z[3]**2)/461 +  81/34 *z[0] *z[2] *z[3]**2 + 488/59 *z[1] *z[2] *z[3]**2 + (147 *z[2]**2 *z[3]**2)/383 + (3856 *z[0] *z[3]**3)/ 355 + 908 *z[1] *z[3]**3 + (600 *z[2] *z[3]**3)/19 + (77 *z[3]**4)/17 + (71 *z[0]**3 *z[4])/56 +    74/327 *z[0]**2 *z[1] *z[4] + 701/222 *z[0] *z[1]**2 *z[4] + (187 *z[1]**3 *z[4])/304 +  205/262 *z[0]**2 *z[2] *z[4] + 1/3 *z[0] *z[1] *z[2] *z[4] + 271/101 *z[1]**2 *z[2] *z[4] +  995/167 *z[0] *z[2]**2 *z[4] + 244/189 *z[1] *z[2]**2 *z[4] + (964 *z[2]**3 *z[4])/539 +     756/691 *z[0]**2 *z[3] *z[4] + 406/317 *z[0] *z[1] *z[3] *z[4] + 18/691 *z[1]**2 *z[3] *z[4] +  726/161 *z[0] *z[2] *z[3] *z[4] + 1292/65 *z[1] *z[2] *z[3] *z[4] + 175/163 *z[2]**2 *z[3] *z[4] +  195/478 *z[0] *z[3]**2 *z[4] + 154/39 *z[1] *z[3]**2 *z[4] + 1566/71 *z[2] *z[3]**2 *z[4] + (308 *z[3]**3 *z[4])/ 115 + (3 *z[0]**2 *z[4]**2)/500 + 302/653 *z[0] *z[1] *z[4]**2 + (563 *z[1]**2 *z[4]**2)/443 +  983/765 *z[0] *z[2] *z[4]**2 + 47/15 *z[1] *z[2] *z[4]**2 + (22 *z[2]**2 *z[4]**2)/179 +  740/423 *z[0] *z[3] *z[4]**2 + 1978/465 *z[1] *z[3] *z[4]**2 + 406/225 *z[2] *z[3] *z[4]**2 + ( 2277 *z[3]**2 *z[4]**2)/919 + (72 *z[0] *z[4]**3)/37 + (587 *z[1] *z[4]**3)/201 + (346 *z[2] *z[4]**3)/61 + (700 *z[3] *z[4]**3)/867 + (353 *z[4]**4)/138) + 2 *z[3]*(-z[0]**3 + z[1]**3/2 - z[2]**3/4 - z[3]**3 - z[4]**3 + z[2]* (z[0]**2 - z[1]**2 + z[3]**2 + z[4]**2))

    dfdx4 = (2 *z[2] *z[4] - 3 *z[4]**2) *(z[0]**2 + z[1]**2 + z[2]**2 + z[3]**2 + z[4]**2) + 0.00195503 *((239 *z[0]**4)/377 + (391 *z[0]**3 *z[1])/133 + (373 *z[0]**2 *z[1]**2)/846 + (12 *z[0] *z[1]**3)/49 + (17 *z[1]**4)/23 + (73 *z[0]**3 *z[2])/105 + 665/607 *z[0]**2 *z[1] *z[2] +  832/361 *z[0] *z[1]**2 *z[2] + (288 *z[1]**3 *z[2])/551 + (841 *z[0]**2 *z[2]**2)/202 +  271/822 *z[0] *z[1] *z[2]**2 + (379 *z[1]**2 *z[2]**2)/290 + (457 *z[0] *z[2]**3)/93 + (973 *z[1] *z[2]**3)/759 + (24 *z[2]**4)/55 + (71 *z[0]**3 *z[3])/56 + 74/327 *z[0]**2 *z[1] *z[3] +  701/222 *z[0] *z[1]**2 *z[3] + (187 *z[1]**3 *z[3])/304 + 205/262 *z[0]**2 *z[2] *z[3] +  1/3 *z[0] *z[1] *z[2] *z[3] + 271/101 *z[1]**2 *z[2] *z[3] + 995/167 *z[0] *z[2]**2 *z[3] +   244/189 *z[1] *z[2]**2 *z[3] + (964 *z[2]**3 *z[3])/539 + (378 *z[0]**2 *z[3]**2)/691 +   203/317 *z[0] *z[1] *z[3]**2 + (9 *z[1]**2 *z[3]**2)/691 + 363/161 *z[0] *z[2] *z[3]**2 +  646/65 *z[1] *z[2] *z[3]**2 + (175 *z[2]**2 *z[3]**2)/326 + (65 *z[0] *z[3]**3)/478 + (154 *z[1] *z[3]**3)/    117 + (522 *z[2] *z[3]**3)/71 + (77 *z[3]**4)/115 + (236 *z[0]**3 *z[4])/119 +  522/455 *z[0]**2 *z[1] *z[4] + 214/19 *z[0] *z[1]**2 *z[4] + (287 *z[1]**3 *z[4])/142 +  147/347 *z[0]**2 *z[2] *z[4] + 1682/15 *z[0] *z[1] *z[2] *z[4] + 537/457 *z[1]**2 *z[2] *z[4] +  237/205 *z[0] *z[2]**2 *z[4] + 347/424 *z[1] *z[2]**2 *z[4] + (401 *z[2]**3 *z[4])/176 +  3/250 *z[0]**2 *z[3] *z[4] + 604/653 *z[0] *z[1] *z[3] *z[4] + 1126/443 *z[1]**2 *z[3] *z[4] + 1966/765 *z[0] *z[2] *z[3] *z[4] + 94/15 *z[1] *z[2] *z[3] *z[4] + 44/179 *z[2]**2 *z[3] *z[4] + 740/423 *z[0] *z[3]**2 *z[4] + 1978/465 *z[1] *z[3]**2 *z[4] + 406/225 *z[2] *z[3]**2 *z[4] + (1518 *z[3]**3 *z[4])/919 + (2988 *z[0]**2 *z[4]**2)/347 + 108/35 *z[0] *z[1] *z[4]**2 + (1968 *z[1]**2 *z[4]**2)/361 + 449/60 *z[0] *z[2] *z[4]**2 + 1437/205 *z[1] *z[2] *z[4]**2 + (1719 *z[2]**2 *z[4]**2)/991 + 216/37 *z[0] *z[3] *z[4]**2 + 587/67 *z[1] *z[3] *z[4]**2 +1038/61 *z[2] *z[3] *z[4]**2 + (350 *z[3]**2 *z[4]**2)/289 + (108 *z[0] *z[4]**3)/49 + (1856 *z[1] *z[4]**3)/165 + (11 *z[2] *z[4]**3)/27 + (706 *z[3] *z[4]**3)/69 + (1595 *z[4]**4)/93) + 2 *z[4] *(-z[0]**3 + z[1]**3/2 - z[2]**3/4 - z[3]**3 - z[4]**3 + z[2]* (z[0]**2 - z[1]**2 + z[3]**2 + z[4]**2))

    return tf.stack([dfdx0, dfdx1, dfdx2, dfdx3, dfdx4], axis=0)


def get_basis(point):
    n = tf.shape(point)[0]
    mask = tf.eye(n, dtype=point.dtype)
    basis = tf.einsum('i,jk->ijk', point, mask)
    basis_antisym = (basis - tf.transpose(basis, perm=[1, 0, 2])) / (2*tf.linalg.norm(point)**2)

    return basis_antisym

def fill_strict_upper_tri(polys):
    # Fill the outputs of the NN to a strictly upper triangular matrix
    n = 4
    # For a more general case:
    # m = tf.shape(polys)[-1]
    # n = tf.cast(tf.sqrt(0.25 + tf.cast(2 * m, dtype=tf.float32)), dtype=tf.int32)
    polys = tf.concat([polys, tf.reverse(polys[n:], axis=[0])],axis=0)
    polys = tf.reshape(polys, (4,4))
    upper_triangular = tf.linalg.band_part(polys, 0, -1)
    # Add one row of zeros at the end, and one column of zeros at the beginning
    upper_triangular = tf.pad(upper_triangular, ((0, 1), (1, 0)))

    return upper_triangular


def delete_columns(tensor, i, j, axis):
    mask = tf.ones(tensor.shape[-1], dtype=tf.bool)
    mask = tf.tensor_scatter_nd_update(mask, [[i], [j]], [False, False])
    result = tf.boolean_mask(tensor, mask, axis=axis)
    return result

def get_restriction(point, const_coord, ignored_coord):

    df = df_tf(point)
    n = tf.size(point)
    id_matrix = tf.eye(n, dtype=point.dtype)

    # Replace the i-th row with -df / df[ignored_coord]
    indices = tf.range(n)[:, tf.newaxis]
    mask = tf.cast(indices == ignored_coord, point.dtype)
    restriction = (1 - mask) * id_matrix + mask * (-df / df[ignored_coord])

    restriction = delete_columns(restriction, const_coord, ignored_coord, axis=1)

    # Remove the two columns
    #restriction = tf.concat([restriction[:, :const_coord], restriction[:, const_coord+1:]], axis=1)
    #restriction = tf.concat([restriction[:, :ignored_affine_coord], restriction[:, ignored_affine_coord+1:]], axis=1)

    return restriction

def get_CY_metrics(args):
    point, const_coord, ignored_coord = args
    with tf.GradientTape(persistent=True) as tape:
        tape.watch(point)
        point_c = tf.cast(point, dtype=tf.complex64)
        dummy = tf.constant([1.0+0.0j, 0.8 + 0.7j, 0.2 - 0.33j, 0.34 + 0.5j, 0.4 + 0.15j], dtype=tf.complex64)
        point_c = tf.stack([point_c, dummy], axis=0)
        g = tf.math.real(mlg.complex_math.complex_hessian(tf.math.real(CY_model(point_c))[0], point_c)[0])
        restriction = get_restriction(point, const_coord, ignored_coord) # (5, 3)
        g = tf.einsum('ij, jk, kl', tf.math.conj(tf.transpose(restriction)), g, restriction)
        #s, u, v = tf.linalg.svd(tf.reshape(g, [3,3]))
        #g_inv = tf.matmul(v, tf.matmul(tf.linalg.pinv(tf.linalg.diag(s)), u, adjoint_b=True))
        g_inv = tf.linalg.inv(g)
        sqrt_det_g = tf.sqrt(tf.linalg.det(g))
    d_g_inv = tape.jacobian(g_inv, point)
    d_sqrt_det_g = tape.jacobian(sqrt_det_g, point)
    d_g_inv = delete_columns(d_g_inv, const_coord, ignored_coord, axis=2)
    d_sqrt_det_g = delete_columns(d_sqrt_det_g, const_coord, ignored_coord, axis=0)
   # d_g_inv = tf.concat([d_g_inv[:, :, :const_coord], d_g_inv[:, :, const_coord+1:]], axis=2)
   # d_g_inv = tf.concat([d_g_inv[:, :, :ignored_affine_coord], d_g_inv[:, :, ignored_affine_coord+1:]], axis=2)
   # d_sqrt_det_g = tf.concat([d_sqrt_det_g[:const_coord], d_sqrt_det_g[const_coord+1:]], axis=0)
   # d_sqrt_det_g = tf.concat([d_sqrt_det_g[:ignored_affine_coord], d_sqrt_det_g[ignored_affine_coord+1:]], axis=0)
    return g, g_inv, d_g_inv, sqrt_det_g, d_sqrt_det_g

#def get_one_from(args):

def loss_func(args):
    point, g, g_inv, d_g_inv, sqrt_det_g, d_sqrt_det_g, const_coord, ignored_coord = args
    # ∂(P_L*ω^k )/∂x^i 
    with tf.GradientTape(persistent=True) as tape:
        tape.watch(point)
        restriction = get_restriction(point, const_coord, ignored_coord) # (5, 3)
        basis = get_basis(point) # (5, 5, 5)
        omega_comp = model(tf.expand_dims(point, 0))[0]
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

    # delete the extra columns
    d_Omega = delete_columns(d_Omega, const_coord, ignored_coord, axis=1)
    #d_Omega = tf.concat([d_Omega[:, :const_coord], d_Omega[:, const_coord+1:]], axis=1)
    #d_Omega = tf.concat([d_Omega[:, :ignored_affine_coord], d_Omega[:, ignored_affine_coord+1:]], axis=1) # (3,3)

    #d_star_Omega = tf.concat([d_star_Omega[:, :, :const_coord], d_star_Omega[:, :, const_coord+1:ignored_coord],
    #                          d_star_Omega[:, :, ignored_coord+1:]], axis=2) # (3, 3, 3)

    d_omega = d_Omega - tf.transpose(d_Omega)
    # The 1/2 factor comes from overcounting the upper / lower triangular
    # Multiplied by sqrt_det_g for the integration
    d_omega_square = (0.5*tf.einsum('ij, ik, jl, kl', d_omega, g, g, d_omega)) * sqrt_det_g
    d_star_Omega = (tf.einsum('m, i, im', d_sqrt_det_g, omega, g_inv) +
                    sqrt_det_g *(tf.einsum('im, im', d_Omega, g_inv) + 
                                 tf.einsum('i, imm ', omega, d_g_inv))) / 2 #(1）

    #d_star_omega_square = ((d_star_Omega[1,2,0] -
    #                        d_star_Omega[0,2,1] +
    #                        d_star_Omega[0,1,2]))**2
    # One can multiple a eps to d_star_Omega and then it can be simplied to the current form

    d_star_omega_square = 1 / sqrt_det_g * d_star_Omega**2
    loss = d_omega_square + d_star_omega_square
    omega_norm = tf.einsum('i, ij, j',omega, g, omega) * sqrt_det_g
    return loss,  omega_norm, d_omega_square, d_star_omega_square


"""

def get_g(point):
    const_coord = 0
    ignored_coord = 3
    point_c = tf.cast(point, dtype=tf.complex64)
    dummy = tf.constant([1.0+0.0j, 0.8 + 0.7j, 0.2 - 0.33j, 0.34 + 0.5j, 0.4 + 0.15j], dtype=tf.complex64)
    point_c = tf.stack([point_c, dummy], axis=0)
    g = tf.math.real(mlg.complex_math.complex_hessian(tf.math.real(CY_model(point_c))[0], point_c)[0])
    restriction = get_restriction(point, const_coord, ignored_coord) # (5, 3)
    g = tf.einsum('ij, jk, kl', tf.math.conj(tf.transpose(restriction)), g, restriction)

    return g

for i in range(5):
    for j in range(4):
        points = np.array(np.real(HS_train.patches[i].patches[j].points), dtype=np.float32)
        GG = tf.vectorized_map(get_g, points)
        for k in range(len(GG)):
            try:
                g = GG[k] 
                g_inv = tf.linalg.inv(g)
               # print("Invertible:", i, j, k, points[k], tf.linalg.det(GG[k]), 1/(df_tf(points[k])**2))
            except:
                #print("Not invertible:", i, j, k, points[k], GG[k])
                print("Not invertible:", i, j, k, points[k], tf.linalg.det(GG[k]), 1/(df_tf(points[k])**2))
                #g_pinv = tf.linalg.pinv(g)
                #print(g_pinv)
print('all computed')

"""
def generate_dataset(HS):
    dataset = None
    for i in range(5):
        for j in range(4):
            points = np.array(np.real(HS.patches[i].patches[j].points), dtype=np.float32)
            points_batch = np.array_split(points, int(len(points)/3000+1))
            for batch in points_batch:
                print("processing patch: ", i, " ", j)
                const_coords = np.full(len(batch), i, dtype=np.int32)
                #ignored_affine_coords = np.full(len(batch), j, dtype=np.int32)
                if j < i:
                    ignored_coords = np.full(len(batch), j, dtype=np.int32)
                else:
                    ignored_coords = np.full(len(batch), j+1, dtype=np.int32)
                gs, g_invs, d_g_invs, sqrt_det_gs, d_sqrt_det_gs = tf.vectorized_map(get_CY_metrics, (batch, const_coords, ignored_coords))
                new_dataset = tf.data.Dataset.from_tensor_slices((batch, gs, g_invs, d_g_invs, sqrt_det_gs, d_sqrt_det_gs, const_coords, ignored_coords))
                if dataset is None:
                    dataset = new_dataset
                else:
                    dataset = dataset.concatenate(new_dataset)
    return dataset

#gs, g_invs, d_g_invs, sqrt_det_gs, d_sqrt_det_gs = tf.vectorized_map(get_CY_metrics, (points, const_coords, ignored_coords))
train_set_path = "dataset/dg_700000_train"
#test_set_path = "dataset/dg_10000_train"

try:
    train_set = tf.data.Dataset.load(train_set_path)
    print('Loaded train sets at ' + train_set_path)
except:
    HS_train = mlg.hypersurface.RealHypersurface(Z, f, 700000)
    train_set = generate_dataset(HS_train)
    if train_set_path is not None:
        tf.data.Dataset.save(train_set, train_set_path)
        print('Datasets saved at ' + train_set_path)

load_path = 'trained_models_one_form/harmonic_one_form_100000_32_128_256_10_p2'
#load_path = None
try:
    model = tf.keras.models.load_model(load_path, compile=False)
    print('Loaded model from ', load_path)
except:
    print('Creating a new model')
    model = tf.keras.Sequential([
      tf.keras.layers.Dense(64, activation=tf.square, input_dim=5),
      tf.keras.layers.Dense(256, activation=tf.square),
      tf.keras.layers.Dense(512, activation=tf.square),
      tf.keras.layers.Dense(10)])

print('start optimizing')
n_points_train = tf.data.Dataset.cardinality(train_set).numpy()
#train_set = train_set.shuffle(n_points_train)

batch_size = n_points_train
train_set_batched = train_set.batch(batch_size)

# rescale the metrics
g_factor = 0.2

for step, (points, gs, g_invs, d_g_invs, sqrt_det_gs, d_sqrt_det_gs, const_coords, ignored_coords) in enumerate(train_set_batched):
    st = time.time()
#for step, entries in enumerate(train_set_batched):
    loss, norm, d_omega_square, d_star_omega_square = tf.vectorized_map(loss_func, (points, g_factor*gs, 1/g_factor*g_invs, 1/g_factor*d_g_invs, g_factor**(3/2)*sqrt_det_gs, g_factor**(3/2)*d_sqrt_det_gs, const_coords, ignored_coords))
    np.save('dataset_for_daniel/points_real.npy', points.numpy())
    np.save('dataset_for_daniel/d_omega_square.npy', d_omega_square.numpy())
    np.save('dataset_for_daniel/d_star_omega_square.npy', d_star_omega_square.numpy())
    np.save('dataset_for_daniel/norm_square.npy', norm.numpy())

    loss = tf.reduce_mean(loss)
        # It's actually the avg_norm squared
    avg_norm = tf.reduce_mean(norm)
    loss = loss / avg_norm
    print('loss', loss)
    print('time', time.time() - st)



