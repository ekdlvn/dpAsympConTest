# C:/Users/N/AppData/Local/Programs/Python/Python39/python.exe
import functions as ft
import numpy as np
import pandas as pd

x = [10,20,30,40]
mu = 0.1
sens = 1
sig=None

# ft.chk()

# print(ft.Sig(0.01, sum(x), sens))

# print(ft.noisyTable(x, mu, sens))
x1 = ft.noisyTable(x, mu, sens)
x-x1
# print(ft.noisyTable2(x, mu, sens))

sig = ft.Sig(mu, sum(x), sens)
print(str(sig) + '입니다.')

x1_sig = ft.noisyTable(x=x, sig=sig)

# since np.random.normal() / seed should be set in np.random.seed()
p = np.repeat(1, 5)/5

sam = np.random.multinomial(100, p, 10) # 100 trials 10 samples => nrows 10
print(sam)
sam.shape # rows and column
sam.ndim # two dimensional
sam.size # number of cells
len(sam) # return 1st value of shape
sam.shape[0]
sam.sum() # sum all values
sam.sum(axis=0) # column-wise sum
sam.sum(axis=1) # row-wise sum

# np.savetxt("outnp.csv", sam, delimiter=",", fmt="%.5f")
# sam_np = np.loadtxt("outnp.csv", delimiter=",")

c_name = ["d"+str(a) for a in range(5)]

# df_sam = pd.DataFrame(sam, columns=c_name)
# df_sam.to_csv("outdp.csv", mode="w")
# sam_df = pd.read_csv("outdp.csv")

# print(sam_np)
# print(sam_df)

p_y = np.repeat(1, 2)/2
# p_x = np.repeat(1, 2)/2
p_x = np.array(list(range(3))[1:])
print(p_x)

# test_py = ft.nabla(p_y, p_x)
# test_py.shape
# print(test_py)

# test_mat = ft.prj_A(p_x, p_y)
# test_mat.keys()
# test_mat['prj_A'] # test_mat[list(test_mat.keys())[0]]
# test_mat['pi'] # test_mat[list(test_mat.keys())[1]]
# test_mat['r']
# test_mat['c']
# test_mat['Jc']
# test_mat['Jr']
# test_mat['D_half_pi']
# test_mat['D_half_inv']
# test_mat['D_pi1']
# test_mat['D_pi2']


test_mat = ft.covDat(p_x, p_y)
test_mat.keys()
test_mat['prj_A'] # test_mat[list(test_mat.keys())[0]]
test_mat['pi'] # test_mat[list(test_mat.keys())[1]]
test_mat['r']
test_mat['c']
test_mat['Jc']
test_mat['Jr']
test_mat['D_half_pi']
test_mat['D_half_inv']
test_mat['D_pi1']
test_mat['D_pi2']
test_mat['covDat']
