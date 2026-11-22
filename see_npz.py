import numpy as np

z = np.load(r"C:\Users\knkyy\MaxEntropyMethod\runs\2026-02-13_144622_debug\artifacts\result.npz", allow_pickle=True)

# 含まれているキー一覧
print(z.files)

# 各キーの中身を確認
for k in z.files:
    a = np.asarray(z[k])
    print(
        k,
        "shape=", a.shape,
        "dtype=", a.dtype,
        "value=", a if a.size <= 5 else a.reshape(-1)[:5]
    )
