import numpy as np
import matplotlib.pyplot as plt

z = np.load(r"C:\Users\knkyy\MaxEntropyMethod\runs\2026-01-01_234745_debug\artifacts\result.npz", allow_pickle=True)
w = z["w"]          # (R,)
alphas = z["alphas"]  # (R,)

idx = np.argsort(-w)
cum = np.cumsum(w[idx])

print("Top-k cumulative weights:")
for k in [1,2,3,5,10,20,50]:
    print(k, float(cum[k-1]))

print("\nTop-10 (alpha, w):")
for i in idx[:10]:
    print(float(alphas[i]), float(w[i]))

n_means = z["n_means"]   # (R, N)
w = z["w"]               # (R,)
omega = z["omega"]

# 全α
n_all = (w[:,None] * n_means).sum(axis=0)

# Top-2
idx = np.argsort(-w)
top2 = idx[:2]
w2 = w[top2] / w[top2].sum()
n_top2 = (w2[:,None] * n_means[top2]).sum(axis=0)

diff = n_top2 - n_all
print("n-space comparison:")
print("||.||_inf =", np.max(np.abs(diff)))
print("rel L2 =", np.linalg.norm(diff)/np.linalg.norm(n_all))
print("rel mean abs =", np.mean(np.abs(diff))/np.mean(n_all))


plt.figure()
plt.plot(omega, n_all, label="All α (weighted)", lw=2)
plt.plot(omega, n_top2, "--", label="Top-2 α only", lw=2)
plt.plot(omega, diff, ":", label="Difference", lw=1)
plt.legend()
plt.xlabel("ω")
plt.ylabel("A(ω)")
plt.title("Top-2 α vs All α reconstruction")
plt.show()