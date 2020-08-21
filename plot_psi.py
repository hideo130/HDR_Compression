import matplotlib.pyplot as plt
import japanize_matplotlib
import numpy as np

plt.rcParams['mathtext.fontset'] = 'stix'  # math fontの設定
plt.rcParams["font.size"] = 15  # 全体のフォントサイズが変更されます。

# 軸関連
plt.rcParams['xtick.direction'] = 'in'
plt.rcParams['ytick.direction'] = 'in'
# plt.rcParams['xtick.major.width'] = 1.2
# plt.rcParams['ytick.major.width'] = 1.2
plt.rcParams['axes.linewidth'] = 1.2
plt.rcParams['axes.grid'] = True  # make grid
plt.rcParams['grid.linestyle'] = '--'
plt.rcParams['grid.linewidth'] = 0.3


japanize_matplotlib
fig = plt.figure()
ax0 = fig.add_subplot(1, 1, 1, title='Plot magnitude modify function α=0.1',
                      xlabel="maginitude", ylabel="scale factor")
x = np.arange(0.01, 1, 0.001)
alpha = 0.1
beta = 0.8
for off_set in range(5):
    use_beta = 0.8 + 0.02*off_set
    c = alpha / x * (x/alpha)**use_beta
    ax0.plot(x, c, label="beta=%0.2f" % use_beta)
    ax0.legend()
plt.savefig("../img/maginitude_modify.png", dpi=400)
