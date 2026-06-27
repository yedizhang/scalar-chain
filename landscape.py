import numpy as np
import matplotlib.pyplot as plt
import matplotlib
plt.style.use('default')
plt.rc('font', family="Arial")
plt.rcParams['mathtext.fontset'] = 'cm'
plt.rcParams['font.size'] = '14'
cmap = matplotlib.colormaps['cividis_r']


def Loss(w, L, model='mlp', w_true=2):
    # Assume E[xx] = 1
    if model == 'mlp':
        return ((w_true-w**L)**2) / 2
    elif model == 'res_b1':
        factor = 1 + w / np.sqrt(L)
        return ((w_true-factor**L)**2) / 2
    elif model == 'res_b2':
        factor = 1 + w**2 / L
        return ((w_true-factor**L)**2) / 2


def plot_loss(model, w_true, th_l, th_r, y_max):
    w = np.linspace(th_l, th_r, 1000)
    plt.figure(figsize=(6, 2.5))
    for i, l in enumerate([1,2,4,6,8,10]):
        loss = Loss(w, l, model, w_true)
        plt.plot(w, loss, c=cmap(i/5), lw=1.4, label=f'$L={l}$')
    plt.xlabel(f'$w$')
    plt.ylabel(r'$\mathcal{L}(w)$')
    plt.xlim([th_l, th_r])
    plt.ylim([-0.05, y_max])
    plt.legend(frameon=False,loc='center left',bbox_to_anchor=(1, 0.5))
    plt.tight_layout(pad=0.2)
    plt.show()

plot_loss('mlp', 1, -1.35, 1.35, 0.65)
plot_loss('res_b1', 2, -7, 2, 2.2)
plot_loss('res_b2', 2, -1.35, 1.35, 0.65)
