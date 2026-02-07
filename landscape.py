import numpy as np
import matplotlib.pyplot as plt
import matplotlib
plt.style.use('default')
plt.rc('font', family="Arial")
plt.rcParams['mathtext.fontset'] = 'cm'
plt.rcParams['font.size'] = '14'
cmap = matplotlib.colormaps['cividis_r']

def Loss(w, L, model='mlp'):
    if model == 'mlp':
        return ((1-w**L)**2) / 2
    elif model == 'res_b1':
        factor = 1 + w / np.sqrt(L)
        return ((2-factor**L)**2) / 2
    elif model == 'res_b2':
        factor = 1 + w**2 / L
        return ((2-factor**L)**2) / 2

th = 1.35
w = np.linspace(-th, th, 1000)
layers = [1,2,4,6,8,10]
plt.figure(figsize=(6, 2.5))
for i, l in enumerate(layers):
    loss = Loss(w, l)
    plt.plot(w, loss, c=cmap(i/5), lw=1.4, label=f'$L={l}$')
plt.xlabel(f'$w$')
plt.ylabel(r'$\mathcal{L}(w)$')
plt.xlim([-th, th])
plt.ylim([-0.05, 0.6])
# plt.gca().set_yticks([0,0.5])
# plt.gca().set_xticks([-1,0,1])
plt.legend(frameon=False,loc='center left',bbox_to_anchor=(1, 0.5))
plt.tight_layout(pad=0.2)
plt.show()
plt.savefig("loss_landscape.svg")
