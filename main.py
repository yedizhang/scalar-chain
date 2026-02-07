import numpy as np
from scipy.special import lambertw
import matplotlib.pyplot as plt
import matplotlib
plt.rc('font', family="Arial")
plt.rcParams['mathtext.fontset'] = 'cm'
plt.rcParams['font.size'] = '16'
plt.rcParams['axes.spines.right'] = False
plt.rcParams['axes.spines.top'] = False
cmap = matplotlib.colormaps['cividis_r']

def lambert_sol(args, lr):
    a_0 = args['init'] * args['xx'] / args['yx']  # convert w^L init to alpha init
    t = np.linspace(0, args['T']*lr, num=args['T'])
    beta = np.log(1/a_0-1) + 1/a_0 - t - 1
    e_beta = np.exp(beta)  
    a_t = 1 / (1+lambertw(e_beta))
    return a_t.real

def sim_total_w(args):
    L = args['L']
    T = args['T']
    lr = args['LR']
    a = np.zeros(T)
    a[0] = args['init']
    for t in range(T-1):
        a[t+1] = a[t] + lr * L * 2 * (1-a[t]) * a[t]**(2-2/L)
    return a

def compute_lr(args, lr, exact=True):
    l = args['L']
    r = args['yx'] / args['xx']
    if exact:
        if args['model'] == 'mlp':
            LR = lr * r**(2/l-2) / l
        elif args['model'] == 'res_b1':
            LR = lr * r**(2/l-2)
        elif args['model'] == 'res_b2':
            LR = lr / (4 * args['xx'] * r**(2-2/l) * (r**(1/l)-1))
    else:
        if args['model'] == 'mlp':
            LR = lr / l
        elif args['model'] == 'res_b1':
            LR = lr
        elif args['model'] == 'res_b2':
            LR = lr * l / (4 * args['xx'] * r**2 * np.log(r))
    return LR

def Loss(args, u):
    if args['model'] == 'mlp':
        return args['xx'] * (args['yx']/args['xx']-u**args['L'])**2
    elif args['model'] == 'res_b1':
        factor = 1 + u / np.sqrt(args['L'])
        return args['xx'] * (args['yx']/args['xx']-factor**args['L'])**2
    elif args['model'] == 'res_b2':
        factor = 1 + u**2 / args['L']
        return args['xx'] * (args['yx']/args['xx']-factor**args['L'])**2

def train(args):
    L = args['L']
    T = args['T']
    lr = args['LR']
    if args['model'] == 'mlp':
        u = args['init']**(1/L)
    else:
        u = args['init']
    results = {'Ls': np.zeros(T),
               'U': np.zeros(T)}
    
    for t in range(T):
        results['U'][t] = u
        results['Ls'][t] = Loss(args, u)
        if args['model'] == 'mlp':
            u += lr * (args['yx'] - args['xx']*u**L) * u**(L-1)
        elif args['model'] == 'res_b1':
            factor = 1 + u / np.sqrt(L)
            u += lr * (args['yx'] - args['xx']*factor**L) * (factor**(L-1)) / np.sqrt(L)
        elif args['model'] == 'res_b2':
            factor = 1 + u**2 / L
            u += 2*lr * (args['yx'] - args['xx']*factor**L) * (factor**(L-1)) * u / L

    return results

def sweep_depth(args):
    plt.figure(figsize=(4, 3))
    lr = args['LR']
    layers = [1,2,4,8,16,32,64,128]
    for i, l in enumerate(layers):
        args['L'] = l
        args['LR'] = compute_lr(args, lr)
        results = train(args)
        a = results['U']**l
        plt.plot(a, c=cmap(i/8), label=f'$L={l}$')
        # plt.plot(results['Ls'], c=cmap(i/8), label=f'$L={l}$')
    if args['model'] == 'mlp':
        a = lambert_sol(args, lr) * args['yx'] / args['xx']
        plt.plot(a, c=cmap(1.), label=r'$L=\infty$')
        plt.title(fr'$\alpha(0)={args['init']}$')
        plt.ylabel(r'$w_l^L(t)$')
    else:
        plt.title(fr'$w_l(0)={args['init']}$')
        # plt.ylabel(r'$\mathcal{L}(t)$')
    plt.xlim([0, args['T']])
    plt.ylim([-0.02, 1.02])
    plt.gca().set_yticks([0,0.5,1])
    plt.xlabel('Steps')
    plt.legend(frameon=False,fontsize='x-small')  # loc='center right',bbox_to_anchor=(1.15, 0.48)
    plt.tight_layout(pad=0.1)
    plt.savefig(f"{args['model']}_init{args['init']}_lr{lr}_depthsweep.pdf")

def hyper_transfer(args, exact=False):
    plt.figure(figsize=(4, 3.3))
    layers = [2,4,8,16,32,64,128]
    lr = np.linspace(0.01, 2, 100, endpoint=True)
    L_t = np.zeros((len(layers), len(lr)))
    for i, l in enumerate(layers):
        args['L'] = l
        for j in range(len(lr)):
            args['LR'] = compute_lr(args, lr[j], exact)
            results = train(args)
            L_t[i,j] = results['Ls'][-1]
        plt.plot(lr, L_t[i,:], c=cmap(i/8), label=f'$L={l}$')    
    plt.title(rf'$\mu_{{yx}}/\mu_{{xx}}={args['yx']/args['xx']}$')
    plt.xlabel(r'$\eta$')
    plt.ylabel(r'$\mathcal{L}(t)$')
    plt.ylim([10e-35, 10])
    plt.yscale('log')
    plt.legend(frameon=False,fontsize='x-small')
    plt.tight_layout(pad=0.2)
    plt.savefig(f'{args['model']}_hypertran_yx{args['yx']}_xx{args['xx']}_{exact}.pdf')

if __name__ == "__main__":
    args = {'model': 'mlp',
            'yx': 0.5,
            'xx': 1,
            'L': 2,
            'init': 0.1,
            'T': 30,
            'LR': 1}

    sweep_depth(args)
    hyper_transfer(args, True)
    plt.show()
