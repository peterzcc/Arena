import argparse
from arena.utils import *

def main():
    dir_path = 'dqn-model-norescale-1E-2-wait2'
    save_path = dir_path + '-cpu'
    name = 'QNet'
    epochs = range(110)
    for epoch in epochs:
        arg_params, aux_params, param_loading_path = load_params(dir_path, epoch, name)
        param_saving_path = save_params(save_path, epoch, name, params=arg_params,
                                        aux_states=aux_params, ctx=mx.cpu())
        print 'Converting %s to %s' %(param_loading_path, param_saving_path)

if __name__ == '__main__':
    main()