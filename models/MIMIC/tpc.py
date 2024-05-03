from eICU_preprocessing.split_train_test import create_folder
from models.run_tpc import TPC
from models.initialise_arguments import initialise_tpc_arguments


if __name__=='__main__':

    c = initialise_tpc_arguments()
    c['exp_name'] = 'TPC'
    c['dataset'] = 'MIMIC'
    c['main_dropout_rate'] = 0
    c['last_linear_size'] = 36
    c['batch_norm'] = 'mybatchnorm'
    c['no_diag'] = True
    c['n_epochs'] = 10 if c['task'] is not 'mortality' else 6
    c['batch_size'] = 8
    c['batch_size_test'] = 8  # purely to keep experiment size small so I can run many in parallel
    c['n_layers'] = 8
    c['kernel_size'] = 5
    c['no_temp_kernels'] = 11
    c['point_size'] = 5
    c['learning_rate'] = 0.002
    c['temp_dropout_rate'] = 0.05
    c['temp_kernels'] = [11] * 8
    c['point_sizes'] = [5] * 8

    log_folder_path = create_folder('models/experiments/MIMIC', c.exp_name)
    tpc = TPC(config=c,
              n_epochs=c.n_epochs,
              name=c.exp_name,
              base_dir=log_folder_path,
              explogger_kwargs={'folder_format': '%Y-%m-%d_%H%M%S{run_number}'})
    tpc.run()