from eICU_preprocessing.split_train_test import create_folder
from models.run_tpc import TPC
from models.initialise_arguments import initialise_tpc_arguments


if __name__=='__main__':

    c = initialise_tpc_arguments()
    c['exp_name'] = 'TPC'
    c['dataset'] = 'eICU'
    c['main_dropout_rate'] = 0.45
    c['last_linear_size'] = 17
    c['diagnosis_size'] = 64
    c['batch_norm'] = 'mybatchnorm'
    c['n_epochs'] = 15
    c['batch_size'] = 32
    c['n_layers'] = 9
    c['kernel_size'] = 4
    c['no_temp_kernels'] = 12
    c['point_size'] = 13
    c['learning_rate'] = 0.002
    c['temp_dropout_rate'] = 0.05
    c['temp_kernels'] = [12] * 9 if not c['share_weights'] else [32] * 9
    c['point_sizes'] = [13] * 9

    log_folder_path = create_folder('models/experiments/eICU', c.exp_name)
    tpc = TPC(config=c,
              n_epochs=c.n_epochs,
              name=c.exp_name,
              base_dir=log_folder_path,
              explogger_kwargs={'folder_format': '%Y-%m-%d_%H%M%S{run_number}'})
    tpc.run()