from eICU_preprocessing.split_train_test import create_folder
from models.run_transformer import BaselineTransformer
from models.initialise_arguments import initialise_transformer_arguments


if __name__=='__main__':

    c = initialise_transformer_arguments()
    c['exp_name'] = 'Transformer'
    c['dataset'] = 'MIMIC'
    c['main_dropout_rate'] = 0
    c['last_linear_size'] = 36
    c['batch_norm'] = 'mybatchnorm'
    c['no_diag'] = True
    c['batch_size'] = 64
    c['n_layers'] = 2
    c['feedforward_size'] = 64
    c['d_model'] = 32
    c['n_heads'] = 1
    c['learning_rate'] = 0.001
    c['trans_dropout_rate'] = 0.05
    c['n_epochs'] = 15

    log_folder_path = create_folder('models/experiments/MIMIC', c.exp_name)
    transformer = BaselineTransformer(config=c,
                                      n_epochs=c.n_epochs,
                                      name=c.exp_name,
                                      base_dir=log_folder_path,
                                      explogger_kwargs={'folder_format': '%Y-%m-%d_%H%M%S{run_number}'})
    transformer.run()