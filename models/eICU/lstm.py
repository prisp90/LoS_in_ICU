from eICU_preprocessing.split_train_test import create_folder
from models.run_lstm import BaselineLSTM
from models.initialise_arguments import initialise_lstm_arguments


if __name__=='__main__':

    c = initialise_lstm_arguments()
    c['main_dropout_rate'] = 0.45
    c['last_linear_size'] = 17
    c['diagnosis_size'] = 64
    c['batch_norm'] = 'mybatchnorm'
    c['exp_name'] = 'StandardLSTM'
    c['dataset'] = 'eICU'
    c['batch_size'] = 512
    c['n_layers'] = 2
    c['hidden_size'] = 128
    c['learning_rate'] = 0.001
    c['lstm_dropout_rate'] = 0.2
    c['n_epochs'] = 8

    log_folder_path = create_folder('models/experiments/eICU', c.exp_name)
    baseline_lstm = BaselineLSTM(config=c,
                                 n_epochs=c.n_epochs,
                                 name=c.exp_name,
                                 base_dir=log_folder_path,
                                 explogger_kwargs={'folder_format': '%Y-%m-%d_%H%M%S{run_number}'})
    baseline_lstm.run()