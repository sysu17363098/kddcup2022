# -*-Encoding: utf-8 -*-

def prep_env():
    # type: () -> dict
    """
    Desc:
        Prepare the experimental settings
    Returns:
        The initialized arguments
    """
    settings = {
        'data_path': '../data/',
        'filename': 'wtbdata_245days.csv',
        'checkpoints': "checkpoints",
        "input_len": 72,
        "train_output_len": 288,
        "output_len": 288,
        'seq_pre': 288,
        'start_col': 3,
        'in_var': 10,
        'out_var': 1,
        'day_len': 144,
        'capacity': 134,
        'train_len': 245,
        'epoch_num': 25,
        'learning_rate': 1e-4,
        'batch_size': 2048,
        'random_seed': 2020,
        'part_num': 24,  # num of group
        'group_config': [0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 1, 1, 2, 2, 2, 2, 2, 2, 3, 3, 3, 3, 3, 3, 4, 4, 4, 4, 4, 4, 5, 5,
                         5, 5, 5, 6, 6, 6, 6, 6, 6, 7, 7, 7, 7, 7, 7, 8, 8, 8, 8, 9, 9, 9, 9, 9, 9, 10, 10, 10, 10, 10,
                         11, 11, 11, 11, 11, 11, 12, 12, 12, 12, 12, 12, 13, 13, 13, 13, 13, 14, 14, 14, 14, 14, 14,
                         15, 15, 15, 15, 15, 16, 16, 16, 16, 16, 16, 17, 17, 17, 17, 17, 18, 18, 18, 18, 18, 18,
                         19, 19, 19, 19, 19, 20, 20, 20, 20, 20, 20, 21, 21, 21, 21, 21, 22, 22, 22, 22, 22, 22,
                         23, 23, 23, 23, 23],
        'pred_file': 'predict.py',
        "framework": "pytorch",
        'stride': 1,
        "gpu": 0,
        'is_debug': True,
        # TODO:
        'remove_features': ['Tmstamp', 'Wdir', 'Etmp', 'Itmp', 'Ndir', 'Pab2', 'Pab3', 'Prtv'], # ！Too much??
        'cat_features': ['time_index', 'hour', 'tid'], # added features to be dropped
        'embed_dim': 2,
        'pos_embed_dim': 16,
        "lstm_layer": 2,
        "dropout": 0.05,
        'nheads': 2,
        'nlayers': 4,
        'has_pos_encoder': False
    }

    return settings
