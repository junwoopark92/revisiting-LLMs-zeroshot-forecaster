import argparse


informer_parser = argparse.ArgumentParser(description='TimesNet')

informer_parser.add_argument('--learn', type=str, required=True, default='offline',
                    help='offline, iid-online, online, hybrid-online, offline-val, offline-sec')

# basic config
informer_parser.add_argument('--task_name', type=str, required=True, default='long_term_forecast',
                    help='task name, options:[long_term_forecast, short_term_forecast, imputation, classification, anomaly_detection]')
informer_parser.add_argument('--is_training', type=int, required=True, default=1, help='status')
informer_parser.add_argument('--model_id', type=str, required=True, default='test', help='model id')
informer_parser.add_argument('--model', type=str, required=True, default='Autoformer',
                    help='model name, options: [Autoformer, Transformer, TimesNet]')

informer_parser.add_argument('--use_label_method', type=str, required=True, default=None, help='individiual, shared, attn')
informer_parser.add_argument('--simulation_type', nargs='+', default=None, help="")
informer_parser.add_argument('--num_class', type=int, default=3, help='input sequence length')

# data loader
informer_parser.add_argument('--data', type=str, required=True, default='ETTm1', help='dataset type')
informer_parser.add_argument('--root_path', type=str, default='./data/ETT/', help='root path of the data file')
informer_parser.add_argument('--data_path', type=str, default='ETTh1.csv', help='data file')
informer_parser.add_argument('--features', type=str, default='M',
                    help='forecasting task, options:[M, S, MS]; M:multivariate predict multivariate, S:univariate predict univariate, MS:multivariate predict univariate')
informer_parser.add_argument('--target', type=str, default='OT', help='target feature in S or MS task')
informer_parser.add_argument('--freq', type=str, default='h',
                    help='freq for time features encoding, options:[s:secondly, t:minutely, h:hourly, d:daily, b:business days, w:weekly, m:monthly], you can also use more detailed freq like 15min or 3h')
informer_parser.add_argument('--checkpoints', type=str, default='./checkpoints/', help='location of model checkpoints')

# forecasting task
informer_parser.add_argument('--seq_len', type=int, default=96, help='input sequence length')
informer_parser.add_argument('--label_len', type=int, default=48, help='start token length')
informer_parser.add_argument('--pred_len', type=int, default=96, help='prediction sequence length')
informer_parser.add_argument('--seasonal_patterns', type=str, default='Monthly', help='subset for M4')
informer_parser.add_argument('--inverse', action='store_true', help='inverse output data', default=False)

# inputation task
informer_parser.add_argument('--mask_rate', type=float, default=0.25, help='mask ratio')

# anomaly detection task
informer_parser.add_argument('--anomaly_ratio', type=float, default=0.25, help='prior anomaly ratio (%)')

# model define
informer_parser.add_argument('--expand', type=int, default=2, help='expansion factor for Mamba')
informer_parser.add_argument('--d_conv', type=int, default=4, help='conv kernel size for Mamba')
informer_parser.add_argument('--top_k', type=int, default=5, help='for TimesBlock')
informer_parser.add_argument('--num_kernels', type=int, default=6, help='for Inception')
informer_parser.add_argument('--enc_in', type=int, default=7, help='encoder input size')
informer_parser.add_argument('--dec_in', type=int, default=7, help='decoder input size')
informer_parser.add_argument('--c_out', type=int, default=7, help='output size')
informer_parser.add_argument('--d_model', type=int, default=512, help='dimension of model')
informer_parser.add_argument('--n_heads', type=int, default=8, help='num of heads')
informer_parser.add_argument('--e_layers', type=int, default=2, help='num of encoder layers')
informer_parser.add_argument('--d_layers', type=int, default=1, help='num of decoder layers')
informer_parser.add_argument('--d_ff', type=int, default=2048, help='dimension of fcn')
informer_parser.add_argument('--moving_avg', type=int, default=25, help='window size of moving average')
informer_parser.add_argument('--factor', type=int, default=1, help='attn factor')
informer_parser.add_argument('--distil', action='store_false',
                    help='whether to use distilling in encoder, using this argument means not using distilling',
                    default=True)
informer_parser.add_argument('--dropout', type=float, default=0.1, help='dropout')
informer_parser.add_argument('--embed', type=str, default='timeF',
                    help='time features encoding, options:[timeF, fixed, learned]')
informer_parser.add_argument('--activation', type=str, default='gelu', help='activation')
informer_parser.add_argument('--output_attention', action='store_true', help='whether to output attention in ecoder')
informer_parser.add_argument('--channel_independence', type=int, default=1,
                    help='0: channel dependence 1: channel independence for FreTS model')
informer_parser.add_argument('--decomp_method', type=str, default='moving_avg',
                    help='method of series decompsition, only support moving_avg or dft_decomp')
informer_parser.add_argument('--use_norm', type=int, default=1, help='whether to use normalize; True 1 False 0')
informer_parser.add_argument('--down_sampling_layers', type=int, default=3, help='num of down sampling layers')
informer_parser.add_argument('--down_sampling_window', type=int, default=2, help='down sampling window size')
informer_parser.add_argument('--down_sampling_method', type=str, default='avg',
                    help='down sampling method, only support avg, max, conv')
informer_parser.add_argument('--seg_len', type=int, default=48,
                    help='the length of segmen-wise iteration of SegRNN')

# optimization
informer_parser.add_argument('--num_workers', type=int, default=10, help='data loader num workers')
informer_parser.add_argument('--itr', type=int, default=1, help='experiments times')
informer_parser.add_argument('--train_epochs', type=int, default=10, help='train epochs')
informer_parser.add_argument('--batch_size', type=int, default=32, help='batch size of train input data')
informer_parser.add_argument('--patience', type=int, default=3, help='early stopping patience')
informer_parser.add_argument('--learning_rate', type=float, default=0.0001, help='optimizer learning rate')
informer_parser.add_argument('--des', type=str, default='test', help='exp description')
informer_parser.add_argument('--loss', type=str, default='MSE', help='loss function')
informer_parser.add_argument('--lradj', type=str, default='type1', help='adjust learning rate')
informer_parser.add_argument('--use_amp', action='store_true', help='use automatic mixed precision training', default=False)

# GPU
informer_parser.add_argument('--use_gpu', type=bool, default=True, help='use gpu')
informer_parser.add_argument('--gpu', type=int, default=0, help='gpu')
informer_parser.add_argument('--use_multi_gpu', action='store_true', help='use multiple gpus', default=False)
informer_parser.add_argument('--devices', type=str, default='0,1,2,3', help='device ids of multile gpus')

# de-stationary projector params
informer_parser.add_argument('--p_hidden_dims', type=int, nargs='+', default=[128, 128],
                    help='hidden layer dimensions of projector (List)')
informer_parser.add_argument('--p_hidden_layers', type=int, default=2, help='number of hidden layers in projector')

# metrics (dtw)
informer_parser.add_argument('--use_dtw', type=bool, default=False, 
                    help='the controller of using dtw metric (dtw is time consuming, not suggested unless necessary)')

# Augmentation
informer_parser.add_argument('--augmentation_ratio', type=int, default=0, help="How many times to augment")
informer_parser.add_argument('--seed', type=int, default=2, help="Randomization seed")
informer_parser.add_argument('--jitter', default=False, action="store_true", help="Jitter preset augmentation")
informer_parser.add_argument('--scaling', default=False, action="store_true", help="Scaling preset augmentation")
informer_parser.add_argument('--permutation', default=False, action="store_true", help="Equal Length Permutation preset augmentation")
informer_parser.add_argument('--randompermutation', default=False, action="store_true", help="Random Length Permutation preset augmentation")
informer_parser.add_argument('--magwarp', default=False, action="store_true", help="Magnitude warp preset augmentation")
informer_parser.add_argument('--timewarp', default=False, action="store_true", help="Time warp preset augmentation")
informer_parser.add_argument('--windowslice', default=False, action="store_true", help="Window slice preset augmentation")
informer_parser.add_argument('--windowwarp', default=False, action="store_true", help="Window warp preset augmentation")
informer_parser.add_argument('--rotation', default=False, action="store_true", help="Rotation preset augmentation")
informer_parser.add_argument('--spawner', default=False, action="store_true", help="SPAWNER preset augmentation")
informer_parser.add_argument('--dtwwarp', default=False, action="store_true", help="DTW warp preset augmentation")
informer_parser.add_argument('--shapedtwwarp', default=False, action="store_true", help="Shape DTW warp preset augmentation")
informer_parser.add_argument('--wdba', default=False, action="store_true", help="Weighted DBA preset augmentation")
informer_parser.add_argument('--discdtw', default=False, action="store_true", help="Discrimitive DTW warp preset augmentation")
informer_parser.add_argument('--discsdtw', default=False, action="store_true", help="Discrimitive shapeDTW warp preset augmentation")
informer_parser.add_argument('--extra_tag', type=str, default="", help="Anything extra")