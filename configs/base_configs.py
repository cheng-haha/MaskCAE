import os
import argparse
import pandas as pd
from configs import parser as _parser
import sys
import yaml

args = None

def parse_arguments():
    parser = argparse.ArgumentParser()
    parser.add_argument('--seed', type=int, default=0, metavar='S',
                        help='random seed (default: 0). if seed=-1, seed is not fixed.')
    parser.add_argument('--batch_size', type=int, default=128, help='batch_size')
    parser.add_argument(
        "--config", help="Config file to use (see configs dir)", default=None
    )
    parser.add_argument('--times', type=int, default=1, help='num of different seed')
    parser.add_argument('--num_workers', type=int, default=4, help='num of workers to use')
    parser.add_argument('--device', type=int, default=2, choices=[0,1,2])
    parser.add_argument('--epochs', type=int, default=200, help='number of training epochs')
    parser.add_argument('--window_width', type=int, default=0, help='window width')
    parser.add_argument('--normalize', action='store_true', default=False, help='normalize signal based on mean/std of training samples')
    parser.add_argument('--pretrain', action='store_true', default=False)
    parser.add_argument('--not_save_res','--nsr', default=False, action='store_true',
                            help='The default value is saved.')
    parser.add_argument('--mode', type=str, default='ce', help='mode')
    parser.add_argument('--dataset_pre', type=str, default='self')
    parser.add_argument('--trial', type=str, default='default', help='trial id')
    parser.add_argument('--not_avg_exp', action='store_false', default=True)
    parser.add_argument('--opt_type', type=str, default='sgd')

    # optimization
    parser.add_argument('--learning_rate','--lr', type=float, default=0.01, help='learning rate')
    parser.add_argument('--weight_decay', type=float, default = 0.0005 , help='weight decay')
    parser.add_argument('--weight_decay_la', type=float, default = 0.0005 , help='weight decay for layer attention')
    parser.add_argument('--momentum', type=float, default=0.9, help='momentum')
    parser.add_argument('--lr_scheduler', type=str, default='S', help='lr_scheduler')
    parser.add_argument('--warm_up_epochs', type=int, default=5, help='linear warm up epochs')
    parser.add_argument('--handcrafted_lr', action='store_true', default=False, help='learning rate with warm up')
    parser.add_argument('--chhander', type=bool, default=False)


    # MultiStepLR or ExponentialLR
    parser.add_argument('--gamma',type=float,default=0.1,help='gamma')##
    parser.add_argument('--milestones',type=list,default=[40,80,120,160],help='optimizer milestones')
    parser.add_argument('--decay_epochs',type=float,default=40,help='n_epochs for CosineAnnealingLR or StepLR')

    # CosineAnnealingLR
    parser.add_argument('--n_epochs',type=float,default=20,help='n_epochs for CosineAnnealingLR')

    # dataset and model
    parser.add_argument('--model', type=str, default='EarlyFusion')
    parser.add_argument('--dataset', type=str, default='ucihar')
    parser.add_argument('--no_clean', action='store_false', default=False)
    parser.add_argument('--no_null', action='store_false', default=True)
    parser.add_argument('--train_portion', type=float, default=1.0, help='use portion of trainset')
    parser.add_argument('--model_path', type=str, default='save', help='path to save model')
    parser.add_argument('--load_model', type=str, default='', help='load the pretrained model')

    # coefficients
    parser.add_argument('--lam', type=float, default=0.0, help='An alternate measure coefficient, which defaults to 0')
    parser.add_argument('--p', type=float, default=0.0, help='An alternate measure coefficient, which defaults to 0')
    parser.add_argument('--beta', type=float, default=0.0, help='An alternate measure coefficient, which defaults to 0')



    # for wandb
    parser.add_argument('--use_wandb', default=False, action='store_true')

    # for maskcae
    parser.add_argument('--partial_ft', type=bool, default=False)
    parser.add_argument('--ft_layernumbers', type=int, default=0)
    parser.add_argument('--semi_number', type=int, default=0)
    parser.add_argument('--mask', type=float, default=-1)
    parser.add_argument('--ablationMode', type=str,default='')
    parser.add_argument('--pre_epochs', type=int)
    parser.add_argument('--decoder_type', type=str)
    parser.add_argument('--IP_type', type=str)
    parser.add_argument('--loss_type', type=str)
    parser.add_argument('--linear_evaluation', type=bool, default=False)
    parser.add_argument('--vis_feature', action='store_true', default=False)
    args = parser.parse_args()  

    get_config(args)

    return args


def get_config(args):
    # get commands from command line
    override_args = _parser.argv_to_vars(sys.argv)
    yaml_path =  os.path.join(os.path.dirname(os.path.dirname(__file__)) , args.config )
    print(f'==> local yaml path:{yaml_path}')
    # load yaml file
    yaml_txt = open( yaml_path).read()

    # override args
    loaded_yaml = yaml.load(yaml_txt, Loader=yaml.FullLoader)
    for v in override_args:
        loaded_yaml[v] = getattr(args, v)

    print(f"==> Reading YAML config from {args.config}")
    args.__dict__.update(loaded_yaml)


def run_args():
    global args
    if args is None:
        args = parse_arguments()


run_args()

if args.pretrain:
    args.lambda_cls = 0.0
    args.lambda_ssl = 1.0
        
def dict_to_markdown(d, max_str_len=120):
    # convert list into its str representation
    d = {k: v.__repr__() if isinstance(v, list) else v for k, v in d.items()}
    # truncate string that is longer than max_str_len
    if max_str_len is not None:
        d = {k: v[-max_str_len:] if isinstance(v, str) else v for k, v in d.items()}
    return pd.DataFrame(d, index=[0]).transpose().to_markdown()

#Display settings
#print(dict_to_markdown(vars(args), max_str_len=120))