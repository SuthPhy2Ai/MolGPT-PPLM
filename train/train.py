import pandas as pd
import argparse
from utils import set_seed
import numpy as np
import wandb

import torch
import torch.nn as nn

from model import GPT, GPTConfig
from trainer import Trainer, TrainerConfig
from dataset import SmileDataset
import re
if __name__ == '__main__':

    parser = argparse.ArgumentParser()

    parser.add_argument('--run_name', type=str,
                        help="name for wandb run", required=False)
    parser.add_argument('--debug', action='store_true',
                        default=False, help='debug')
    # in moses dataset, on average, there are only 5 molecules per scaffold
    parser.add_argument('--scaffold', action='store_true',
                        default=False, help='condition on scaffold')
    parser.add_argument('--lstm', action='store_true',
                        default=False, help='use lstm for transforming scaffold')
    parser.add_argument('--data_name', type=str, default='moses2',
                        help="name of the dataset to train on", required=False)
    parser.add_argument('--data_dir', type=str, default='../datasets',
                        help="directory containing the dataset files", required=False)
    parser.add_argument('--weights_dir', type=str, default='../cond_gpt/weights',
                        help="directory to save model weights", required=False)
    # parser.add_argument('--property', type=str, default = 'qed', help="which property to use for condition", required=False)
    parser.add_argument('--props', nargs="+", default=['qed'],
                        help="properties to be used for condition", required=False)
    parser.add_argument('--num_props', type=int, default = 0, help="number of properties to use for condition", required=False)
    # parser.add_argument('--prop1_unique', type=int, default = 0, help="unique values in that property", required=False)
    parser.add_argument('--n_layer', type=int, default=8,
                        help="number of layers", required=False)
    parser.add_argument('--n_head', type=int, default=8,
                        help="number of heads", required=False)
    parser.add_argument('--n_embd', type=int, default=256,
                        help="embedding dimension", required=False)
    parser.add_argument('--max_epochs', type=int, default=10,
                        help="total epochs", required=False)
    parser.add_argument('--batch_size', type=int, default=512,
                        help="batch size", required=False)
    parser.add_argument('--learning_rate', type=float,
                        default=6e-4, help="learning rate", required=False)
    parser.add_argument('--lstm_layers', type=int, default=0,
                        help="number of layers in lstm", required=False)

    # EDL arguments
    parser.add_argument('--use_edl', action='store_true',
                        default=False, help='use Evidential Deep Learning')
    parser.add_argument('--edl_loss_type', type=str, default='mse',
                        choices=['mse', 'log'], help='EDL loss type: mse or log')
    parser.add_argument('--edl_annealing_step', type=int, default=10,
                        help='number of epochs for KL annealing in EDL')
    parser.add_argument('--edl_kl_weight', type=float, default=1.0,
                        help='scaling factor for KL loss in EDL')

    args = parser.parse_args()

    set_seed(42)

    # Initialize wandb if available
    try:
        wandb.init(project="lig_gpt", name=args.run_name)
        use_wandb = True
    except Exception as e:
        print(f"Warning: wandb initialization failed: {e}")
        print("Continuing without wandb logging...")
        use_wandb = False
        # Create a dummy wandb object for compatibility
        class DummyWandb:
            def log(self, *args, **kwargs):
                pass
        wandb = DummyWandb()

    import os
    data_path = os.path.join(args.data_dir, args.data_name + '.csv')
    data = pd.read_csv(data_path)
    data = data.dropna(axis=0).reset_index(drop=True)
    # data = data.sample(frac = 0.1).reset_index(drop=True)
    data.columns = data.columns.str.lower()

    if 'moses' in args.data_name:
        train_data = data[data['split'] == 'train'].reset_index(
            drop=True)   # 'split' instead of 'source' in moses
    else:
        train_data = data[data['source'] == 'train'].reset_index(
            drop=True)   # 'split' instead of 'source' in moses

    # train_data = train_data.sample(frac = 0.1, random_state = 42).reset_index(drop=True)

    if 'moses' in args.data_name:
        val_data = data[data['split'] == 'test'].reset_index(
            drop=True)   # test for Moses. val for guacamol
    else:
        val_data = data[data['source'] == 'val'].reset_index(
            drop=True)   # test for Moses. val for guacamol

    # val_data = val_data.sample(frac = 0.1, random_state = 42).reset_index(drop=True)

    smiles = train_data['smiles']
    vsmiles = val_data['smiles']

    # prop = train_data[['qed']]
    # vprop = val_data[['qed']]

    prop = train_data[args.props].values.tolist()
    vprop = val_data[args.props].values.tolist()
    num_props = args.num_props

    scaffold = train_data['scaffold_smiles']
    vscaffold = val_data['scaffold_smiles']

    pattern = r"(\[[^\]]+]|<|Br?|Cl?|N|O|S|P|F|I|b|c|n|o|s|p|\(|\)|\.|=|#|-|\+|\\\\|\/|:|~|@|\?|>|\*|\$|\%[0-9]{2}|[0-9])"
    regex = re.compile(pattern)

    lens = [len(regex.findall(i.strip()))
              for i in (list(smiles.values) + list(vsmiles.values))]
    max_len = max(lens)
    print('Max len: ', max_len)

    lens = [len(regex.findall(i.strip()))
            for i in (list(scaffold.values) + list(vscaffold.values))]
    scaffold_max_len = max(lens)
    print('Scaffold max len: ', scaffold_max_len)

    smiles = [i + str('<')*(max_len - len(regex.findall(i.strip())))
                for i in smiles]
    vsmiles = [i + str('<')*(max_len - len(regex.findall(i.strip())))
                for i in vsmiles]

    scaffold = [i + str('<')*(scaffold_max_len -
                                len(regex.findall(i.strip()))) for i in scaffold]
    vscaffold = [i + str('<')*(scaffold_max_len -
                                len(regex.findall(i.strip()))) for i in vscaffold]

    # Generate vocabulary from actual data
    whole_string = ' '.join(smiles + vsmiles + scaffold + vscaffold)
    whole_string = sorted(list(set(regex.findall(whole_string))))
    print(f'Vocabulary size: {len(whole_string)}')
    print(f'Vocabulary: {whole_string}')

    train_dataset = SmileDataset(args, smiles, whole_string, max_len, prop=prop, aug_prob=0, scaffold=scaffold, scaffold_maxlen= scaffold_max_len)
    valid_dataset = SmileDataset(args, vsmiles, whole_string, max_len, prop=vprop, aug_prob=0, scaffold=vscaffold, scaffold_maxlen= scaffold_max_len)

    mconf = GPTConfig(train_dataset.vocab_size, train_dataset.max_len, num_props=num_props,  # args.num_props,
                        n_layer=args.n_layer, n_head=args.n_head, n_embd=args.n_embd, scaffold=args.scaffold, scaffold_maxlen=scaffold_max_len,
                        lstm=args.lstm, lstm_layers=args.lstm_layers,
                        use_edl=args.use_edl, edl_loss_type=args.edl_loss_type, edl_annealing_step=args.edl_annealing_step,
                        edl_kl_weight=args.edl_kl_weight)
    model = GPT(mconf)

    weights_path = os.path.join(args.weights_dir, f'{args.run_name}.pt')
    tconf = TrainerConfig(max_epochs=args.max_epochs, batch_size=args.batch_size, learning_rate=args.learning_rate,
                            lr_decay=True, warmup_tokens=0.1*len(train_data)*max_len, final_tokens=args.max_epochs*len(train_data)*max_len,
                            num_workers=10, ckpt_path=weights_path, block_size=train_dataset.max_len, generate=False)
    trainer = Trainer(model, train_dataset, valid_dataset,
                        tconf, train_dataset.stoi, train_dataset.itos)
    df = trainer.train(wandb)

    # Only save CSV if generation was enabled
    if df is not None:
        df.to_csv(f'{args.run_name}.csv', index=False)
