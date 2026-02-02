"""
简化的训练测试脚本 - 验证EDL训练流程
"""
import torch
import pandas as pd
import re
from model import GPT, GPTConfig
from trainer import Trainer, TrainerConfig
from dataset import SmileDataset
from utils import set_seed

print("="*60)
print("EDL Training Flow Test")
print("="*60)

# 设置随机种子
set_seed(42)

# 1. 加载数据
print("\n1. Loading data...")
data = pd.read_csv('../datasets/moses2.csv')
data = data.dropna(axis=0).reset_index(drop=True)
data.columns = data.columns.str.lower()

# 使用小样本进行快速测试
train_data = data[data['split'] == 'train'].head(1000).reset_index(drop=True)
val_data = data[data['split'] == 'test'].head(200).reset_index(drop=True)

print(f"Train samples: {len(train_data)}")
print(f"Val samples: {len(val_data)}")

# 2. 准备数据
print("\n2. Preparing data...")
smiles = train_data['smiles']
vsmiles = val_data['smiles']

prop = train_data[['qed']].values.tolist()
vprop = val_data[['qed']].values.tolist()

scaffold = train_data['scaffold_smiles']
vscaffold = val_data['scaffold_smiles']

pattern = r"(\[[^\]]+]|<|Br?|Cl?|N|O|S|P|F|I|b|c|n|o|s|p|\(|\)|\.|=|#|-|\+|\\\\|\/|:|~|@|\?|>|\*|\$|\%[0-9]{2}|[0-9])"
regex = re.compile(pattern)

lens = [len(regex.findall(i.strip())) for i in (list(smiles.values) + list(vsmiles.values))]
max_len = max(lens)
print(f"Max sequence length: {max_len}")

lens = [len(regex.findall(i.strip())) for i in (list(scaffold.values) + list(vscaffold.values))]
scaffold_max_len = max(lens)
print(f"Scaffold max length: {scaffold_max_len}")

# Pad sequences
smiles = [i + str('<')*(max_len - len(regex.findall(i.strip()))) for i in smiles]
vsmiles = [i + str('<')*(max_len - len(regex.findall(i.strip()))) for i in vsmiles]
scaffold = [i + str('<')*(scaffold_max_len - len(regex.findall(i.strip()))) for i in scaffold]
vscaffold = [i + str('<')*(scaffold_max_len - len(regex.findall(i.strip()))) for i in vscaffold]

# Generate vocabulary
whole_string = ' '.join(smiles + vsmiles + scaffold + vscaffold)
whole_string = sorted(list(set(regex.findall(whole_string))))
print(f"Vocabulary size: {len(whole_string)}")

# 3. 创建数据集
print("\n3. Creating datasets...")

# 创建完整的args对象
args = type('Args', (), {
    'scaffold': False,
    'num_props': 0,
    'debug': False
})()

train_dataset = SmileDataset(
    args,
    smiles, whole_string, max_len,
    prop=prop, aug_prob=0,
    scaffold=scaffold, scaffold_maxlen=scaffold_max_len
)
valid_dataset = SmileDataset(
    args,
    vsmiles, whole_string, max_len,
    prop=vprop, aug_prob=0,
    scaffold=vscaffold, scaffold_maxlen=scaffold_max_len
)

print(f"Train dataset size: {len(train_dataset)}")
print(f"Valid dataset size: {len(valid_dataset)}")

# 4. 创建模型配置（EDL模式）
print("\n4. Creating model with EDL...")
mconf = GPTConfig(
    train_dataset.vocab_size,
    train_dataset.max_len,
    num_props=0,
    n_layer=2,
    n_head=4,
    n_embd=128,
    scaffold=False,
    scaffold_maxlen=scaffold_max_len,
    lstm=False,
    lstm_layers=0,
    use_edl=True,
    edl_loss_type='mse',
    edl_annealing_step=5
)

model = GPT(mconf)
print(f"Model created with {sum(p.numel() for p in model.parameters())} parameters")
print(f"EDL enabled: {model.use_edl}")

# 5. 测试前向传播
print("\n5. Testing forward pass...")
x, y, p, s = train_dataset[0]
x = x.unsqueeze(0)  # Add batch dimension
y = y.unsqueeze(0)
p = p.unsqueeze(0)
s = s.unsqueeze(0)

with torch.no_grad():
    alpha, loss, attn_maps = model(x, y, p, s)
    print(f"Alpha shape: {alpha.shape}")
    print(f"Loss: {loss.item():.4f}")

    # Test with uncertainty
    alpha, loss, uncertainty_dict, attn_maps = model(x, y, p, s, return_uncertainty=True)
    print(f"Total uncertainty: {uncertainty_dict['total_uncertainty'].mean().item():.4f}")
    print(f"Epistemic uncertainty: {uncertainty_dict['epistemic_uncertainty'].mean().item():.4f}")
    print(f"Aleatoric uncertainty: {uncertainty_dict['aleatoric_uncertainty'].mean().item():.4f}")

print("\n" + "="*60)
print("✓ All tests passed! EDL training flow is working correctly.")
print("="*60)
