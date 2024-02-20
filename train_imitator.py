import yaml
import torch
import torch.nn as nn
from tqdm import tqdm
from pathlib import Path

import utils
from imitator import Imitator


torch.manual_seed(0)
torch.use_deterministic_algorithms(True)

device = 'cuda' if torch.cuda.is_available() else 'cpu'


def train(model, dataloader, optimizer):
    mse_loss = nn.MSELoss()
    model.train()

    tot_loss = 0.

    for i, data in enumerate(dataloader):
        img, param = data
        img, param = img.to(device), param.to(device)

        optimizer.zero_grad()

        rendered_img = model(param.float())
        loss = mse_loss(img, rendered_img)

        loss.backward()
        optimizer.step()

        tot_loss += loss.item()

    avg_loss = tot_loss / len(dataloader)

    return avg_loss


if __name__ == '__main__':
    print('Running with', device)

    with open ('config.yaml', 'r') as f:
        cfg = yaml.safe_load(f)
    
    model = Imitator(cfg).float().to(device)
    train_loader, val_loader = utils.get_dataloader(cfg)
    optimizer = utils.get_optimizer(model, cfg)

    n_epochs = cfg['trainer']['n_epochs']
    best_val_loss = 1_000_000

    for epoch in tqdm(range(n_epochs), leave=True):
        train_loss = train(model, train_loader, optimizer)

        model.eval()
        loss = nn.MSELoss()
        val_loss = 0.

        with torch.no_grad():
            for i, data in enumerate(val_loader):
                img, param = data
                img, param = img.to(device), param.to(device)
                rendered_img = model(param.float())
                val_loss += loss(img, rendered_img)
            avg_val_loss = val_loss / (i+1)
        
        print('LOSS train {} valid {}'.format(train_loss, avg_val_loss))

        if avg_val_loss < best_val_loss:
            best_val_loss = avg_val_loss

            save_dir = Path(cfg['imitator']['checkpoint_dir'])
            utils.remove_folder(save_dir)       # Remove old result to save space 
            save_dir.mkdir(parents=True, exist_ok=True)
            model_path = save_dir / f'imitator_{epoch+1}.pt'
            torch.save(model.state_dict(), model_path.as_posix())   # Dump best result

