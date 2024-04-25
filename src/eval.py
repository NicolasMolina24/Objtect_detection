import torch

def predict(model, data_loader, device):
    """
    """
    model.eval()
    all_idx = []
    all_preds = []
    with torch.no_grad():
        for i, (data, target) in enumerate(data_loader):
            data = data.to(device)

            prediction = model(data)
    return {t: p for t,p  in zip(all_idx, all_preds)}