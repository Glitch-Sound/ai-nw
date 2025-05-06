import os
import json
import torch
import numpy as np
import matplotlib.pyplot as plt
from torch import nn
from torch.utils.data import Dataset, DataLoader, random_split
from shapely.geometry import Polygon
from shapely.geometry import Point
from shapely.geometry.polygon import Polygon as ShapelyPolygon


# Dataset
class JsonDataset(Dataset):
    def __init__(self, folder, augment=False):
        self.files = [os.path.join(folder, f) for f in os.listdir(folder) if f.endswith(".json")]
        self.augment = augment

    def __len__(self):
        return len(self.files)

    def __getitem__(self, idx):
        with open(self.files[idx], 'r') as f:
            data = json.load(f)
        polygon = np.array(data['polygon'])
        nodes = np.array(data['nodes'])
        polyline = np.array(data['polyline'])

        return torch.tensor(polygon, dtype=torch.float32), torch.tensor(nodes, dtype=torch.float32), torch.tensor(polyline, dtype=torch.float32)

# Collate
def collate_fn(batch):
    polygons, nodes, polylines = zip(*batch)

    max_polyline_len = max(len(p) for p in polylines)
    padded_polylines = [torch.cat([p, torch.zeros(max_polyline_len - len(p), 2)]) for p in polylines]

    max_polygon_len = max(len(p) for p in polygons)
    padded_polygons = [torch.cat([p, torch.zeros(max_polygon_len - len(p), 2)]) for p in polygons]

    max_node_len = max(len(n) for n in nodes)
    padded_nodes = [torch.cat([n, torch.zeros(max_node_len - len(n), 2)]) for n in nodes]

    return (
        torch.stack(padded_polygons),
        torch.stack(padded_nodes),
        torch.stack(padded_polylines),
        torch.tensor([len(p) for p in polylines])
    )

# モデル
class PolygonToPolylineModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.poly_encoder = nn.Sequential(
            nn.Linear(2, 128), nn.ReLU(),
            nn.Linear(128, 256), nn.ReLU(),
            nn.Linear(256, 256)
        )
        self.poly_transformer = nn.TransformerEncoder(
            nn.TransformerEncoderLayer(d_model=256, nhead=8, batch_first=True),
            num_layers=4
        )
        self.node_pair_encoder = nn.Sequential(
            nn.Linear(6, 128), nn.ReLU(),
            nn.Linear(128, 256), nn.ReLU(),
            nn.Linear(256, 256)
        )
        self.decoder = nn.GRU(512, 256, num_layers=2, batch_first=True, dropout=0.1)
        self.output = nn.Linear(256, 2)

    def forward(self, polygons, nodes, seq_lens):
        poly_encoded = self.poly_encoder(polygons)
        poly_feat = self.poly_transformer(poly_encoded).mean(dim=1)
        node_pair = torch.cat([nodes[:, 0], nodes[:, 1], nodes[:, 1] - nodes[:, 0]], dim=1)
        node_feat = self.node_pair_encoder(node_pair)
        context = torch.cat([poly_feat, node_feat], dim=1).unsqueeze(1)
        max_len = seq_lens.max().item()
        x = context.repeat(1, max_len, 1)
        out, _ = self.decoder(x)
        out = self.output(out)
        out[:, 0] = nodes[:, 0]
        for i in range(out.shape[0]):
            out[i, -1] = nodes[i, 1]
        # clip each intermediate point to the polygon (except start and end)
        for i in range(out.shape[0]):
            poly = ShapelyPolygon(polygons[i].cpu().numpy())
            for j in range(1, out.shape[1] - 1):
                pt = out[i, j].detach().cpu().numpy()
            if not poly.contains(Point(pt)):
                # Clip by intersection between original point and polygon boundary
                line_to_start = np.linspace(out[i, 0].detach().cpu().numpy(), pt, num=10)
                line_to_end = np.linspace(pt, out[i, -1].detach().cpu().numpy(), num=10)
                clipped = None
                for candidate in np.concatenate([line_to_start, line_to_end]):
                    if poly.contains(Point(candidate)):
                        clipped = candidate
                        break
                if clipped is not None:
                    out[i, j] = torch.tensor(clipped, dtype=out.dtype, device=out.device)
                if not poly.contains(Point(pt)):
                    # Project to nearest boundary point
                    nearest = np.array(poly.exterior.interpolate(poly.exterior.project(Point(pt))).coords[0])
                    out[i, j] = torch.tensor(nearest, dtype=out.dtype, device=out.device)
        return out

# Trainer
class Trainer:
    def __init__(self, model, lr=1e-3, device='cpu'):
        self.model = model.to(device)
        self.optimizer = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=1e-5)
        self.criterion = nn.MSELoss()
        self.grad_clip = 1.0
        self.device = device

    def train_epoch(self, loader):
        self.model.train()
        total_loss = 0
        for polygons, nodes, polylines, seq_lens in loader:
            polygons, nodes, polylines, seq_lens = polygons.to(self.device), nodes.to(self.device), polylines.to(self.device), seq_lens.to(self.device)
            self.optimizer.zero_grad()
            preds = self.model(polygons, nodes, seq_lens)
            loss = 0
            for i in range(len(preds)):
                pred = preds[i, :seq_lens[i]]
                target = polylines[i, :seq_lens[i]]
                loss += self.criterion(pred, target)
            loss /= len(preds)
            loss.backward()
            nn.utils.clip_grad_norm_(self.model.parameters(), self.grad_clip)
            self.optimizer.step()
            total_loss += loss.item()
        return total_loss / len(loader)

    def save_model(self, path):
        torch.save(self.model.state_dict(), path)

    def load_model(self, path):
        self.model.load_state_dict(torch.load(path, map_location=self.device))

# Predictor
class Predictor:
    def __init__(self, model, device='cpu'):
        self.model = model.to(device)
        self.device = device

    def predict(self, polygons, nodes, seq_len):
        self.model.eval()
        with torch.no_grad():
            return self.model(polygons.to(self.device), nodes.to(self.device), torch.tensor([seq_len], device=self.device))

    def plot_prediction(self, polygon, nodes, preds, save_path=None):
        plt.figure(figsize=(4, 4))
        plt.fill(*polygon.T, color='blue', alpha=0.3)
        plt.plot(*preds.T, color='red')
        plt.scatter(*nodes.T, color='green')
        plt.xlim(0, 1)
        plt.ylim(0, 1)
        plt.axis('off')
        if save_path:
            plt.savefig(save_path)
        else:
            plt.show()
        plt.close()

# メイン
if __name__ == "__main__":
    os.makedirs("models", exist_ok=True)
    os.makedirs("data_result", exist_ok=True)

    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    model = PolygonToPolylineModel()

    trainer = Trainer(model, device=device)
    predictor = Predictor(model, device=device)

    dataset = JsonDataset("data/train", augment=True)
    train_size = int(0.8 * len(dataset))
    val_size = len(dataset) - train_size
    train_dataset, val_dataset = random_split(dataset, [train_size, val_size])

    train_loader = DataLoader(train_dataset, batch_size=8, shuffle=True, collate_fn=collate_fn)
    val_loader = DataLoader(val_dataset, batch_size=8, shuffle=False, collate_fn=collate_fn)

    from torch.optim.lr_scheduler import ReduceLROnPlateau
    scheduler = ReduceLROnPlateau(trainer.optimizer, mode='min', factor=0.5, patience=3, threshold=1e-4, cooldown=2)

    patience = 80
    best_loss = float('inf')
    patience_counter = 0
    train_losses = []
    val_losses = []
    lr_history = []

    total_epochs = 1000
    for epoch in range(total_epochs):
        print(f"=== Epoch {epoch+1} ===")
        loss = trainer.train_epoch(train_loader)
        train_losses.append(loss)
        print(f"Train Loss: {loss:.4f}")

        val_loss = 0
        with torch.no_grad():
            for polygons, nodes, polylines, seq_lens in val_loader:
                polygons, nodes, polylines, seq_lens = polygons.to(device), nodes.to(device), polylines.to(device), seq_lens.to(device)
                preds = model(polygons, nodes, seq_lens)
                batch_loss = 0
                for i in range(len(preds)):
                    pred = preds[i, :seq_lens[i]]
                    target = polylines[i, :seq_lens[i]]
                    batch_loss += nn.functional.mse_loss(pred, target)
                val_loss += batch_loss.item() / len(preds)
        val_loss /= len(val_loader)
        val_losses.append(val_loss)
        print(f"Val Loss: {val_loss:.4f}")

        lr = trainer.optimizer.param_groups[0]['lr']
        lr_history.append(lr)
        print(f"Current LR: {lr:.6f}")

        if val_loss < best_loss:
            best_loss = val_loss
            trainer.save_model("models/best_model.pth")
            print("  Saved new best model.")
            patience_counter = 0
        else:
            patience_counter += 1
            print(f"  No improvement. Patience: {patience_counter}/{patience}")
            if patience_counter >= patience:
                print("Early stopping.")
                break

        scheduler.step(val_loss)

    # Plot loss curve
    plt.figure()
    plt.plot(train_losses, label="Train Loss")
    plt.plot(val_losses, label="Val Loss")
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.title("Training & Validation Loss")
    plt.legend()
    plt.grid(True)
    plt.savefig("models/loss_curve.png")
    plt.close()

    # Plot LR schedule
    plt.figure()
    plt.plot(lr_history, label="Learning Rate")
    plt.xlabel("Epoch")
    plt.ylabel("LR")
    plt.title("Learning Rate Schedule")
    plt.grid(True)
    plt.savefig("models/lr_curve.png")
    plt.close()

    # Predict test set
    trainer.load_model("models/best_model.pth")
    test_dataset = JsonDataset("data/test")
    for i in range(len(test_dataset)):
        polygon, nodes, polyline = test_dataset[i]
        pred = predictor.predict(polygon.unsqueeze(0), nodes.unsqueeze(0), len(polyline)).squeeze(0)
        fname = f"data_result/sample_{i:03d}.png"
        predictor.plot_prediction(polygon, nodes, pred, save_path=fname)
        print(f"Saved: {fname}")
