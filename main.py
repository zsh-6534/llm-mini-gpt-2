import torch
from torch.utils.data import Dataset, DataLoader

from src.model import GPTConfig, GPT, MyDataset, maxBatchSize
from src.train import train, eval

device = 'cuda' if torch.cuda.is_available() else 'cpu'
config = GPTConfig()
model = GPT(config)
model.to(device)


def main(epochs, source='玄幻小说', lr=1e-4, loading=False):

    # 1. 设备检测
    # 2. 模型初始化
    if loading:
        model.load_state_dict(torch.load(f'src/model/{source}.pt')['model_state_dict'])

    # 3. 参数量
    total_params = sum(p.numel() for p in model.parameters())
    print(f"Total Parameters: {(total_params / 1e6):.2f} M")

    # 4. 最大 batch_size
    # maxBatchSize(model, config, device)

    # 5. 数据准备
    dataset = MyDataset(f'src/dataset/target/{source}.txt', config.block_size)
    train_dataset, val_dataset = torch.utils.data.random_split(dataset, [0.9, 0.1])

    # 6. 加载数据
    train_dataloader = DataLoader(train_dataset, batch_size=config.batch_size, shuffle=True)
    val_dataloader = DataLoader(val_dataset, batch_size=config.batch_size, shuffle=False)

    # 7. 优化器配置
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, fused=True)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=len(train_dataloader))

    for epoch in range(1, epochs + 1):
        train_loss = train(model, optimizer, scheduler, train_dataloader, source, epoch, device)
        val_loss = eval(model, val_dataloader, device)
        print(f"Epoch {epoch + 1}, Train Loss: {train_loss:.4f}, Val Loss: {val_loss:.4f}")

        checkpoint = {
            'epoch': epoch,
            'model_state_dict': model.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            'scheduler_state_dict': scheduler.state_dict(),
            'val_loss': val_loss
        }

        torch.save(checkpoint, f'src/model/{source}.pt')


def test(text, source="玄幻小说", max_new_tokens=400, temperature=1.0):
    model.load_state_dict(torch.load(f'src/model/{source}.pt')['model_state_dict'])
    model.eval()

    with torch.no_grad():
        model.generate(text, max_new_tokens, device, temperature)


# test('在附近一个小城的酒楼,给人当大掌柜,是他父母口中的大能', "凡人修仙传", 800, 1.0)

# 训练批次增加啊 lr 逐步下降
main(3, "凡人修仙传", 10 * 1e-5, True)
