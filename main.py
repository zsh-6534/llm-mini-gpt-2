import src.model as GPT
import torch
from torch.utils.data import Dataset, DataLoader

from src.model import GPTConfig, GPT, MyDataset, maxBatchSize

import pandas as pd
import matplotlib.pyplot as plt

# 记录 loss
loss_records = []
learning_rate = 8e-5
source = '小说'

device = 'cuda' if torch.cuda.is_available() else 'cpu'
config = GPTConfig()


def plot_loss(epoch, batch_id, loss, final=False):
    loss_records.append({'loss': loss, 'epoch': epoch, 'batch_id': batch_id})
    # 动态绘图初始化
    if not hasattr(plot_loss, 'fig'):
        plt.ion()
        plot_loss.fig, plot_loss.ax = plt.subplots()
        plot_loss.line, = plot_loss.ax.plot([], [])
        plot_loss.ax.set_xlabel('Batch')
        plot_loss.ax.set_ylabel('Loss')

        # 新增y轴格式化
        plot_loss.ax.yaxis.set_major_formatter(plt.FormatStrFormatter('%.4f'))
        # 新增网格线
        plot_loss.ax.grid(True, linestyle='--', alpha=0.5)
        # 新增实时数值显示文本
        plot_loss.text = plot_loss.ax.text(
            0.02, 0.95, '', transform=plot_loss.ax.transAxes)

    # 更新文本内容（新增代码）
    plot_loss.text.set_text(f'Current Loss: {loss:.4f}')

    # 更新当前epoch数据
    current_data = [r for r in loss_records if r['epoch'] == epoch]
    x = [d['batch_id'] for d in current_data]
    y = [d['loss'] for d in current_data]

    # 实时更新曲线
    plot_loss.line.set_data(x, y)
    plot_loss.ax.relim()
    plot_loss.ax.autoscale_view()
    plot_loss.fig.canvas.draw()
    plt.pause(0.01)

    # epoch结束时保存图表
    if final:
        plt.ioff()
        full_df = pd.DataFrame(current_data)
        full_df['smooth'] = full_df.loss.rolling(20).mean()

        # 使用matplotlib原生绘图控制精度
        fig, ax = plt.subplots()
        ax.plot(full_df['batch_id'], full_df['smooth'])
        ax.yaxis.set_major_formatter(plt.FormatStrFormatter('%.4f'))  # 设置精度
        ax.set_title(f'Epoch {epoch} Loss')
        ax.grid(True, linestyle='--', alpha=0.5)
        plt.savefig(f'src/plot/{source}-{epoch}.png')
        plt.close()

# train 今天 了什么
# val    今天 了什么


def train(model, optimizer, scheduler, train_loader, epoch, device):
    model.train()
    total_loss = 0
    best_val_loss = float('inf')
    # 新增梯度缩放器 ↓
    # scaler = torch.amp.GradScaler(device)

    for batch_id, (x, y) in enumerate(train_loader):
        # cuda
        x, y = x.to(device), y.to(device)

        # # 混合精度
        # with torch.autocast(device_type=device, dtype=torch.float16):
        #     # 前向传播
        #     logits, loss = model(x, y)

        logits, loss = model(x, y)

        # 反向传播
        optimizer.zero_grad()

        # 梯度缩放 - 计算
        # scaler.scale(loss).backward()
        # # 更新参数
        # scaler.step(optimizer)
        # # 调整缩放系数
        # scaler.update()

        # 计算梯度（自动微分）
        loss.backward()
        # 更新参数（梯度下降）
        optimizer.step()
        # # 更新学习率
        scheduler.step()

        # 记录损失
        total_loss += loss.item()
        # print(batch_id, x.shape, y.shape)
        # 每 100 个batch 输出一次

        # 动态绘图, epoch 保存图片
        plot_loss(epoch, batch_id, loss.item(),
                  batch_id == len(train_loader) - 1)

        if batch_id % 100 == 0:
            print(
                f"Epochs {epoch} Batch {batch_id}, Train_Loss: {loss.item():.4f}")
            # if loss.item() < best_val_loss:
            #     best_val_loss = loss.item()
            #     torch.save(model.state_dict(), f'src/model/{source}-Best.pt')

        # 计算平均损失
        if batch_id % 3000 == 0 and loss.item() < best_val_loss:
            best_val_loss = loss.item()
            torch.save(model.state_dict(), f'src/model/{source}-Dot.pt')

    return total_loss / len(train_loader)


def eval(model, val_loader, device):
    # 评估模式
    model.eval()
    total_loss = 0

    with torch.no_grad():
        for batch_id, (x, y) in enumerate(val_loader):
            # cuda
            x, y = x.to(device), y.to(device)
            # 前向传播
            logits, loss = model(x, y)
            # 记录损失
            total_loss += loss.item()
            # 每 100 个batch 输出一次
            if batch_id % 100 == 0:
                print(f"Batch {batch_id}, Val_Loss: {loss.item():.4f}")
                # 计算平均损失
    return total_loss / len(val_loader)


def main(epochs=4):

    # 1. 设备检测
    # 2. 模型初始化
    model = GPT(config)
    model.to(device)

    # 3. 参数量
    total_params = sum(p.numel() for p in model.parameters())
    print(f"Total Parameters: {(total_params / 1e6):.2f} M")

    # 4. 最大 batch_size
    # maxBatchSize(model, config, device)

    # 5. 数据准备
    dataset = MyDataset(f'src/data/{source}.txt', config.block_size)
    train_dataset, val_dataset = torch.utils.data.random_split(dataset, [
                                                               0.9, 0.1])

    # 6. 加载数据
    train_dataloader = DataLoader(
        train_dataset, batch_size=config.batch_size, shuffle=True)
    val_dataloader = DataLoader(
        val_dataset, batch_size=config.batch_size, shuffle=False)

    # 7. 优化器配置
    optimizer = torch.optim.AdamW(
        model.parameters(), lr=learning_rate, fused=True)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=len(train_dataloader))

    for epoch in range(1, epochs + 1):
        train_loss = train(model, optimizer, scheduler,
                           train_dataloader, epoch, device)
        val_loss = eval(model, val_dataloader, device)
        print(
            f"Epoch {epoch + 1}, Train Loss: {train_loss:.4f}, Val Loss: {val_loss:.4f}")

        checkpoint = {
            'epoch': epoch,
            'model_state_dict': model.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            'scheduler_state_dict': scheduler.state_dict(),
            'val_loss': val_loss
        }

        torch.save(checkpoint, f'src/model/{source}.pt')


def test(text, max_new_tokens=400, temperature=1.0):
    # 1. 设备检测

    model = GPT(config)
    model.to(device)
    model.load_state_dict(torch.load(
        f'src/model/{source}.pt')['model_state_dict'])
    model.eval()

    with torch.no_grad():
        model.generate(text, max_new_tokens, device, temperature)


test('在附近一个小城的酒楼,给人当大掌柜,是他父母口中的大能人。韩家近百年来,可能就出了三叔这么', 3000, 0.92)

# main()
