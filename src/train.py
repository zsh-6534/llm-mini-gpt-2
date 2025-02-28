import torch
from torch.utils.data import Dataset, DataLoader

from src.model import GPTConfig, GPT, MyDataset, maxBatchSize

import pandas as pd
import matplotlib.pyplot as plt

# 记录 loss

batch = 1
loss_records = []

# plot model save path
train_plot = "src/train/"
train_model = "src/model/"


def plot_loss(source, epoch, batch_id, loss, final=False):
    global batch
    loss_records.append({'loss': loss, 'batch': batch, 'epoch': epoch, 'batch_id': batch_id})
    batch += 1
    # 动态绘图初始化（仅首次调用时执行）
    if not hasattr(plot_loss, 'fig'):
        plt.ion()
        plot_loss.fig, plot_loss.ax = plt.subplots()
        plot_loss.line, = plot_loss.ax.plot([], [])
        plot_loss.ax.set_xlabel('Loss / Batch')
        plot_loss.ax.yaxis.set_major_formatter(plt.FormatStrFormatter('%.2f'))
        plot_loss.ax.grid(True, linestyle='--', alpha=0.5)
        # text 位置
        plot_loss.text = plot_loss.ax.text(0.5, 0.95, '', transform=plot_loss.ax.transAxes, ha='center')

    # 使用全部训练数据更新曲线（不再过滤当前epoch）
    x = [d['batch'] for d in loss_records]
    y = [d['loss'] for d in loss_records]

    # 更新文本和曲线数据
    plot_loss.text.set_text(f'Epoch {epoch} Batch {batch_id} - Loss: {loss:.4f}')
    plot_loss.line.set_data(x, y)

    # 调整坐标轴范围
    plot_loss.ax.relim()
    plot_loss.ax.autoscale_view()
    plot_loss.fig.canvas.draw()
    plt.pause(0.001)

    if final:
        # 保存时创建新图表，不影响动态绘图窗口
        full_df = pd.DataFrame([d for d in loss_records])
        full_df['smooth'] = full_df.loss.rolling(20).mean()

        fig, ax = plt.subplots()
        ax.plot(full_df['batch'], full_df['smooth'])
        ax.yaxis.set_major_formatter(plt.FormatStrFormatter('%.2f'))
        ax.set_title(f'Epoch {epoch} Batch {batch_id} - Loss: {loss:.4f}')
        ax.grid(True, linestyle='--', alpha=0.5)
        plt.savefig(f'{train_plot}{source}-{epoch}.png')
        plt.close(fig)  # 仅关闭保存用的临时图表


def train(model, optimizer, scheduler, train_loader, source, epoch, device):
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
        plot_loss(source, epoch, batch_id, loss.item(), batch_id == len(train_loader) - 1)

        if batch_id % 100 == 0:
            print(
                f"Epochs {epoch} Batch {batch_id}, Train_Loss: {loss.item():.4f}")
            # if loss.item() < best_val_loss:
            #     best_val_loss = loss.item()
            #     torch.save(model.state_dict(), f'src/model/{source}-Best.pt')

        # 计算平均损失
        if batch_id % 2000 == 0 and loss.item() < best_val_loss:
            best_val_loss = loss.item()
            # torch.save(model.state_dict(), f'{train_model}{source}-Dot.pt')

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
