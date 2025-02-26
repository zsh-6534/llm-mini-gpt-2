import torch
from torch.utils.data import Dataset, DataLoader

from src.model import GPTConfig, GPT, MyDataset, maxBatchSize

import pandas as pd
import matplotlib.pyplot as plt

# 记录 loss
loss_records = []

# plot model save path
train_plot = "src/train/"
train_model = "src/model/"


def plot_loss(source, epoch, batch_id, loss, final=False):
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
        plt.savefig(f'{train_plot}{source}-{epoch}.png')
        plt.close()


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
            torch.save(model.state_dict(), f'{train_model}{source}-Dot.pt')

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
