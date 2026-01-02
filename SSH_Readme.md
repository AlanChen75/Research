# RTX 4090 遠端 GPU 伺服器完整指南

## 目錄

1. [連線資訊](#連線資訊)
2. [SSH 連線](#ssh-連線)
3. [環境設定](#環境設定)
4. [資料路徑對應](#資料路徑對應)
5. [檔案傳輸](#檔案傳輸)
6. [VS Code Remote SSH](#vs-code-remote-ssh)
7. [tmux 長時間運行](#tmux-長時間運行)
8. [Web UI / API（可選）](#web-ui--api可選)
9. [訓練流程](#訓練流程)
10. [監控與管理](#監控與管理)
11. [常見問題](#常見問題)

---

## 連線資訊

| 項目 | 值 |
|------|-----|
| 主機名稱 | `ac-4090.taile9e967.ts.net` |
| Tailscale IP | `100.109.200.56` |
| 使用者 | `ac-4090` |
| GPU | NVIDIA GeForce RTX 4090 D (24GB VRAM) |
| Driver | 570.195.03 |
| CUDA | 12.8 |
| Python | 3.13.3（系統） / 3.10（Conda 環境） |

---

## SSH 連線

```bash
# 使用 MagicDNS 主機名稱（推薦）
ssh ac-4090@ac-4090.taile9e967.ts.net

# 或使用 -l 參數
ssh -l ac-4090 ac-4090.taile9e967.ts.net
```

> ⚠️ **注意**：直接用 IP `ssh ac-4090@100.109.200.56` 會被 Tailscale ACL 阻擋，必須使用 MagicDNS 主機名稱。

### 快速連線設定

在本地 `~/.ssh/config` 加入：

```
Host 4090
    HostName ac-4090.taile9e967.ts.net
    User ac-4090
```

之後只需輸入：

```bash
ssh 4090
```

---

## 環境設定

### 步驟 1：安裝 Miniconda

```bash
# 下載 Miniconda
wget https://repo.anaconda.com/miniconda/Miniconda3-latest-Linux-x86_64.sh

# 安裝（遵循提示）
bash Miniconda3-latest-Linux-x86_64.sh

# 重新載入 shell
source ~/.bashrc

# 驗證安裝
conda --version
```

### 步驟 2：建立 GPU 專用環境

```bash
# 建立環境（Python 3.10 相容性較佳）
conda create -n gpu python=3.10 -y

# 啟用環境
conda activate gpu
```

### 步驟 3：安裝 ML / GPU 套件

```bash
# PyTorch + CUDA 12.1（穩定版）
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121

# 或 CUDA 12.8（最新版）
# pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu128

# Power-CLIP 專案依賴
pip install sentence-transformers numpy pandas tqdm matplotlib

# 其他常用套件
pip install transformers accelerate datasets
pip install jupyter jupyterlab  # 如需遠端 notebook

# TensorFlow GPU（可選）
# pip install tensorflow

# Diffusers / ComfyUI（可選）
# pip install diffusers
```

### 步驟 4：驗證 GPU

```bash
python -c "import torch; print(f'CUDA: {torch.cuda.is_available()}, Device: {torch.cuda.get_device_name(0)}')"
```

預期輸出：
```
CUDA: True, Device: NVIDIA GeForce RTX 4090 D
```

---

## 資料路徑對應

### 本地 (Mac / Google Drive) → 遠端 (4090)

| 用途 | 本地路徑 | 遠端路徑 |
|------|----------|----------|
| 專案目錄 | `/Users/user/Desktop/REDD Dataset/Power_CLIP` | `~/Power_CLIP` |
| 資料集目錄 | `/Users/user/Desktop/REDD Dataset/` | `~/data/` |
| 主要資料檔 | `output_old/output/combination_labels.jsonl` | `~/data/combination_labels.jsonl` |
| 模型權重 | `output_old/output/*.pth` | `~/Power_CLIP/models/` |
| Google Drive | `/content/drive/MyDrive/Reserach/REDD_Dataset` | `~/data/` |

### Notebook 路徑修改

在 Colab 中：
```python
BASE_PATH = '/content/drive/MyDrive/Reserach/REDD_Dataset'
```

在 4090 中：
```python
BASE_PATH = '/home/ac-4090/data'
```

### 需要上傳的主要檔案

| 檔案 | 大小 | 說明 |
|------|------|------|
| `combination_labels.jsonl` | 270 MB | 主要訓練資料 |
| `aggregate_signals.csv` | 34 MB | 聚合信號資料 |
| `power_clip_model.pth` | 460 KB | 預訓練模型 |
| `multilabel_clip.pth` | 622 KB | 多標籤模型 |

---

## 檔案傳輸

### 使用 scp 上傳

```bash
# 上傳整個專案
scp -r "/Users/user/Desktop/REDD Dataset/Power_CLIP" ac-4090@ac-4090.taile9e967.ts.net:~/

# 上傳資料集（建立 data 目錄）
ssh ac-4090@ac-4090.taile9e967.ts.net "mkdir -p ~/data"
scp "/Users/user/Desktop/REDD Dataset/output_old/output/combination_labels.jsonl" ac-4090@ac-4090.taile9e967.ts.net:~/data/
scp "/Users/user/Desktop/REDD Dataset/output_old/output/aggregate_signals.csv" ac-4090@ac-4090.taile9e967.ts.net:~/data/
```

### 使用 rsync 同步（推薦大檔案）

```bash
# 同步專案（增量更新，只傳輸變更）
rsync -avhP "/Users/user/Desktop/REDD Dataset/Power_CLIP/" ac-4090@ac-4090.taile9e967.ts.net:~/Power_CLIP/

# 同步資料集
rsync -avhP "/Users/user/Desktop/REDD Dataset/output_old/output/" ac-4090@ac-4090.taile9e967.ts.net:~/data/
```

### 下載檔案

```bash
# 下載訓練好的模型
scp ac-4090@ac-4090.taile9e967.ts.net:~/Power_CLIP/*.pth ./

# 下載結果目錄
scp -r ac-4090@ac-4090.taile9e967.ts.net:~/Power_CLIP/results ./
```

---

## VS Code Remote SSH

### 設定步驟

1. 安裝 VS Code 擴充套件：**Remote - SSH**

2. 按 `Cmd+Shift+P` → 輸入 `Remote-SSH: Connect to Host`

3. 選擇 `4090`（如已設定 `~/.ssh/config`）或輸入：
   ```
   ac-4090@ac-4090.taile9e967.ts.net
   ```

4. 開啟遠端資料夾：`~/Power_CLIP`

### 執行 Jupyter Notebook

在 VS Code Remote 中：

1. 安裝 Python 擴充套件（遠端）
2. 開啟 `.ipynb` 檔案
3. 選擇 Kernel：`gpu` (conda 環境)
4. 直接執行 cell

---

## tmux 長時間運行

SSH 斷線時訓練會繼續執行。

### 基本操作

```bash
# 安裝 tmux（如尚未安裝）
sudo apt install tmux -y

# 建立新 session
tmux new -s train

# 離開 session（訓練繼續）
# 按 Ctrl+B，然後按 D

# 重新連接
tmux attach -t train

# 列出所有 session
tmux ls

# 刪除 session
tmux kill-session -t train
```

### 分割視窗

```bash
# 水平分割
Ctrl+B, %

# 垂直分割
Ctrl+B, "

# 切換視窗
Ctrl+B, 方向鍵

# 關閉當前視窗
Ctrl+D
```

---

## Web UI / API（可選）

如需從 Mac 瀏覽器訪問遠端服務（ComfyUI、Jupyter Lab 等）。

### 方法 1：SSH Port Forwarding（簡單）

```bash
# 在本地執行，將遠端 8888 port 映射到本地
ssh -L 8888:localhost:8888 ac-4090@ac-4090.taile9e967.ts.net

# 在遠端啟動 Jupyter
jupyter lab --no-browser --port=8888

# 本地瀏覽器開啟 http://localhost:8888
```

### 方法 2：Cloudflare Tunnel（需公開域名）

```bash
# 在遠端安裝 cloudflared
wget https://github.com/cloudflare/cloudflared/releases/latest/download/cloudflared-linux-amd64 -O cloudflared
sudo mv cloudflared /usr/local/bin/
sudo chmod +x /usr/local/bin/cloudflared

# 建立隧道
cloudflared tunnel --url http://localhost:8888
```

---

## 訓練流程

### 完整步驟

```bash
# 1. 連線
ssh 4090

# 2. 啟用環境
conda activate gpu

# 3. 開啟 tmux
tmux new -s train

# 4. 進入專案目錄
cd ~/Power_CLIP

# 5. 執行訓練
python train.py

# 6. 離開 tmux（Ctrl+B, D）

# 7. 之後重新連線查看進度
ssh 4090
tmux attach -t train
```

### 執行 Notebook

```bash
# 轉換 notebook 為 Python 腳本
jupyter nbconvert --to script notebooks/Power_CLIP_vNext_Attribute.ipynb

# 或直接用 papermill 執行 notebook
pip install papermill
papermill notebooks/Power_CLIP_vNext_Attribute.ipynb output.ipynb
```

---

## 監控與管理

### GPU 監控

```bash
# 即時監控（每秒更新）
watch -n 1 nvidia-smi

# 簡潔輸出
nvidia-smi --query-gpu=name,memory.used,memory.total,utilization.gpu --format=csv

# 持續監控到檔案
nvidia-smi --query-gpu=timestamp,memory.used,utilization.gpu --format=csv -l 5 > gpu_log.csv
```

### 程序監控

```bash
# 查看 Python 程序
ps aux | grep python

# 查看 GPU 程序
nvidia-smi pmon -i 0

# 系統資源
htop
```

### 硬碟空間

```bash
# 檢查磁碟使用
df -h

# 檢查目錄大小
du -sh ~/data/*
```

---

## 常見問題

### 1. SSH 連線被拒絕

```
tailscale: tailnet policy does not permit you to SSH to this node
```

**解決**：使用 MagicDNS 主機名稱 `ac-4090.taile9e967.ts.net` 而非 IP。

### 2. Tailscale 未連線

```bash
# 檢查狀態
tailscale status

# 重新連線
tailscale up
```

### 3. CUDA out of memory

```python
# 降低 batch size
BATCH_SIZE = 16  # 原本 32

# 或使用 gradient accumulation
accumulation_steps = 4
```

### 4. Conda 指令找不到

```bash
source ~/.bashrc
# 或重新登入
```

### 5. 訓練中斷

如果忘記用 tmux，SSH 斷線會導致訓練中斷。

**預防**：永遠在 tmux 內執行長時間任務！

### 6. Permission denied

```bash
# 檢查檔案權限
ls -la ~/data/

# 修改權限
chmod 644 ~/data/combination_labels.jsonl
```

---

## 快速指令參考

```bash
# 連線
ssh 4090

# 啟用環境
conda activate gpu

# 檢查 GPU
nvidia-smi

# 開始訓練（在 tmux 內）
tmux new -s train
python train.py

# 上傳檔案（本地執行）
rsync -avhP local_file ac-4090@ac-4090.taile9e967.ts.net:~/

# 下載檔案（本地執行）
scp ac-4090@ac-4090.taile9e967.ts.net:~/remote_file ./
```
