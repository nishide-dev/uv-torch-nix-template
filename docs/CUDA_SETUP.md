# CUDA/PyTorch環境セットアップガイド

このドキュメントは、`uv-torch-nix-template`を使用したPyTorch/CUDA開発環境のセットアップ方法を説明します。

## 🎯 概要

このテンプレートは、Nixを使用してCUDA環境を完全に管理し、再現可能なPyTorch開発環境を提供します。

### 主な特徴

- **完全な再現性**: CUDAバージョン、cuDNN、PyTorchを固定
- **システムレベル管理**: Nixによる依存関係の完全な管理
- **direnv統合**: プロジェクトディレクトリに入ると自動アクティベート

## 🚀 クイックスタート

### 1. ベーステンプレートでプロジェクト生成

まず、`uv-nix-template`でベースプロジェクトを作成します：

```bash
uvx copier copy --trust gh:nishide-dev/uv-nix-template my-torch-project
cd my-torch-project
```

### 2. PyTorch/CUDA拡張を適用

次に、このPyTorch拡張テンプレートを適用します：

```bash
uvx copier copy --trust gh:nishide-dev/uv-torch-nix-template .
```

対話的に以下の質問に答えます：

- **PyTorchバージョン**: 例: `2.5.1`, `2.4.1`
- **CUDAバージョン**: 例: `12.4`, `12.1`, `11.8`
- **cuDNNバージョン**: 例: `9.1.0`, `8.9.7`
- **PyTorch CUDA architecture**: 例: `cu124`, `cu121`（空白で自動生成）
- **torchvision**: 必要に応じて `yes`
- **torchaudio**: 必要に応じて `yes`

### 3. Nix環境を構築

```bash
# direnvを許可（初回のみ）
direnv allow

# または手動でNix環境に入る
nix develop
```

### 4. PyTorchをインストール

CUDA対応PyTorchをインストールします：

```bash
# PyTorch + torchvisionをインストール（CUDA 12.4の例）
uv add torch==2.5.1+cu124 torchvision==0.20.1+cu124 --index-url https://download.pytorch.org/whl/cu124

# または、CUDA 12.1の場合
uv add torch==2.5.1+cu121 torchvision==0.20.1+cu121 --index-url https://download.pytorch.org/whl/cu121

# torchaudioも必要な場合
uv add torchaudio==2.5.1+cu124 --index-url https://download.pytorch.org/whl/cu124
```

**注意**: PyTorchのCUDAビルドは公式のPyPIではなく、PyTorchの専用インデックスからダウンロードする必要があります。

### 5. 動作確認

```bash
# PyTorchとCUDAの確認
uv run python -c "
import torch
print(f'PyTorch version: {torch.__version__}')
print(f'CUDA available: {torch.cuda.is_available()}')
print(f'CUDA version: {torch.version.cuda}')
print(f'cuDNN version: {torch.backends.cudnn.version()}')
if torch.cuda.is_available():
    print(f'GPU device: {torch.cuda.get_device_name(0)}')
"
```

期待される出力例：

```
PyTorch version: 2.5.1+cu124
CUDA available: True
CUDA version: 12.4
cuDNN version: 9001
GPU device: NVIDIA GeForce RTX 4090
```

## 📦 依存関係の管理

### 一般的なPyTorchライブラリの追加

```bash
# データサイエンス系
uv add numpy pandas scikit-learn matplotlib seaborn

# Deep Learning系
uv add transformers accelerate datasets

# コンピュータビジョン系
uv add opencv-python pillow albumentations

# 開発ツール
uv add --dev jupyter ipython tensorboard
```

### PyTorchバージョンの更新

```bash
# 新しいバージョンにアップグレード
uv add torch==2.6.0+cu124 --index-url https://download.pytorch.org/whl/cu124
```

## 🔧 Nixによる環境カスタマイズ

### CUDAバージョンの変更

`flake.nix`を編集してCUDAバージョンを変更できます：

```nix
# 例: CUDA 12.1に変更
cudaVersion = "12_1";
cudaPackages = pkgs.cudaPackages_12_1;
```

変更後、環境を再構築：

```bash
nix flake update
direnv reload
```

### 追加のCUDAライブラリ

`flake.nix`の`cudaLibs`セクションに追加：

```nix
cudaLibs = with cudaPackages; [
  cuda_cudart
  cuda_nvcc
  cudnn
  nccl          # 追加: 分散学習用
  cutlass       # 追加: CUDA C++テンプレートライブラリ
];
```

## 🐛 トラブルシューティング

### CUDA not available

**症状**: `torch.cuda.is_available()`が`False`を返す

**原因と対処**:

1. **GPUドライバーの確認**
   ```bash
   nvidia-smi
   ```
   正しく表示されない場合、NVIDIAドライバーをインストール

2. **CUDA互換性の確認**
   - ドライバーバージョンとCUDAバージョンの互換性を確認
   - CUDA 12.4はドライバー525.60.13以上が必要

3. **PyTorchビルドの確認**
   ```bash
   uv run python -c "import torch; print(torch.version.cuda)"
   ```
   `None`の場合、CPU版がインストールされています

### LD_LIBRARY_PATH エラー

**症状**: `libcudart.so.12` などが見つからない

**対処**:

```bash
# Nix環境内で実行
echo $LD_LIBRARY_PATH

# パスが設定されていない場合、direnvを再読み込み
direnv reload
```

### Out of Memory (OOM)

**症状**: GPUメモリ不足エラー

**対処**:

```python
# バッチサイズを減らす
batch_size = 16  # → 8

# 混合精度学習を使用
from torch.cuda.amp import autocast, GradScaler
scaler = GradScaler()

with autocast():
    output = model(input)
```

### Nixビルドエラー

**症状**: `nix develop`でエラー

**対処**:

1. **キャッシュをクリア**
   ```bash
   nix-collect-garbage
   ```

2. **Flakeを更新**
   ```bash
   nix flake update
   ```

3. **unfreeパッケージの許可確認**
   `flake.nix`で`allowUnfree = true`が設定されているか確認

## 📊 パフォーマンス最適化

### cuDNN Benchmark

```python
import torch
torch.backends.cudnn.benchmark = True  # 最初の実行は遅いが、以降高速化
```

### データローダーの最適化

```python
from torch.utils.data import DataLoader

loader = DataLoader(
    dataset,
    batch_size=32,
    num_workers=4,      # CPUコア数に応じて調整
    pin_memory=True,    # GPU転送を高速化
    prefetch_factor=2,  # プリフェッチバッファ
)
```

### 混合精度学習

```python
from torch.cuda.amp import autocast, GradScaler

scaler = GradScaler()

for data, target in loader:
    optimizer.zero_grad()

    with autocast():
        output = model(data)
        loss = criterion(output, target)

    scaler.scale(loss).backward()
    scaler.step(optimizer)
    scaler.update()
```

## 🔗 参考リンク

- [PyTorch公式ドキュメント](https://pytorch.org/docs/stable/index.html)
- [CUDA Toolkit Documentation](https://docs.nvidia.com/cuda/)
- [Nix CUDA Support](https://nixos.wiki/wiki/CUDA)
- [uv Documentation](https://docs.astral.sh/uv/)

## 💡 ベストプラクティス

1. **バージョン固定**: `uv.lock`で依存関係を固定し、チーム全体で同じ環境を共有
2. **GPU監視**: `nvidia-smi`や`nvtop`でGPU使用率を監視
3. **実験管理**: Weights & BiasesやMLflowで実験を追跡
4. **型チェック**: `uv run ty check`で型安全性を確保
5. **テスト**: GPUコードもpytestでテスト可能

```python
import pytest
import torch

@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA not available")
def test_cuda_operation():
    x = torch.randn(100, 100).cuda()
    y = torch.matmul(x, x)
    assert y.is_cuda
```
