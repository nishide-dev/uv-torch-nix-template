# uv-torch-nix-template

**PyTorch/CUDA開発環境の拡張テンプレート**

[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![uv](https://img.shields.io/endpoint?url=https://raw.githubusercontent.com/astral-sh/uv/main/assets/badge/v0.json)](https://github.com/astral-sh/uv)

[uv-nix-template](https://github.com/nishide-dev/uv-nix-template)に**PyTorch**と**CUDA**環境を追加する拡張テンプレートです。Nixによる完全な環境再現性を提供し、**NixOS**でも完全に動作します。

## ✨ 特徴

### 🔥 PyTorch/CUDA統合

- **CUDAバージョン管理**: Nixで固定されたCUDA環境
- **PyTorchバージョン選択**: 任意のPyTorchバージョンを指定可能
- **自動インデックス設定**: `pyproject.toml`にPyTorch専用インデックスを自動生成
- **cuDNN統合**: cuDNNも自動でセットアップ
- **torchvision/torchaudio**: オプションで追加可能

### 🐧 NixOS完全対応

- **nix-ldサポート**: 未パッチのPyTorch Wheelが動作
- **GPU自動検出**: `/run/opengl-driver/lib`を自動で`LD_LIBRARY_PATH`に追加
- **ビルド環境完備**: flash-attn等のコンパイルに必要なツール（ninja, cmake等）を標準装備
- **`--no-build-isolation`対応**: CUDA依存パッケージのビルドをサポート

### 🔒 完全な再現性

- **システムレベル管理**: CUDA、cuDNN、PyTorchすべてバージョン固定
- **決定論的ビルド**: チーム全体で完全に同一の環境
- **Nix + direnv**: プロジェクトに入ると自動でGPU環境アクティベート
- **ABI互換性保証**: libstdc++を優先的にロードしてABI問題を回避

### 🚀 高速開発体験

- **uv統合**: 高速な依存関係解決（pip/poetryより10-100倍高速）
- **ベーステンプレート継承**: uv-nix-templateのすべての機能を継承
- **AI開発支援**: Claude Code CLI自動インストール

## 🎯 対象ユーザー

- GPUを使ったディープラーニング開発者
- PyTorchで研究・実験を行うMLエンジニア
- 再現可能なGPU環境が必要なチーム
- CUDAバージョン管理に悩んでいる方

## 🚀 クイックスタート

### 前提条件

1. **Nix + direnv**: [uv-nix-template](https://github.com/nishide-dev/uv-nix-template#前提条件)のセットアップを完了
2. **NVIDIAドライバー**: GPUドライバーがインストール済みであること
   ```bash
   nvidia-smi  # 動作確認
   ```
3. **NixOSユーザー**: `/etc/nixos/configuration.nix`でNVIDIAドライバーとnix-ldを有効化（詳細は[docs/NIX_SETUP.md](docs/NIX_SETUP.md)参照）

### 使用方法

#### ステップ1: ベーステンプレートでプロジェクト生成

```bash
# uv-nix-templateでベースプロジェクトを作成
uvx copier copy --trust gh:nishide-dev/uv-nix-template my-torch-project
cd my-torch-project
```

#### ステップ2: PyTorch/CUDA拡張を適用

```bash
# PyTorch拡張を追加適用
uvx copier copy --trust gh:nishide-dev/uv-torch-nix-template .
```

対話的に以下を設定：

- **PyTorchバージョン**: 例: `2.5.1`
- **CUDAバージョン**: 例: `12.4`
- **cuDNNバージョン**: 例: `9.1.0`
- **PyTorch CUDA architecture**: 例: `cu124`（空白で自動生成）
- **torchvision**: 必要なら `yes`
- **torchaudio**: 必要なら `yes`

#### ステップ3: 環境構築とPyTorchインストール

```bash
# Nix環境を構築（direnvが自動でアクティベート）
direnv allow

# 依存関係を同期（pyproject.tomlに自動設定されたPyTorchインデックスを使用）
uv sync

# 動作確認
uv run python -c "
import torch
print(f'PyTorch: {torch.__version__}')
print(f'CUDA available: {torch.cuda.is_available()}')
if torch.cuda.is_available():
    print(f'GPU: {torch.cuda.get_device_name(0)}')
"
```

**注**: テンプレート生成時に指定したバージョンに基づき、`pyproject.toml`に自動的にPyTorchインデックスが設定されます。

期待される出力：

```
PyTorch: 2.5.1+cu124
CUDA available: True
GPU: NVIDIA GeForce RTX 4090
```

## 📝 生成時の質問

| 質問 | 説明 | デフォルト | 例 |
|-----|------|-----------|-----|
| `pytorch_version` | PyTorchバージョン | `2.5.1` | `2.4.1`, `2.3.1` |
| `cuda_version` | CUDAバージョン | `12.4` | `12.1`, `11.8` |
| `cudnn_version` | cuDNNバージョン | `9.1.0` | `8.9.7`, `8.9.2` |
| `pytorch_cuda_arch` | PyTorch CUDA architecture | 自動生成 | `cu124`, `cu121`, `cu118` |
| `use_torchvision` | torchvisionを含めるか | `true` | - |
| `use_torchaudio` | torchaudioを含めるか | `false` | - |
| `additional_cuda_libs` | 追加のCUDAライブラリ | なし | `nccl,cutlass` |

## 📂 拡張されるファイル

```
my-torch-project/
├── flake.nix                 # CUDA環境が追加される（nix-ld、LD_LIBRARY_PATH設定含む）
├── pyproject.toml            # PyTorchインデックスが自動設定される
├── docs/
│   ├── NIX_SETUP.md          # NixOS固有の設定ガイド（新規・必読）
│   └── CUDA_SETUP.md         # GPU環境セットアップ全般（新規）
└── .envrc                    # direnv設定（変更なし）
```

## 💡 開発ワークフロー例

```bash
# プロジェクト生成
uvx copier copy --trust gh:nishide-dev/uv-nix-template ml-experiment
cd ml-experiment

# PyTorch拡張を適用
uvx copier copy --trust gh:nishide-dev/uv-torch-nix-template .

# 環境構築
direnv allow

# PyTorchインストール（pyproject.tomlに自動設定されたインデックスを使用）
uv sync

# よく使うライブラリを追加
uv add transformers accelerate datasets
uv add --dev jupyter tensorboard

# flash-attn等のビルドが必要なパッケージ（NixOS環境に最適化）
uv add flash-attn --no-build-isolation

# 開発開始
uv run jupyter lab

# テスト実行（GPU使用）
uv run pytest tests/
```

## 🔧 カスタマイズ

### CUDAバージョンの変更

プロジェクト生成後に`flake.nix`を直接編集可能：

```nix
# CUDA 12.1に変更
cudaVersion = "12_1";
cudaPackages = pkgs.cudaPackages_12_1;
```

変更後：

```bash
nix flake update
direnv reload
```

### 追加のCUDAライブラリ

分散学習（NCCL）やその他のCUDAライブラリを追加：

```nix
cudaLibs = with cudaPackages; [
  cuda_cudart
  cuda_nvcc
  cudnn
  nccl          # 追加
];
```

## 🐛 トラブルシューティング

詳細は以下のドキュメントを参照してください：
- **NixOSユーザー**: [docs/NIX_SETUP.md](docs/NIX_SETUP.md) - NixOS固有の設定とトラブルシューティング
- **一般的な問題**: [docs/CUDA_SETUP.md](docs/CUDA_SETUP.md) - PyTorch/CUDA環境全般

### よくある問題

**Q: `torch.cuda.is_available()`が`False`を返す（NixOS）**

A: NixOS固有の設定を確認：
1. `/etc/nixos/configuration.nix`でNVIDIAドライバーが有効化されているか
2. `/run/opengl-driver/lib/libcuda.so*`が存在するか
3. `programs.nix-ld.enable = true`が設定されているか
4. `direnv reload`で環境を再読み込み

詳細: [docs/NIX_SETUP.md](docs/NIX_SETUP.md)

**Q: `libcuda.so.1: cannot open shared object file`**

A: NixOSドライバー設定の問題です：
```bash
# ドライバーの再インストール
sudo nixos-rebuild switch
# 環境の再読み込み
direnv reload
```

**Q: flash-attnのビルドエラー**

A: ビルド分離を無効化してインストール：
```bash
uv add flash-attn --no-build-isolation
```

**Q: Nixビルドでエラーが出る**

A:
```bash
nix-collect-garbage
nix flake update
```

**Q: GPUメモリ不足**

A: バッチサイズを減らすか、混合精度学習を使用：
```python
from torch.cuda.amp import autocast, GradScaler
```

## 📚 ドキュメント

### このテンプレートのドキュメント

- **[NIX_SETUP.md](docs/NIX_SETUP.md)** - NixOS固有の設定とトラブルシューティング（NixOSユーザー必読）
- **[CUDA_SETUP.md](docs/CUDA_SETUP.md)** - PyTorch/CUDA環境セットアップ全般

### 外部ドキュメント

- [ベーステンプレート](https://github.com/nishide-dev/uv-nix-template) - uv-nix-templateのドキュメント
- [PyTorch公式ドキュメント](https://pytorch.org/docs/stable/index.html)
- [uv PyTorch統合ガイド](https://docs.astral.sh/uv/guides/integration/pytorch/)
- [CUDA Toolkit](https://docs.nvidia.com/cuda/)
- [NixOS NVIDIA Wiki](https://nixos.wiki/wiki/Nvidia)

## 🔄 テンプレートの更新

ベーステンプレートまたは拡張テンプレートが更新された場合：

```bash
# ベーステンプレートの更新
uvx copier update --trust

# PyTorch拡張の更新（別途適用）
uvx copier copy --trust gh:nishide-dev/uv-torch-nix-template .
```

## 🎯 ベーステンプレートとの関係

このテンプレートは**拡張テンプレート**です：

```
uv-nix-template (ベース)
├── Python環境
├── uv
├── Nix + direnv
├── Ruff/ty/pytest
└── GitHub Actions

     ↓ 拡張適用

uv-torch-nix-template (拡張)
├── 上記すべて +
├── CUDA環境
├── cuDNN
└── PyTorch設定
```

## 🤝 関連プロジェクト

- **[uv-nix-template](https://github.com/nishide-dev/uv-nix-template)**: ベーステンプレート
- **[uv](https://github.com/astral-sh/uv)**: 高速Pythonパッケージマネージャー
- **[Copier](https://copier.readthedocs.io/)**: テンプレートエンジン

## 📊 サポートするバージョン

### CUDA

Nixpkgsでサポートされているバージョン（変動するため`nix search nixpkgs cudaPackages`で確認）：
- CUDA 12.1（推奨）
- CUDA 11.8（安定）
- その他のバージョン（nixpkgsの対応状況に依存）

**注**: CUDA 12.4等の新しいバージョンはnixpkgsから削除されている場合があります。詳細は[docs/NIX_SETUP.md](docs/NIX_SETUP.md)参照。

### PyTorch

任意のバージョンを指定可能（PyTorchの公式リリースに依存）：
- 2.5.x（最新）
- 2.4.x
- 2.3.x
- それ以前のバージョン

### 推奨バージョン組み合わせ

| PyTorch | CUDA | cuDNN | GPU Architecture |
|---------|------|-------|------------------|
| 2.5.1   | 12.4, 12.1 | 9.1.0 | RTX 40xx (Ada) |
| 2.4.1   | 12.1, 11.8 | 8.9.7 | RTX 30xx (Ampere) |
| 2.3.1   | 12.1, 11.8 | 8.9.2 | 汎用 |

## 📄 ライセンス

MIT License

## 💬 コントリビューション

Issue/PRは大歓迎です！

特に以下の改善案を募集中：
- 他のディープラーニングフレームワーク（JAX、TensorFlowなど）への対応
- マルチGPU設定のベストプラクティス
- Docker統合
- CI/CDでのGPUテスト

## 🙏 謝辞

このテンプレートは以下のプロジェクトに依存しています：
- [uv](https://github.com/astral-sh/uv) by Astral
- [Nix](https://nixos.org/)
- [PyTorch](https://pytorch.org/)
- [Copier](https://copier.readthedocs.io/)
