# uv-torch-nix-template

**PyTorch/CUDA開発環境テンプレート**

[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![uv](https://img.shields.io/endpoint?url=https://raw.githubusercontent.com/astral-sh/uv/main/assets/badge/v0.json)](https://github.com/astral-sh/uv)

[uv-nix-template](https://github.com/nishide-dev/uv-nix-template)ベースの**PyTorch/CUDA特化型テンプレート**です。Nixによる完全な環境再現性を提供し、**Ubuntu/Debian等のLinux環境で動作**します（NixOSにも対応）。

## ✨ 特徴

### 🔥 PyTorch/CUDA統合

- **プリセットシステム**: 16種類の検証済みPyTorch/CUDA組み合わせから選択可能
- **CUDAバージョン管理**: Nixで固定されたCUDA環境（CUDA 13.0/12.8/12.6/12.4/12.1/11.8をサポート）
- **PyTorchバージョン選択**: 最新版（2.9.0）から旧版まで対応
- **自動インデックス設定**: `pyproject.toml`にPyTorch専用インデックスを自動生成
- **cuDNN統合**: NVIDIA公式互換性マトリクスに基づいた正確なバージョン管理
- **torchvision/torchaudio**: オプションで追加可能

### 🔧 幅広いLinux環境に対応

- **Ubuntu/Debian等での動作**: 簡単なセットアップでCUDAを利用可能
- **GPU自動検出**: `/run/opengl-driver/lib`経由でNVIDIAドライバーにアクセス
- **ビルド環境完備**: flash-attn等のコンパイルに必要なツール（ninja, cmake等）を標準装備
- **`--no-build-isolation`対応**: CUDA依存パッケージのビルドをサポート
- **NixOSにも対応**: nix-ldによる完全なサポート

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

1. **Nix + direnv**

   Nixを使用すると、uvとPythonの両方が自動管理されます。

   ```bash
   # Nixのインストール
   sh <(curl -L https://nixos.org/nix/install) --daemon

   # Flakesを有効化
   mkdir -p ~/.config/nix
   echo "experimental-features = nix-command flakes" >> ~/.config/nix/nix.conf

   # direnvのインストール
   # macOS
   brew install direnv
   # Ubuntu/Debian
   sudo apt install direnv

   # シェル統合
   echo 'eval "$(direnv hook bash)"' >> ~/.bashrc  # or zsh
   ```

2. **NVIDIAドライバー**: GPUドライバーがインストール済みであること
   ```bash
   nvidia-smi  # 動作確認
   ```

3. **CUDA用システムセットアップ（Ubuntu/Debian等）**

   NixのCUDAアプリケーションは`/run/opengl-driver/lib`にlibcuda.soがあることを期待します。以下のコマンドでセットアップしてください：

   ```bash
   # /run/opengl-driver/libを作成してlibcuda.soをシンボリックリンク
   sudo mkdir -p /run/opengl-driver/lib
   sudo find /usr/lib -name 'libcuda.so*' -exec ln -s {} /run/opengl-driver/lib/ \;

   # 再起動後も永続化（推奨）
   echo "L /run/opengl-driver/lib/libcuda.so - - - - /usr/lib/x86_64-linux-gnu/libcuda.so" | \
     sudo tee /etc/tmpfiles.d/cuda-driver-for-nix.conf
   sudo systemd-tmpfiles --create
   ```

   **注**: libcuda.soのパスはディストリビューションにより異なる場合があります（`/usr/lib64`, `/lib64`等）。詳細は[Nix CUDA on non-NixOS systems](https://danieldk.eu/Nix-CUDA-on-non-NixOS-systems)を参照。

   <details>
   <summary>NixOSユーザーの場合（クリックして展開）</summary>

   `/etc/nixos/configuration.nix`でNVIDIAドライバーとnix-ldを有効化してください。詳細は[docs/NIX_SETUP.md](docs/NIX_SETUP.md)を参照。

   </details>

### 使用方法

```bash
# プロジェクトを生成
uvx copier copy --trust gh:nishide-dev/uv-torch-nix-template my-torch-project
cd my-torch-project
```

対話的に以下を設定：

- **PyTorch + CUDAプリセット**: 16種類から選択可能
  - `PyTorch 2.9.0 + CUDA 12.6` (最新・推奨) ← デフォルト
  - `PyTorch 2.9.0 + CUDA 13.0`
  - `PyTorch 2.8.0 + CUDA 12.6/12.8`
  - `PyTorch 2.7.1 + CUDA 12.6/11.8`
  - `PyTorch 2.6.0 + CUDA 12.4/11.8`
  - `PyTorch 2.5.1 + CUDA 12.4/12.1/11.8`
  - `PyTorch 2.4.1 + CUDA 12.4/12.1/11.8`
  - `PyTorch 2.3.1 + CUDA 12.1/11.8`
  - `Custom` (手動設定)
- **torchvision**: 必要なら `yes`
- **torchaudio**: 必要なら `yes`

**注**: プリセットを選択すると、PyTorch/CUDA/cuDNNのバージョンが自動的に設定されます。手動で設定したい場合は`Custom`を選択してください。

### CUDA動作確認

プロジェクト生成時に環境構築は自動的に完了しています（`uv venv`, `uv sync`実行済み）。CUDAが正しく動作するか確認しましょう：

```bash
# CUDA用システムセットアップ（初回のみ・非NixOSの場合）
# 前提条件のセクション参照

# Nix環境をアクティベート
direnv allow

# CUDA動作確認
uv run python -c "
import torch
print(f'PyTorch: {torch.__version__}')
print(f'CUDA available: {torch.cuda.is_available()}')
if torch.cuda.is_available():
    print(f'GPU: {torch.cuda.get_device_name(0)}')
"
```

期待される出力（PyTorch 2.9.0 + CUDA 12.6の場合）：

```
PyTorch: 2.9.0+cu126
CUDA available: True
GPU: NVIDIA GeForce RTX 4060 Ti
```

## 📝 生成時の質問

| 質問 | 説明 | デフォルト | 例 |
|-----|------|-----------|-----|
| `pytorch_cuda_preset` | PyTorch + CUDAプリセット | `PyTorch 2.9.0 + CUDA 12.6` | 16種類から選択 |
| `pytorch_version` | PyTorchバージョン（Customのみ） | `2.9.0` | `2.8.0`, `2.7.1` |
| `cuda_version` | CUDAバージョン（Customのみ） | `12.6` | `13.0`, `12.8`, `11.8` |
| `cudnn_version` | cuDNNバージョン（Customのみ） | `9.16.0` | `9.1.0`, `8.9.7` |
| `pytorch_cuda_arch` | PyTorch CUDA architecture | 自動生成 | `cu126`, `cu124`, `cu118` |
| `use_torchvision` | torchvisionを含めるか | `true` | - |
| `use_torchaudio` | torchaudioを含めるか | `false` | - |
| `additional_cuda_libs` | 追加のCUDAライブラリ | なし | `nccl,cutlass` |

## 📂 追加されるファイル

```
my-torch-project/
├── flake.nix                 # CUDA環境が追加される（LD_LIBRARY_PATH設定含む）
├── pyproject.toml            # PyTorchインデックスが自動設定される
├── docs/
│   ├── NIX_SETUP.md          # NixOS固有の設定ガイド（新規・NixOSユーザー向け）
│   └── CUDA_SETUP.md         # GPU環境セットアップ全般（新規）
└── .envrc                    # direnv設定（変更なし）
```

## 💡 開発ワークフロー例

```bash
# プロジェクト生成（PyTorch/CUDA環境込み）
uvx copier copy --trust gh:nishide-dev/uv-torch-nix-template ml-experiment
cd ml-experiment

# Nix環境をアクティベート
direnv allow

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

**Q: `torch.cuda.is_available()`が`False`を返す（非NixOS）**

A: `/run/opengl-driver/lib`のセットアップが必要です：

```bash
# libcuda.soのシンボリックリンクを作成
sudo mkdir -p /run/opengl-driver/lib
sudo find /usr/lib -name 'libcuda.so*' -exec ln -s {} /run/opengl-driver/lib/ \;

# 環境の再読み込み
direnv reload
```

詳細: [Nix CUDA on non-NixOS systems](https://danieldk.eu/Nix-CUDA-on-non-NixOS-systems)

**Q: `torch.cuda.is_available()`が`False`を返す（NixOS）**

A: NixOS固有の設定を確認：
1. `/etc/nixos/configuration.nix`でNVIDIAドライバーが有効化されているか
2. `/run/opengl-driver/lib/libcuda.so*`が存在するか
3. `programs.nix-ld.enable = true`が設定されているか
4. `direnv reload`で環境を再読み込み

詳細: [docs/NIX_SETUP.md](docs/NIX_SETUP.md)

**Q: `libcuda.so.1: cannot open shared object file`**

A: システムによって対処が異なります：

**非NixOS**:
```bash
# libcuda.soのパスを確認
find /usr/lib /lib -name 'libcuda.so*' 2>/dev/null

# 見つかったパスをシンボリックリンク
sudo ln -s /path/to/libcuda.so* /run/opengl-driver/lib/

# 環境の再読み込み
direnv reload
```

**NixOS**:
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
- [Nix CUDA on non-NixOS systems](https://danieldk.eu/Nix-CUDA-on-non-NixOS-systems) - 非NixOSでのCUDA設定ガイド
- [PyTorch公式ドキュメント](https://pytorch.org/docs/stable/index.html)
- [uv PyTorch統合ガイド](https://docs.astral.sh/uv/guides/integration/pytorch/)
- [CUDA Toolkit](https://docs.nvidia.com/cuda/)
- [NixOS NVIDIA Wiki](https://nixos.wiki/wiki/Nvidia)

## 🔄 テンプレートの更新

テンプレートが更新された場合：

```bash
# テンプレートの更新
uvx copier update --trust
```

## 🎯 ベーステンプレートとの関係

このテンプレートは**[uv-nix-template](https://github.com/nishide-dev/uv-nix-template)をベースにしたPyTorch/CUDA特化版**です。

独立したテンプレートとして使用でき、PyTorch/CUDA環境込みで1コマンドでプロジェクト生成できます：

```
uv-nix-template
├── Python環境
├── uv
├── Nix + direnv
├── Ruff/ty/pytest
└── GitHub Actions

     ↓ 派生

uv-torch-nix-template
├── 上記すべて +
├── PyTorch/CUDA環境
├── cuDNN
├── プリセットシステム
└── GPU自動検出
```

**PyTorch/CUDAが不要な場合**: [uv-nix-template](https://github.com/nishide-dev/uv-nix-template)を直接使用してください。

## 🤝 関連プロジェクト

- **[uv-nix-template](https://github.com/nishide-dev/uv-nix-template)**: ベーステンプレート
- **[uv](https://github.com/astral-sh/uv)**: 高速Pythonパッケージマネージャー
- **[Copier](https://copier.readthedocs.io/)**: テンプレートエンジン

## 📊 サポートするバージョン

### CUDA

Nixpkgsでサポートされているバージョン（変動するため`nix search nixpkgs cudaPackages`で確認）：
- CUDA 13.0（最新）
- CUDA 12.8/12.6/12.4/12.1（推奨）
- CUDA 11.8（安定・旧GPU対応）
- その他のバージョン（nixpkgsの対応状況に依存）

**注**: 新しいCUDAバージョンはnixpkgsから削除されている場合があります。詳細は[docs/NIX_SETUP.md](docs/NIX_SETUP.md)参照。

### PyTorch

任意のバージョンを指定可能（PyTorchの公式リリースに依存）：
- 2.9.x（最新）
- 2.8.x
- 2.7.x
- 2.6.x
- 2.5.x
- 2.4.x
- 2.3.x
- それ以前のバージョン

### 推奨バージョン組み合わせ

| PyTorch | CUDA | cuDNN | GPU Architecture | 備考 |
|---------|------|-------|------------------|------|
| 2.9.0   | 12.6, 13.0 | 9.16.0 | RTX 40xx/30xx | 最新・推奨 |
| 2.8.0   | 12.6, 12.8 | 9.16.0 | RTX 40xx/30xx | 安定版 |
| 2.7.1   | 12.6, 11.8 | 9.16.0/8.9.7 | RTX 40xx/30xx | 安定版 |
| 2.6.0   | 12.4, 11.8 | 9.1.0/8.9.7 | RTX 40xx/30xx | 安定版 |
| 2.5.1   | 12.4, 12.1, 11.8 | 9.1.0/8.9.7 | 汎用 | 後方互換性 |
| 2.4.1   | 12.4, 12.1, 11.8 | 8.9.7 | 汎用 | 旧プロジェクト |
| 2.3.1   | 12.1, 11.8 | 8.9.2 | 汎用 | 旧プロジェクト |

**参考**:
- [PyTorch公式バージョン情報](https://pytorch.org/get-started/previous-versions/)
- [NVIDIA cuDNN互換性マトリクス](https://docs.nvidia.com/deeplearning/cudnn/backend/latest/reference/support-matrix.html)

## 📄 ライセンス

MIT License

## 💬 コントリビューション

Issue/PRは大歓迎です！
