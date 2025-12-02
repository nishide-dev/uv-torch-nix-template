# NixOS固有のセットアップガイド

このドキュメントは、NixOS環境でPyTorch/CUDA開発環境を構築する際の固有の設定要件と注意点を説明します。

## 🎯 NixOSでのPyTorch/CUDA環境の特殊性

NixOSは標準的なLinux（FHS: Filesystem Hierarchy Standard）とは異なる独自のファイルシステム構造を持つため、PyTorchのようなプリビルドバイナリ（Wheel）を動作させるには特別な設定が必要です。

### 主な課題と解決策

| 課題 | 原因 | 解決策 |
|------|------|--------|
| GPUドライバーが見つからない | `/run/opengl-driver/lib`がLD_LIBRARY_PATHにない | shellHookで明示的にパス追加 |
| libstdc++.so.6が見つからない | PyTorchが期待するABI互換性 | `stdenv.cc.cc.lib`を優先的にロード |
| 未パッチバイナリが動かない | NixOSの非FHS構造 | nix-ldシムを使用 |
| flash-attn等のビルド失敗 | ビルド環境でCUDAが見えない | `--no-build-isolation`で実行 |

## 🔧 必須設定: /etc/nixos/configuration.nix

### NVIDIAドライバーの有効化

NixOS上でGPUを使用するには、システム全体でNVIDIAドライバーを有効化する必要があります：

```nix
# /etc/nixos/configuration.nix
{ config, pkgs, ... }:

{
  # ===================================
  # NVIDIA GPU設定（必須）
  # ===================================
  services.xserver.videoDrivers = [ "nvidia" ];

  hardware.opengl = {
    enable = true;
    driSupport = true;
    driSupport32Bit = true;  # 32bitアプリケーションのサポート（オプション）
  };

  hardware.nvidia = {
    # モジュラーカーネルを使用（推奨）
    modesetting.enable = true;

    # Power management（ノートPCの場合は有効化推奨）
    powerManagement.enable = false;
    powerManagement.finegrained = false;

    # オープンソースドライバー（実験的、通常はfalse）
    open = false;

    # nvidia-settings（GUI設定ツール）
    nvidiaSettings = true;

    # ドライバーバージョン（production, beta, legacy_xxx）
    # 最新の安定版を使用
    package = config.boot.kernelPackages.nvidiaPackages.stable;
  };

  # ===================================
  # nix-ld設定（未パッチバイナリ実行用）
  # ===================================
  programs.nix-ld.enable = true;
  programs.nix-ld.libraries = with pkgs; [
    stdenv.cc.cc.lib
    zlib
    glib
  ];
}
```

設定後、システムを再構築：

```bash
sudo nixos-rebuild switch
```

### 設定確認

```bash
# NVIDIAドライバーが正しくロードされているか確認
nvidia-smi

# /run/opengl-driver/libが存在するか確認
ls -la /run/opengl-driver/lib/libcuda.so*
```

期待される出力：

```
lrwxrwxrwx 1 root root 79 Dec  1 10:00 /run/opengl-driver/lib/libcuda.so -> /nix/store/xxxxx-nvidia-x11-550.90.07/lib/libcuda.so
lrwxrwxrwx 1 root root 79 Dec  1 10:00 /run/opengl-driver/lib/libcuda.so.1 -> /nix/store/xxxxx-nvidia-x11-550.90.07/lib/libcuda.so.1
```

## 📚 技術的背景: nix-ldの仕組み

### 問題: 未パッチバイナリとNixOS

PyTorchのWheelは、標準的なLinux（FHS）を前提にビルドされています：

```bash
# PyTorchが期待するパス（FHS）
/lib64/ld-linux-x86-64.so.2         # 動的リンカー
/usr/lib/x86_64-linux-gnu/libstdc++.so.6  # 標準C++ライブラリ
/usr/local/cuda/lib64/libcudart.so  # CUDA Runtime
```

NixOSでは、すべてのパッケージが`/nix/store`に隔離されているため、これらのパスは存在しません：

```bash
# NixOSの実際のパス
/nix/store/xxxxx-glibc-2.38/lib/ld-linux-x86-64.so.2
/nix/store/yyyyy-gcc-13.2.0-lib/lib/libstdc++.so.6
/nix/store/zzzzz-cuda-12.4/lib/libcudart.so
```

### 解決策: nix-ldシムとLD_LIBRARY_PATH

このテンプレートは2段階のアプローチで解決します：

#### 1. nix-ldによる動的リンカーの提供

```nix
# flake.nix内
NIX_LD_LIBRARY_PATH = pkgs.lib.makeLibraryPath (systemDependencies ++ cudaLibs);
NIX_LD = pkgs.lib.fileContents "${pkgs.stdenv.cc}/nix-support/dynamic-linker";
```

これにより、未パッチのバイナリが動的リンカーを見つけられるようになります。

#### 2. LD_LIBRARY_PATHによるライブラリ解決

```bash
# shellHook内
export LD_LIBRARY_PATH="${pkgs.lib.makeLibraryPath (...)}/run/opengl-driver/lib:..."
```

**重要**: `/run/opengl-driver/lib`を含めることで、`libcuda.so`（GPUドライバー）が見つかるようになります。

#### 3. libstdc++の優先的ロード

```bash
# ABI互換性のため、libstdc++を最優先
export LD_LIBRARY_PATH="${pkgs.stdenv.cc.cc.lib}/lib:$LD_LIBRARY_PATH"
```

PyTorchが期待するGCC ABIバージョンのlibstdc++を確実にロードします。

## 🔨 flash-attn等のビルドが必要なパッケージ

### 問題: ビルド分離環境でCUDAが見えない

uvは通常、ビルド時に分離された環境を使用します。この環境では、CUDA_HOME等の環境変数が見えません。

### 解決策: --no-build-isolation

```bash
# ビルド分離を無効化してインストール
uv add flash-attn --no-build-isolation

# 他のCUDA依存パッケージも同様
uv add xformers --no-build-isolation
uv add apex --no-build-isolation
```

このテンプレートのflake.nixには、ビルドに必要なツールが含まれています：

```nix
systemDependencies = with pkgs; [
  stdenv.cc.cc.lib
  ninja          # flash-attnのビルドに必須
  cmake          # ビルドシステム
  which          # ビルドスクリプトが使用
  pkg-config     # ライブラリ検出
  # ...
];
```

また、ビルド用のフラグも設定されています：

```bash
export EXTRA_LDFLAGS="-L/lib -L${cudaPackages.cudatoolkit}/lib"
export EXTRA_CCFLAGS="-I/usr/include -I${cudaPackages.cudatoolkit}/include"
```

## 🐛 トラブルシューティング

### GPU not available (CUDA available: False)

**症状**: `torch.cuda.is_available()`が`False`を返す

**診断手順**:

1. **NVIDIAドライバーの確認**
   ```bash
   nvidia-smi
   ```
   エラーが出る場合、`/etc/nixos/configuration.nix`の設定を確認

2. **libcuda.soの存在確認**
   ```bash
   ls -la /run/opengl-driver/lib/libcuda.so*
   ```
   存在しない場合、`hardware.opengl.enable = true`が設定されているか確認

3. **LD_LIBRARY_PATHの確認**
   ```bash
   echo $LD_LIBRARY_PATH | grep opengl-driver
   ```
   含まれていない場合、direnvを再読み込み：
   ```bash
   direnv reload
   ```

4. **PyTorchビルドの確認**
   ```bash
   uv run python -c "import torch; print(torch.version.cuda)"
   ```
   `None`の場合、CPU版がインストールされています。`pyproject.toml`の`[[tool.uv.index]]`設定を確認

### libcuda.so.1: cannot open shared object file

**症状**: `libcuda.so.1`が見つからないエラー

**原因**: NVIDIAドライバーが正しくインストールされていない、または`/run/opengl-driver/lib`がLD_LIBRARY_PATHにない

**対処**:

```bash
# ドライバーの再インストール
sudo nixos-rebuild switch

# direnvの再読み込み
direnv reload

# 手動で確認
export LD_LIBRARY_PATH="/run/opengl-driver/lib:$LD_LIBRARY_PATH"
uv run python -c "import torch; print(torch.cuda.is_available())"
```

### libstdc++.so.6: version GLIBCXX_3.4.30 not found

**症状**: libstdc++のABIバージョンエラー

**原因**: PyTorchが期待するGCCバージョンのlibstdc++が見つからない

**対処**: このテンプレートは自動的に対処していますが、問題が発生した場合：

```bash
# 現在のlibstdc++の確認
ldd $(uv run python -c "import torch; print(torch.__file__)")

# libstdc++.so.6の検索
find /nix/store -name "libstdc++.so.6" 2>/dev/null | head -5

# 手動でLD_LIBRARY_PATHに追加（テスト用）
export LD_LIBRARY_PATH="/nix/store/xxxxx-gcc-lib/lib:$LD_LIBRARY_PATH"
```

### flash-attnのビルドエラー

**症状**: `ModuleNotFoundError: No module named 'torch'` during build

**原因**: ビルド分離環境でtorchが見えない

**対処**:

```bash
# --no-build-isolationを使用
uv add flash-attn --no-build-isolation

# CUDA_HOMEが設定されているか確認
echo $CUDA_HOME

# 手動ビルドテスト
uv run python -m pip install flash-attn --no-build-isolation -v
```

### Nix develop時のエラー

**症状**: `error: experimental Nix feature 'nix-command' is disabled`

**対処**: Nix Flakeを有効化

```bash
# ~/.config/nix/nix.conf または /etc/nix/nix.conf
experimental-features = nix-command flakes
```

または、一時的に有効化：

```bash
nix --extra-experimental-features 'nix-command flakes' develop
```

## 📊 パフォーマンスと最適化

### CUDAバージョンの選択

異なるCUDAバージョンは異なるパフォーマンス特性を持ちます：

| CUDA Version | 推奨環境 | 特徴 |
|--------------|---------|------|
| 11.8 | 古いGPU (Maxwell以降) | 広い互換性、安定 |
| 12.1 | Ampere以降 | 新機能、最適化 |
| 12.4 | Ada Lovelace (RTX 40xx) | 最新機能、FP8サポート |

### nixpkgsのCUDA対応状況の確認

```bash
# 利用可能なCUDAバージョンを確認
nix search nixpkgs cudaPackages

# 特定バージョンの詳細
nix eval nixpkgs#cudaPackages_12_1.cudatoolkit.version
```

### キャッシュの活用

NixはバイナリキャッシュからCUDAパッケージをダウンロードします：

```nix
# flake.nix内（自動的に設定済み）
config = {
  allowUnfree = true;  # CUDA関連パッケージはunfree
  cudaSupport = true;
};
```

## 🔗 参考リンク

- [NixOS Manual - NVIDIA](https://nixos.wiki/wiki/Nvidia)
- [Nixpkgs CUDA Support](https://github.com/NixOS/nixpkgs/blob/master/pkgs/development/cuda-modules/README.md)
- [nix-ld Documentation](https://github.com/Mic92/nix-ld)
- [PyTorch Installation Guide](https://pytorch.org/get-started/locally/)
- [uv PyTorch Integration](https://docs.astral.sh/uv/guides/integration/pytorch/)

## 💡 NixOSでのベストプラクティス

### 1. Flake Lockの管理

```bash
# 依存関係を更新
nix flake update

# 特定の入力のみ更新
nix flake lock --update-input nixpkgs
```

### 2. direnvとの統合

`.envrc`が自動生成されるため、プロジェクトディレクトリに入るだけで環境が有効化されます：

```bash
cd my-project
# 自動的にnix develop環境に入る
```

### 3. 開発環境の共有

チームメンバーと同じ環境を共有：

```bash
# flake.lockをgitにコミット
git add flake.lock
git commit -m "Lock Nix dependencies"

# 他のメンバーは同じ環境を再現可能
git clone <repo>
cd <repo>
direnv allow
```

### 4. システム全体とプロジェクト環境の分離

- **システム全体**: NVIDIAドライバーのみ
- **プロジェクト環境**: CUDA Toolkit、cuDNN、PyTorch

この分離により、異なるプロジェクトで異なるCUDAバージョンを使用できます。

### 5. 環境変数のデバッグ

```bash
# 現在の環境変数を確認
printenv | grep -E "(CUDA|LD_LIBRARY|NIX_LD)"

# Nixシェルの詳細情報
nix develop --print-build-logs
```

## 🎓 深い理解のために

### NixOSの設計哲学

NixOSは「純粋な関数型パッケージ管理」を実現するため、すべてのパッケージを`/nix/store`に隔離します。これにより：

- **再現性**: 同じ入力から常に同じ出力
- **並行性**: 複数バージョンの共存が可能
- **安全性**: パッケージの更新が既存環境を破壊しない

しかし、この設計はFHSを前提とするプリビルドバイナリと相容れません。

### なぜPyTorch Wheelが動くのか？

このテンプレートが実現している3つの仕組み：

1. **nix-ld**: 動的リンカーのシム（`/lib64/ld-linux-x86-64.so.2`のエミュレート）
2. **LD_LIBRARY_PATH**: 共有ライブラリの検索パスを明示的に指定
3. **NIX_LD_LIBRARY_PATH**: nix-ldが参照するライブラリパス

これらにより、未パッチのPyTorch WheelがNixOS上で動作します。

### 代替アプローチ: nixpkgsのPyTorch

nixpkgsにもPyTorchパッケージがありますが、以下の理由でこのテンプレートはuvを使用します：

- **最新バージョン**: uvは公式PyTorchインデックスから最新版を取得
- **柔軟性**: プロジェクトごとに異なるバージョンを簡単に管理
- **エコシステム**: Pythonパッケージエコシステムと完全互換

## 📝 まとめ

NixOS上でPyTorch/CUDA環境を構築するには：

1. ✅ `/etc/nixos/configuration.nix`でNVIDIAドライバーを有効化
2. ✅ `programs.nix-ld.enable = true`を設定
3. ✅ このテンプレートを使用（nix-ld、LD_LIBRARY_PATH自動設定）
4. ✅ `direnv allow`で環境を有効化
5. ✅ `uv sync`でPyTorchをインストール

これで、再現可能で柔軟なPyTorch開発環境が完成します。
