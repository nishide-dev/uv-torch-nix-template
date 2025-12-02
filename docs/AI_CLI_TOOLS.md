# AI CLIツールガイド

Nix環境には、以下のAI CLIツールが自動的にインストールされます。

## 含まれるツール

### 1. Claude Code CLI

**公式**: https://claude.com/claude-code

Anthropic社のClaude AIをコマンドラインから利用できるツールです。

#### インストール方法

Nix環境では自動的にインストールされます（初回環境構築時）：

```bash
# 方法1: npm経由（デフォルト）
npm install -g @anthropic-ai/claude-code

# 方法2: curl経由（フォールバック）
curl -fsSL https://claude.ai/install.sh | bash
```

#### 基本的な使い方

```bash
# Claudeと対話
claude

# ファイルを指定してコード解析
claude analyze src/main.py

# コードレビュー
claude review

# ヘルプ
claude --help
```

#### 設定

初回起動時にAPIキーの設定が必要な場合があります：

```bash
# APIキーの設定
export ANTHROPIC_API_KEY="your-api-key"

# または設定ファイルに記載
# ~/.config/claude/config.json
```

### 2. OpenAI Codex

**公式**: https://openai.com/

OpenAI社のCodexをコマンドラインから利用できるツールです。

#### インストール方法

Nix環境では自動的にインストールされます（初回環境構築時）：

```bash
npm install -g @openai/codex
```

#### 基本的な使い方

```bash
# Codexと対話
codex

# コード生成
codex generate "Write a Python function to calculate fibonacci"

# ヘルプ
codex --help
```

#### 設定

初回起動時にAPIキーの設定が必要です：

```bash
# APIキーの設定
export OPENAI_API_KEY="your-api-key"

# または設定ファイルに記載
# ~/.config/openai/config.json
```

### 3. Gemini CLI

**公式**: https://ai.google.dev/

Google社のGemini AIをコマンドラインから利用できるツールです。

#### インストール方法

Nix環境では自動的にインストールされます（初回環境構築時）：

```bash
npm install -g @google/gemini-cli
```

#### 基本的な使い方

```bash
# Geminiと対話
gemini

# プロンプトを直接指定
gemini "Explain this code: $(cat main.py)"

# ヘルプ
gemini --help
```

#### 設定

初回起動時にAPIキーの設定が必要です：

```bash
# APIキーの設定
export GOOGLE_API_KEY="your-api-key"

# または
gemini config set apiKey your-api-key
```

## トラブルシューティング

### CLIツールが見つからない

```bash
# パスの確認
echo $PATH | grep npm-global

# 手動でパスを追加
export PATH="$HOME/.npm-global/bin:$PATH"

# 環境の再読み込み
direnv reload
```

### インストールが失敗する

```bash
# npm cacheをクリア
npm cache clean --force

# 手動で再インストール
npm install -g @anthropic-ai/claude-code
npm install -g @google/gemini-cli

# Claude Codeの代替インストール方法
curl -fsSL https://claude.ai/install.sh | bash
```

### npmグローバルインストール先の確認

```bash
# 設定の確認
echo $NPM_CONFIG_PREFIX
# 出力: /Users/yourusername/.npm-global

# インストール済みパッケージの確認
npm list -g --depth=0
```

### 権限エラーが出る

Nix環境では`NPM_CONFIG_PREFIX`が`~/.npm-global`に設定されているため、sudoなしでインストール可能です。

もしエラーが出る場合：

```bash
# ディレクトリの作成
mkdir -p ~/.npm-global

# 権限の確認
ls -la ~/.npm-global

# 環境変数の再設定
export NPM_CONFIG_PREFIX="$HOME/.npm-global"
export PATH="$HOME/.npm-global/bin:$PATH"
```

## アンインストール

```bash
# Claude Code CLIのアンインストール
npm uninstall -g @anthropic-ai/claude-code

# Gemini CLIのアンインストール
npm uninstall -g @google/gemini-cli

# すべてのグローバルパッケージを削除
rm -rf ~/.npm-global
```

## 追加のAI CLIツール

他のAI CLIツールも同様にインストール可能です：

```bash
# GitHub Copilot CLI
npm install -g @githubnext/github-copilot-cli

# OpenAI CLI
npm install -g openai-cli

# その他のツール
npm search ai-cli
```

これらを`flake.nix`のshellHookに追加することで、自動インストールできます。

## 参考資料

- [Claude Code 公式ドキュメント](https://claude.com/docs)
- [Gemini CLI GitHub](https://github.com/google/generative-ai-js)
- [npm グローバルパッケージ管理](https://docs.npmjs.com/downloading-and-installing-packages-globally)
