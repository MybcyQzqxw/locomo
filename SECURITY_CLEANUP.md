# 🔒 从 Git 历史中移除敏感信息

由于 HuggingFace Token 已经被提交到 Git 历史，需要以下步骤清理：

## ⚠️ 重要：先撤销 Token

**立即前往 HuggingFace 撤销旧 Token，生成新 Token！**
1. 访问：https://huggingface.co/settings/tokens
2. 撤销 `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`
3. 生成新 Token 并保存到 `.env.local`

## 方案 1：使用 git filter-repo（推荐）

```bash
# 1. 安装 git-filter-repo
pip install git-filter-repo

# 2. 备份仓库
cp -r ~/gitspace/mem/locomo ~/gitspace/mem/locomo_backup

# 3. 进入项目目录
cd ~/gitspace/mem/locomo

# 4. 创建要删除的密钥列表
cat > /tmp/secrets.txt << 'EOF'
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
EOF

# 5. 从所有历史中移除密钥
git filter-repo --replace-text /tmp/secrets.txt --force

# 6. 重新添加远程仓库
git remote add origin git@github.com:MybcyQzqxw/locomo.git

# 7. 强制推送（会覆盖远程历史）
git push origin --force --all
git push origin --force --tags

# 8. 清理临时文件
rm /tmp/secrets.txt
```

## 方案 2：使用 BFG Repo-Cleaner

```bash
# 1. 下载 BFG
wget https://repo1.maven.org/maven2/com/madgag/bfg/1.14.0/bfg-1.14.0.jar

# 2. 创建密钥替换文件
echo "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==>***REMOVED***" > replacements.txt

# 3. 清理历史
java -jar bfg-1.14.0.jar --replace-text replacements.txt ~/gitspace/mem/locomo

# 4. 进入仓库并清理
cd ~/gitspace/mem/locomo
git reflog expire --expire=now --all
git gc --prune=now --aggressive

# 5. 强制推送
git push origin --force --all
```

## 方案 3：删除整个历史重新开始（最简单但会丢失历史）

```bash
# 1. 删除 .git 目录
cd ~/gitspace/mem/locomo
rm -rf .git

# 2. 重新初始化
git init
git add .
git commit -m "Initial commit (cleaned)"

# 3. 连接远程仓库
git remote add origin git@github.com:MybcyQzqxw/locomo.git

# 4. 强制推送
git push origin main --force
```

## ✅ 清理完成后的验证

```bash
# 检查历史中是否还有 Token
git log -p | grep -i "hf_yXAXYaW"

# 如果没有任何输出，说明清理成功
```

## 📝 今后的安全实践

1. **永远不要提交密钥到 Git**
2. **使用 `.env.local` 保存本地密钥**（已加入 `.gitignore`）
3. **提交前检查**：`git diff` 确认没有密钥
4. **使用环境变量**：通过系统环境变量传递密钥
5. **启用 pre-commit hook**：自动检查密钥泄露

## 🔐 设置 Pre-commit Hook（可选）

```bash
# 创建 pre-commit hook
cat > .git/hooks/pre-commit << 'EOF'
#!/bin/bash
# 检查是否有密钥泄露

if git diff --cached | grep -iE "hf_[a-zA-Z0-9]{34}|sk-[a-zA-Z0-9]{48}"; then
    echo "❌ 错误：检测到 API Token！"
    echo "请从文件中移除密钥，使用 .env.local 替代"
    exit 1
fi
EOF

chmod +x .git/hooks/pre-commit
```

## 📚 相关链接

- [GitHub 密钥扫描文档](https://docs.github.com/en/code-security/secret-scanning)
- [git-filter-repo 文档](https://github.com/newren/git-filter-repo)
- [BFG Repo-Cleaner](https://rtyley.github.io/bfg-repo-cleaner/)
