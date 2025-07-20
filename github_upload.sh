#!/bin/bash

# GitHub Repository Upload Script
# ================================

echo "🚀 GitHub'a Temporal Intelligence Framework yükleniyor..."

# Git remote ekle (URL'nizi buraya yazın)
GITHUB_USERNAME="soyluoltu"  # ← GitHub username'inizi buraya yazın
REPO_NAME="dynamic-temporal-learning-intelligence-model"

# Remote repository URL
REPO_URL="https://github.com/${GITHUB_USERNAME}/${REPO_NAME}.git"

echo "📡 Remote repository ekleniyor: $REPO_URL"
git remote add origin $REPO_URL

echo "📤 GitHub'a push ediliyor..."
git push -u origin main

echo "🎉 Upload tamamlandı!"
echo ""
echo "🔗 Repository URL: https://github.com/${GITHUB_USERNAME}/${REPO_NAME}"
echo ""
echo "📋 Sonraki adımlar:"
echo "1. Repository sayfanızı ziyaret edin"
echo "2. README.md'yi kontrol edin"
echo "3. Issues ve discussions'ı aktifleştirin"
echo "4. Topics ekleyin: machine-learning, pytorch, transformers, temporal-intelligence"