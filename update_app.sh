#!/bin/bash
set -e

# طباعة التاريخ الحالي
date
echo "Updating Python application on VM..."

# تعريف المجلد ومسار الـ Repo
APP_DIR="/home/azureuser"  # بما أن الملفات موجودة في الجذر
GIT_REPO="git@github.com:Sarah1234n/chatbot.git"
BRANCH="master"  # تأكد من تغيير هذا إذا كان الفرع مختلف

# تحديث الشيفرة
if [ -d "$APP_DIR/.git" ]; then
    echo "Pulling latest code..."
    sudo -u azureuser git -C "$APP_DIR" pull origin "$BRANCH"
else
    echo "Cloning repo..."
    sudo -u azureuser git clone -b "$BRANCH" "git@github.com:Sarah1234n/chatbot.git" "$APP_DIR"
fi

# تنشيط البيئة الافتراضية وتثبيت المتطلبات
VENV_PATH="$APP_DIR/venv/bin/activate"
REQ_FILE="$APP_DIR/requirements.txt"

# التأكد من وجود البيئة الافتراضية
if [ -f "$VENV_PATH" ]; then
    echo "Activating virtual environment and installing requirements..."
    
    # تفعيل البيئة الافتراضية
    source "$VENV_PATH"
    
    # التأكد من أن pip محدث
    "$APP_DIR/venv/bin/pip" install --upgrade pip

    # تثبيت المتطلبات
    if [ -f "$REQ_FILE" ]; then
        "$APP_DIR/venv/bin/pip" install -r "$REQ_FILE"
    else
        echo "requirements.txt not found at $REQ_FILE"
        exit 1
    fi
else
    echo "Virtual environment not found at $VENV_PATH"
    exit 1
fi

# إعادة تشغيل الخدمات
sudo systemctl restart chroma.service
sudo systemctl restart backend.service
sudo systemctl restart chatbot.service

echo "✅ Python application update completed!"

