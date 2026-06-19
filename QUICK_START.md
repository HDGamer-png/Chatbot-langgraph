# 🚀 Quick Start — Deploy lên Replit (2 phút)

## 📋 Bạn đã có

✅ Git repo đã khởi tạo tại: `D:\Python\Thực tập\Thuc_tap\Chatbot AI\.git`
✅ Tất cả files đã được commit (hash: xem `git log`)
✅ `.gitignore` đã setup (`.env`, `__pycache__`, `chat_history/` được ignored)

---

## 🔧 Tiếp theo (3 bước)

### **Bước 1: Tạo GitHub Repository**

1. Truy cập: https://github.com/new
2. Repository name: `chatbot-langgraph`
3. Public (để dễ share)
4. **Bỏ chọn** "Add a README.md"
5. Click **"Create repository"**

GitHub sẽ cho bạn lệnh setup — copy & chạy.

---

### **Bước 2: Push lên GitHub**

Chạy lệnh này trong PowerShell (tại thư mục project):

```bash
git remote add origin https://github.com/YOUR-USERNAME/chatbot-langgraph.git
git branch -M main
git push -u origin main
```

⚠️ **Thay `YOUR-USERNAME` bằng username GitHub của bạn**

Nếu bị hỏi password → dùng **Personal Access Token** (không phải password):
- Tạo tại: https://github.com/settings/tokens
- Chọn `repo` scope
- Copy token vào password field

---

### **Bước 3: Import vào Replit**

1. Truy cập: https://replit.com
2. Đăng nhập / Tạo tài khoản
3. Click **"+ Create"** → **"Import from GitHub"**
4. Paste GitHub URL: `https://github.com/YOUR-USERNAME/chatbot-langgraph`
5. Click **"Import"** — chờ 1-2 phút

---

## 🔐 Cấu hình Secrets (QUAN TRỌNG!)

Trên Replit, **ĐỌC KỸ**:

1. Click **🔐 Secrets** (bên trái Files)
2. Thêm các key-value:

| Key | Value |
|-----|-------|
| `GROQ_API_KEY` | `<GROQ_API_KEY>` |
| `TAVILY_API_KEY` | `<TAVILY_API_KEY>` |

3. Click **"Add secret"** sau mỗi cái

---

## ▶️ Chạy trên Replit

1. Click nút xanh **"Run"** phía trên
2. Chờ dependencies cài đặt (~3 phút lần đầu)
3. Khi xong → sẽ thấy URL: `https://chatbot-langgraph.replit.dev`
4. **Chia sẻ link này cho bất kỳ ai!** 🎉

---

## 📝 Cập nhật sau này

Khi có thay đổi mới:

```bash
git add .
git commit -m "fix: [mô tả]"
git push origin main
```

Replit sẽ **tự động pull & restart** (nếu bạn bật auto-deploy).

---

## 🆘 Lỗi thường gặp

| Lỗi | Cách sửa |
|-----|---------|
| `git: command not found` | Cài Git: https://git-scm.com |
| `Authentication failed` | Dùng GitHub Personal Access Token (không phải password) |
| `API Key not found` | Kiểm tra Secrets trên Replit đã add chưa |
| `Port already in use` | Restart Replit (Ctrl+C, click Run lại) |

---

**Hoàn tất? Bắt đầu từ Bước 1 ngay! 🚀**
