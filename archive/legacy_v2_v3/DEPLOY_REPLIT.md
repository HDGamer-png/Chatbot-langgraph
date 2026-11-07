# LangGraph Chatbot — Hướng dẫn Deploy Replit

## 📋 Điều kiện tiên quyết
- GitHub account (tạo miễn phí tại https://github.com/signup)
- Replit account (đăng ký tại https://replit.com)
- Git đã cài đặt trên máy tính

---

## 🚀 Quy trình Deploy (5 bước)

### **Bước 1: Khởi tạo Git Repository (trên máy local)**

Mở terminal/PowerShell trong thư mục project và chạy:

```bash
cd "d:\Python\Thực tập\Thuc_tap\Chatbot AI"
git init
git add .
git commit -m "Initial commit: LangGraph Chatbot with dark theme UI"
```

### **Bước 2: Tạo Repository trên GitHub**

1. Truy cập https://github.com/new
2. Đặt tên repo: `chatbot-langgraph` (hoặc tùy chọn)
3. Chọn **Public** hoặc **Private**
4. Bỏ chọn "Add a README.md" (vì đã có)
5. Click **Create repository**

### **Bước 3: Push code lên GitHub**

Sao chép lệnh từ GitHub (nó sẽ tương tự như dưới) và chạy:

```bash
git remote add origin https://github.com/YOUR-USERNAME/chatbot-langgraph.git
git branch -M main
git push -u origin main
```

⚠️ **Thay `YOUR-USERNAME` bằng username GitHub của bạn!**

### **Bước 4: Tạo Repl từ GitHub**

1. Truy cập https://replit.com/~
2. Click **+ Create** → chọn **Import from GitHub**
3. Dán URL repo: `https://github.com/YOUR-USERNAME/chatbot-langgraph`
4. Click **Import**
5. Đợi Replit clone code (~1-2 phút)

### **Bước 5: Cấu hình Secrets trên Replit**

**QUAN TRỌNG**: API keys KHÔNG được commit vào Git. Phải thêm vào Replit Secrets:

1. Trong Replit, click **🔐 Secrets** (bên trái, dưới Files)
2. Thêm các biến sau:
   - Key: `GROQ_API_KEY` → Value: `<GROQ_API_KEY>`
   - Key: `TAVILY_API_KEY` → Value: `<TAVILY_API_KEY>`
   - Key: `ANTHROPIC_API_KEY` (để trống nếu không có)

3. Click "Add secret" sau mỗi cái

### **Bước 6: Chạy ứng dụng**

Trên Replit:
1. Click nút **Run** (xanh, phía trên)
2. Chờ dependencies cài đặt (~2-3 phút lần đầu)
3. Khi xong, sẽ thấy URL: `https://chatbot-langgraph.replit.dev`
4. Copy & chia sẻ link này với ai cũng được! 🎉

---

## 📝 Lệnh tự động (cho những lần cập nhật sau)

Khi bạn muốn push bản cập nhật:

```bash
git add .
git commit -m "Update: [mô tả thay đổi]"
git push origin main
```

Sau đó Replit sẽ **tự động pull & deploy** (nếu bạn bật auto-deploy).

---

## 🆘 Troubleshooting

| Lỗi | Nguyên nhân | Giải pháp |
|-----|-----------|----------|
| "Not a git repo" | Chưa init git | Chạy `git init` |
| "Permission denied" | GitHub auth fail | Kiểm tra SSH key hoặc dùng HTTPS + token |
| "API Key not found" | Secrets chưa add | Kiểm tra Replit → Secrets |
| "Port already in use" | Server đã chạy | Restart Replit (Ctrl+C, click Run lại) |
| "Rate limit exceeded" | Groq quota | Chờ 24h hoặc nâng cấp gói |

---

## 💡 Tips

- **Dev mode**: Thêm `FLASK_DEBUG=true` vào Secrets để bật debug mode
- **Backup**: Commit thường xuyên trên GitHub
- **Update deps**: Khi thêm package mới, chạy `pip freeze > requirements.txt` & push
- **Logs**: Xem logs real-time trên Replit → Console

---

**Sẵn sàng chưa? Bắt đầu từ Bước 1 nhé! 🚀**
