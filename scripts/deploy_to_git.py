#!/usr/bin/env python3
"""
Automated Git → Replit Deploy Script
Tự động khởi tạo git, commit, push lên GitHub
"""

import os
import subprocess
import sys

def run_cmd(cmd, check=True):
    """Chạy lệnh shell và trả về output."""
    try:
        result = subprocess.run(cmd, shell=True, capture_output=True, text=True, check=check)
        return result.returncode, result.stdout.strip(), result.stderr.strip()
    except Exception as e:
        return -1, "", str(e)

def main():
    print("=" * 60)
    print("🚀 LangGraph Chatbot — Automated Deploy to Replit")
    print("=" * 60)
    
    # Bước 1: Kiểm tra git
    print("\n[1/4] Kiểm tra Git...")
    ret, out, err = run_cmd("git --version", check=False)
    if ret != 0:
        print("❌ Git chưa cài đặt. Vui lòng cài đặt Git từ https://git-scm.com")
        sys.exit(1)
    print("✓ Git đã cài đặt")
    
    # Bước 2: Khởi tạo git repo (nếu chưa có)
    print("\n[2/4] Khởi tạo Git repository...")
    ret, out, err = run_cmd("git status", check=False)
    if ret != 0:
        print("   → Khởi tạo git...")
        run_cmd("git init")
        run_cmd("git config user.email 'chatbot-deploy@example.com'")
        run_cmd("git config user.name 'Chatbot Deployer'")
        print("✓ Git repo khởi tạo")
    else:
        print("✓ Git repo đã tồn tại")
    
    # Bước 3: Add & commit
    print("\n[3/4] Staging files...")
    ret, out, err = run_cmd("git add .", check=False)
    if ret == 0:
        print("✓ Files đã stage")
    
    print("\n   → Đang commit...")
    ret, out, err = run_cmd('git commit -m "Deploy: Chatbot with dark theme & Replit config" --allow-empty', check=False)
    if ret == 0:
        print("✓ Commit thành công")
        print(f"   {out}")
    else:
        print("⚠ Commit không thành công (có thể không có thay đổi)")
    
    # Bước 4: Hỏi GitHub URL
    print("\n[4/4] Cấu hình GitHub...")
    print("\n📝 Hướng dẫn:")
    print("   1. Tạo repo GitHub mới tại: https://github.com/new")
    print("   2. Đặt tên: 'chatbot-langgraph' (hoặc tùy chọn)")
    print("   3. Copy HTTPS URL từ GitHub")
    
    github_url = input("\n🔗 Dán GitHub repo URL (HTTPS): ").strip()
    if not github_url:
        print("❌ Không nhập URL. Thoát.")
        sys.exit(1)
    
    print(f"\n   → Thêm remote: {github_url}")
    ret, out, err = run_cmd(f"git remote add origin {github_url}", check=False)
    if "already exists" in err:
        print("   (Remote đã tồn tại, bỏ qua)")
    
    print("   → Rename branch → main")
    run_cmd("git branch -M main", check=False)
    
    print("   → Push lên GitHub...")
    ret, out, err = run_cmd("git push -u origin main", check=False)
    if ret == 0:
        print("✓ Push thành công!")
    else:
        print(f"⚠ Push lỗi: {err}")
        print("   💡 Nếu cần, hãy khởi tạo git credential hoặc SSH key")
    
    # Kết thúc
    print("\n" + "=" * 60)
    print("✅ Xong! Tiếp theo trên Replit:")
    print("   1. Truy cập: https://replit.com/~")
    print("   2. Click '+ Create' → 'Import from GitHub'")
    print("   3. Dán URL repo")
    print("   4. Thêm Secrets (🔐): GROQ_API_KEY, TAVILY_API_KEY, etc.")
    print("   5. Click 'Run'")
    print("=" * 60)
    print("\n📖 Chi tiết xem: DEPLOY_REPLIT.md")

if __name__ == "__main__":
    main()
