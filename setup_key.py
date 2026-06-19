"""
╔══════════════════════════════════════════════════════════╗
║   SETUP API KEYS — HUONG DAN LAY KEY & LUU AN TOAN     ║
║   Chay: python setup_key.py                             ║
╚══════════════════════════════════════════════════════════╝
"""
import os, sys, webbrowser
from pathlib import Path

ENV_FILE = Path(__file__).parent / ".env"

def sep(char="─", n=54): print(char * n)

def banner(title: str):
    sep("═")
    print(f"  {title}")
    sep("═")

def step(n: int, title: str):
    print(f"\n[Buoc {n}] {title}")
    sep()

def save_env(key: str, val: str):
    lines = ENV_FILE.read_text().splitlines() if ENV_FILE.exists() else []
    found = False
    for i, l in enumerate(lines):
        if l.startswith(f"{key}="):
            lines[i] = f"{key}={val}"; found = True; break
    if not found:
        lines.append(f"{key}={val}")
    ENV_FILE.write_text("\n".join(lines) + "\n")

def read_env(key: str) -> str:
    if not ENV_FILE.exists(): return ""
    for l in ENV_FILE.read_text().splitlines():
        if l.startswith(f"{key}="):
            return l.split("=", 1)[1].strip()
    return ""

# ──────────────────────────────────────────────────────────────────
banner("SETUP API KEYS CHO LANGGRAPH CHATBOT")

print("""
Script nay se huong dan ban:
  1. Lay Anthropic API Key (bat buoc) — de chay LLM Claude
  2. Lay Tavily Search Key (tuy chon) — de tim kiem web that
  3. Luu an toan vao file .env (KHONG nhung vao code)
""")

# ══════════════════════════════════════════════════════════════════
#  ANTHROPIC API KEY
# ══════════════════════════════════════════════════════════════════
banner("PHAN 1: ANTHROPIC API KEY (BAT BUOC)")

existing_ant = read_env("ANTHROPIC_API_KEY")
if existing_ant and not existing_ant.startswith("your-"):
    print(f"\n  Da tim thay key hien tai: ...{existing_ant[-8:]}")
    keep = input("  Giu nguyen? (Enter=co / n=doi moi): ").strip().lower()
    if keep != "n":
        print("  [OK] Giu nguyen Anthropic Key.")
        ant_key = existing_ant
    else:
        ant_key = ""
else:
    ant_key = ""

if not ant_key:
    step(1, "Dang ky tai khoan Anthropic")
    print("""
  Neu chua co tai khoan Anthropic:
  a) Truy cap: https://console.anthropic.com
  b) Chon "Sign Up" -> Dang ky bang Google hoac email
  c) Xac nhan email
  d) Dang nhap vao Console
""")
    mo = input("  Mo trinh duyet den console.anthropic.com? (Enter=co / n=bo qua): ").strip().lower()
    if mo != "n":
        try:
            webbrowser.open("https://console.anthropic.com")
            print("  [OK] Da mo trinh duyet.")
        except:
            print("  [!] Khong mo duoc. Vui long tu truy cap: https://console.anthropic.com")

    step(2, "Tao API Key")
    print("""
  Trong Console Anthropic:
  1. Menu ben trai -> chon "API Keys"  (hoac /settings/keys)
  2. Nhan nut "Create Key"
  3. Dat ten key: vi du "langgraph-chatbot"
  4. Copy toan bo key (bat dau bang "sk-ant-...")
  5. LUU Y: Key chi hien 1 lan, sau do khong xem lai duoc!
""")
    input("  Nhan Enter sau khi da copy key... ")

    step(3, "Nhap key vao day")
    while True:
        ant_key = input("  Nhap Anthropic API Key (sk-ant-...): ").strip()
        if not ant_key:
            print("  [!] Ban bo trong. Thu lai.")
            continue
        if not ant_key.startswith("sk-ant-"):
            yn = input(f"  Key khong bat dau bang 'sk-ant-'. Tiep tuc? (y/n): ").strip().lower()
            if yn != "y": continue
        break

    save_env("ANTHROPIC_API_KEY", ant_key)
    os.environ["ANTHROPIC_API_KEY"] = ant_key
    print(f"  [OK] Da luu Anthropic Key: ...{ant_key[-8:]}")

    step(4, "Kiem tra hoa don & Free tier")
    print("""
  - Anthropic KHONG co free tier vinh vien.
  - Khi dang ky, ban nhan $5 credit dung thu (du de chay ~1000 luot chat).
  - De xem so du: Console -> "Billing" -> "Usage"
  - Neu het credit: "Billing" -> "Add Credits" (toi thieu $5)
  - Gia: claude-sonnet-4-20250514 ~ $0.003 / 1000 tokens input
           (moi cau hoi trung binh tieu khoang $0.001-0.003)
""")

# ══════════════════════════════════════════════════════════════════
#  TAVILY SEARCH KEY (TUY CHON)
# ══════════════════════════════════════════════════════════════════
banner("PHAN 2: TAVILY SEARCH KEY (TUY CHON)")
print("""
  Tavily cho phep Chatbot tim kiem web that.
  Neu bo qua, Chatbot van chay duoc voi Knowledge Base noi bo.
  Free tier: 1,000 requests/thang (du de demo).
""")

existing_tav = read_env("TAVILY_API_KEY")
if existing_tav and not existing_tav.startswith("your-"):
    print(f"  Da co Tavily Key: ...{existing_tav[-6:]}")
    tav_key = existing_tav
else:
    want = input("  Ban muon cai Tavily de tim kiem web that? (y/Enter=co / n=bo qua): ").strip().lower()
    tav_key = ""
    if want != "n":
        print("""
  Cach lay Tavily API Key (mien phi):
  1. Truy cap: https://app.tavily.com
  2. "Sign Up" bang Google hoac email
  3. Sau khi dang nhap -> chon "API" tren menu
  4. Copy key (bat dau bang "tvly-...")
""")
        mo2 = input("  Mo trinh duyet den app.tavily.com? (Enter=co / n=bo qua): ").strip().lower()
        if mo2 != "n":
            try:
                webbrowser.open("https://app.tavily.com")
                print("  [OK] Da mo trinh duyet.")
            except:
                print("  [!] Tu truy cap: https://app.tavily.com")
        input("\n  Nhan Enter sau khi da copy Tavily key... ")
        tav_key = input("  Nhap Tavily Key (tvly-...) hoac Enter de bo qua: ").strip()
        if tav_key:
            save_env("TAVILY_API_KEY", tav_key)
            print(f"  [OK] Da luu Tavily Key: ...{tav_key[-6:]}")
        else:
            print("  [--] Bo qua Tavily. Dung Knowledge Base noi bo.")

# ══════════════════════════════════════════════════════════════════
#  KIEM TRA KET NOI
# ══════════════════════════════════════════════════════════════════
banner("PHAN 3: KIEM TRA KET NOI")

print("\n  Dang kiem tra Anthropic API Key...", end="", flush=True)
try:
    from langchain_anthropic import ChatAnthropic
    from langchain_core.messages import HumanMessage
    llm  = ChatAnthropic(model="claude-haiku-20240307",
                         anthropic_api_key=ant_key, max_tokens=20)
    resp = llm.invoke([HumanMessage(content="Hi")])
    print(f" OK! (phan hoi: '{resp.content[:30]}')")
    ant_ok = True
except Exception as ex:
    print(f"\n  [LOI] {ex}")
    print("  -> Key co the sai hoac chua co credit. Kiem tra lai tren console.anthropic.com")
    ant_ok = False

if tav_key:
    print("  Dang kiem tra Tavily Key...", end="", flush=True)
    try:
        from tavily import TavilyClient
        r = TavilyClient(api_key=tav_key).search("test", max_results=1)
        print(f" OK! ({len(r.get('results',[]))} ket qua)")
    except Exception as ex:
        print(f"\n  [LOI] {ex}")

# ══════════════════════════════════════════════════════════════════
#  TOM TAT
# ══════════════════════════════════════════════════════════════════
banner("KET QUA SETUP")
print(f"""
  File .env  : {ENV_FILE}
  Anthropic  : {'OK - ...'+ant_key[-8:] if ant_ok else 'LOI - Kiem tra lai'}
  Tavily     : {'OK - ...'+tav_key[-6:] if tav_key else 'Bo qua (dung KB noi bo)'}

  BAO MAT:
  - File .env chua key - KHONG commit len GitHub/GitLab
  - Them ".env" vao .gitignore:
      echo ".env" >> .gitignore
  - KHONG nhung key truc tiep vao code Python

  CHAY CHATBOT:
      python multi_agent_chatbot.py
""")

if ant_ok:
    run = input("  Chay chatbot ngay bay gio? (Enter=co / n=thoat): ").strip().lower()
    if run != "n":
        print()
        import subprocess, sys
        subprocess.run([sys.executable, Path(__file__).parent / "multi_agent_chatbot.py"])
