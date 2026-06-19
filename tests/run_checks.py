import os
import time
import requests
import sys

BASE = os.getenv("TEST_BASE_URL", "http://127.0.0.1:5000")
HEADERS = {"Content-Type": "application/json"}

def poll_health(timeout=60):
    start = time.time()
    while time.time() - start < timeout:
        try:
            r = requests.get(f"{BASE}/api/health", timeout=5)
            if r.status_code == 200:
                j = r.json()
                if j.get("status") == "ok":
                    print("HEALTH_OK", j)
                    return True
                else:
                    print("HEALTH_RESP", j)
            else:
                print("HEALTH_STATUS", r.status_code)
        except Exception as e:
            print("HEALTH_ERR", e)
        time.sleep(1)
    return False


def create_session():
    r = requests.post(f"{BASE}/api/session/new", headers=HEADERS, json={})
    r.raise_for_status()
    j = r.json()
    print("NEW_SESSION", j)
    return j.get("session_id")


def send_chat(session_id, message):
    payload = {"message": message, "session_id": session_id}
    r = requests.post(f"{BASE}/api/chat", headers=HEADERS, json=payload, timeout=60)
    try:
        j = r.json()
    except Exception:
        print("NON_JSON_RESPONSE", r.text)
        r.raise_for_status()
    print("CHAT_RESPONSE", j)
    return j


if __name__ == '__main__':
    ok = poll_health(timeout=60)
    if not ok:
        print("ERROR: health endpoint not ready within timeout")
        sys.exit(2)

    try:
        session_id = create_session()
    except Exception as e:
        print("ERROR creating session:", e)
        sys.exit(3)

    tests = [
        "Viết 1 đoạn mô tả ngắn về Python",
        "Giải phương trình 2x + 3 = 11",
        "Viết hàm Python để tính giai thừa của một số",
    ]

    failures = 0
    for t in tests:
        try:
            resp = send_chat(session_id, t)
            if resp.get("error"):
                print("FAIL: error returned:", resp.get("error"))
                failures += 1
        except Exception as e:
            print("EXCEPTION during chat:", e)
            failures += 1

    if failures:
        print(f"TESTS finished with {failures} failures")
        sys.exit(4)
    else:
        print("ALL TESTS PASSED")
        sys.exit(0)
