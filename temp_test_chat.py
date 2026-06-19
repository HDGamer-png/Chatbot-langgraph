import json
import urllib.request

url = "http://127.0.0.1:5000/api/chat"
data = {"message": "Hello from test"}
req = urllib.request.Request(url, data=json.dumps(data).encode("utf-8"), headers={"Content-Type": "application/json"})
with urllib.request.urlopen(req, timeout=30) as resp:
    print(resp.read().decode("utf-8"))
