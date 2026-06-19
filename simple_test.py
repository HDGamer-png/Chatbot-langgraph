#!/usr/bin/env python3
"""Test multi-agent routing without external dependencies"""
import json
import time
import urllib.request
import urllib.error

time.sleep(1)

def test_chat(message, session_id):
    """Send chat request and return response"""
    url = "http://localhost:5000/api/chat"
    payload = json.dumps({
        "message": message,
        "session_id": session_id
    }).encode('utf-8')
    
    req = urllib.request.Request(
        url,
        data=payload,
        headers={'Content-Type': 'application/json'},
        method='POST'
    )
    
    try:
        with urllib.request.urlopen(req, timeout=30) as response:
            return json.loads(response.read().decode('utf-8'))
    except urllib.error.HTTPError as e:
        print(f"HTTP Error: {e.code} - {e.reason}")
        return None

print("=" * 60)
print("MULTI-AGENT ROUTING TEST")
print("=" * 60)

# Test 1: General query
print("\n[TEST 1] General Query")
print("-" * 60)
result = test_chat("What is artificial intelligence?", "test_general")
if result:
    print(f"Intent: {result.get('intent')}")
    print(f"Agents dispatched: {list(result.get('agent_timings', {}).keys())}")
    print(f"✓ PASS - Multi-agent routing works!" if result.get('intent') == 'general' and result.get('agent_timings') else "✗ FAIL")
else:
    print("✗ FAIL - No response")

# Test 2: Greeting query  
print("\n[TEST 2] Greeting Query")
print("-" * 60)
result = test_chat("Hello! How are you?", "test_greet")
if result:
    print(f"Intent: {result.get('intent')}")
    print(f"Agents dispatched: {list(result.get('agent_timings', {}).keys())}")
    print(f"✓ PASS - Multi-agent routing works!" if result.get('intent') in ('greet', 'general') and result.get('agent_timings') else "✗ FAIL")
else:
    print("✗ FAIL - No response")

print("\n" + "=" * 60)
print("TEST COMPLETED")
print("=" * 60)
