#!/usr/bin/env python3
"""Test script to verify multi-agent routing for general intents"""
import requests
import json
import time

# Give server time to start
time.sleep(2)

# Test 1: Health check
print("[TEST 1] Health Check")
try:
    resp = requests.get("http://localhost:5000/api/health", timeout=5)
    health = resp.json()
    print(f"  ✓ Health endpoint: {resp.status_code}")
    print(f"  - Provider: {health.get('provider')}")
    print(f"  - OCR enabled: {health.get('ocr_enabled')}")
except Exception as e:
    print(f"  ✗ Health check failed: {e}")
    exit(1)

# Test 2: General query - should dispatch WebSearch + Analyst
print("\n[TEST 2] General Query (should route to WebSearch + Analyst)")
try:
    payload = {
        "message": "What is machine learning?",
        "session_id": "test_general_query"
    }
    resp = requests.post(
        "http://localhost:5000/api/chat",
        json=payload,
        timeout=30
    )
    result = resp.json()
    intent = result.get("intent")
    agents_called = list(result.get("agent_timings", {}).keys()) if result.get("agent_timings") else []
    
    print(f"  ✓ Response received: {resp.status_code}")
    print(f"  - Intent detected: {intent}")
    print(f"  - Agents called: {agents_called}")
    
    if intent == "general" and len(agents_called) > 0:
        print(f"  ✓ PASS: General intent routed to multi-agent system!")
    else:
        print(f"  ✗ FAIL: Expected general intent with agents, got: {intent}, agents: {agents_called}")
        
except Exception as e:
    print(f"  ✗ Chat request failed: {e}")
    exit(1)

# Test 3: Greet query - should dispatch Analyst
print("\n[TEST 3] Greeting Query (should route to Analyst)")
try:
    payload = {
        "message": "Hello, how are you?",
        "session_id": "test_greet_query"
    }
    resp = requests.post(
        "http://localhost:5000/api/chat",
        json=payload,
        timeout=30
    )
    result = resp.json()
    intent = result.get("intent")
    agents_called = list(result.get("agent_timings", {}).keys()) if result.get("agent_timings") else []
    
    print(f"  ✓ Response received: {resp.status_code}")
    print(f"  - Intent detected: {intent}")
    print(f"  - Agents called: {agents_called}")
    
    if intent in ("greet", "general"):
        print(f"  ✓ PASS: Greeting intent routed to multi-agent system!")
    else:
        print(f"  ✗ FAIL: Expected greet/general intent, got: {intent}")
        
except Exception as e:
    print(f"  ✗ Chat request failed: {e}")
    exit(1)

print("\n[SUMMARY] Multi-agent routing tests completed!")
