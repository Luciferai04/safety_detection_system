#!/usr/bin/env python3
"""
Test API health by starting server and making test requests
"""

import requests
import time
import subprocess
import threading
import signal
import sys
from pathlib import Path

def test_api_endpoints():
 """Test API endpoints with health checks"""

 base_url = "http://localhost:5001" # Use port 5001 to avoid conflicts

 print(" Testing API Endpoints")
 print("=" * 40)

 try:
 # Test health check
 print("1. Testing health check endpoint...")
 response = requests.get(f"{base_url}/api/health", timeout=5)
 if response.status_code == 200:
 health_data = response.json()
 print(f" Health check successful")
 print(f" Status: {health_data.get('status')}")
 print(f" Detector ready: {health_data.get('detector_ready')}")
 print(f" Device: {health_data.get('device')}")
 else:
 print(f" Health check failed: {response.status_code}")
 return False

 # Test API documentation
 print("\n2. Testing API documentation...")
 response = requests.get(f"{base_url}/api/docs", timeout=5)
 if response.status_code == 200:
 print(f" API documentation accessible")
 else:
 print(f" API documentation failed: {response.status_code}")

 # Test configuration endpoint
 print("\n3. Testing configuration endpoint...")
 response = requests.get(f"{base_url}/api/config", timeout=5)
 if response.status_code == 200:
 config_data = response.json()
 print(f" Configuration endpoint working")
 print(f" Confidence threshold: {config_data.get('confidence_threshold')}")
 else:
 print(f" Configuration endpoint failed: {response.status_code}")

 print(f"\n API endpoints are working correctly!")
 return True

 except requests.exceptions.ConnectionError:
 print(f" Could not connect to API server at {base_url}")
 return False
 except Exception as e:
 print(f" Error testing API: {e}")
 return False

def start_api_server():
 """Start API server on port 5001"""

 print(" Starting API server on port 5001...")

 # Create a simple API startup script
 api_script = """
import sys
sys.path.append('src')
from api import app

if __name__ == '__main__':
 app.run(debug=False, host='0.0.0.0', port=5001, use_reloader=False)
"""

 # Write temporary script
 with open("temp_api_test.py", "w") as f:
 f.write(api_script)

 try:
 # Start server in subprocess
 env = {"PYTHONPATH": ".", "PATH": "/Users/soumyajitghosh/safety_detection_system/venv/bin:" +
 subprocess.os.environ.get("PATH", "")}

 process = subprocess.Popen([
 "/Users/soumyajitghosh/safety_detection_system/venv/bin/python3",
 "temp_api_test.py"
 ],
 env=env,
 stdout=subprocess.PIPE,
 stderr=subprocess.PIPE)

 # Wait for server to start
 time.sleep(3)

 # Test if server is running
 if process.poll() is None: # Process is still running
 print(" API server started successfully")
 return process
 else:
 print(" API server failed to start")
 stdout, stderr = process.communicate()
 print(f" Error: {stderr.decode()}")
 return None

 except Exception as e:
 print(f" Error starting API server: {e}")
 return None

def main():
 """Main test function"""

 print(" Safety Detection System - API Testing")
 print("=" * 50)

 # Start API server
 server_process = start_api_server()

 if server_process:
 try:
 # Test API endpoints
 api_test_passed = test_api_endpoints()

 print(f"\n API TEST SUMMARY")
 print(f"=" * 50)
 print(f" API Server: {'STARTED' if server_process.poll() is None else 'FAILED'}")
 print(f" API Endpoints: {'PASSED' if api_test_passed else 'FAILED'}")

 if api_test_passed:
 print(f"\n API TESTS PASSED! The API server is working correctly.")
 else:
 print(f"\n Some API tests failed.")

 finally:
 # Clean up
 print(f"\n Cleaning up...")
 server_process.terminate()
 server_process.wait()

 # Remove temporary file
 if Path("temp_api_test.py").exists():
 Path("temp_api_test.py").unlink()

 print(f" Server stopped and cleanup complete")
 else:
 print(f"\n Could not start API server for testing")
 return False

 return api_test_passed if 'api_test_passed' in locals() else False

if __name__ == "__main__":
 success = main()
 sys.exit(0 if success else 1)
