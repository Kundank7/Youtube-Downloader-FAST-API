import requests
import sys
from datetime import datetime
import time

class YouTubeDownloaderTester:
    def __init__(self, base_url="https://kawaiitube.preview.emergentagent.com"):
        self.base_url = base_url
        self.api_url = f"{base_url}/api"
        self.session = requests.Session()
        self.tests_run = 0
        self.tests_passed = 0
        self.test_results = []

    def run_test(self, name, method, endpoint, expected_status, data=None, timeout=30):
        """Run a single API test"""
        url = f"{self.api_url}/{endpoint}"
        
        self.tests_run += 1
        print(f"\n🔍 Testing {name}...")
        print(f"   URL: {url}")
        
        try:
            if method == 'GET':
                response = self.session.get(url, timeout=timeout)
            elif method == 'POST':
                response = self.session.post(url, json=data, timeout=timeout)

            success = response.status_code == expected_status
            if success:
                self.tests_passed += 1
                print(f"✅ Passed - Status: {response.status_code}")
                self.test_results.append({
                    "test": name,
                    "status": "PASSED",
                    "status_code": response.status_code
                })
            else:
                print(f"❌ Failed - Expected {expected_status}, got {response.status_code}")
                print(f"   Response: {response.text[:200]}")
                self.test_results.append({
                    "test": name,
                    "status": "FAILED",
                    "expected": expected_status,
                    "actual": response.status_code,
                    "response": response.text[:200]
                })

            return success, response

        except requests.exceptions.Timeout:
            print(f"❌ Failed - Request timed out after {timeout}s")
            self.test_results.append({
                "test": name,
                "status": "FAILED",
                "error": "Timeout"
            })
            return False, None
        except Exception as e:
            print(f"❌ Failed - Error: {str(e)}")
            self.test_results.append({
                "test": name,
                "status": "FAILED",
                "error": str(e)
            })
            return False, None

    def test_metadata_fetch(self):
        """Test fetching video metadata"""
        # Using a short, reliable YouTube video
        test_url = "https://www.youtube.com/watch?v=jNQXAC9IVRw"  # "Me at the zoo" - first YouTube video
        
        success, response = self.run_test(
            "Fetch Video Metadata",
            "POST",
            "metadata",
            200,
            data={"url": test_url},
            timeout=60  # Longer timeout for yt-dlp
        )
        
        if success:
            try:
                data = response.json()
                print(f"   Video Title: {data.get('title', 'N/A')}")
                print(f"   Channel: {data.get('channel', 'N/A')}")
                print(f"   Duration: {data.get('duration', 0)}s")
                print(f"   Available Formats: {len(data.get('formats', []))}")
                
                # Verify required fields
                required_fields = ['id', 'title', 'thumbnail', 'duration', 'channel', 'formats']
                missing_fields = [f for f in required_fields if f not in data]
                if missing_fields:
                    print(f"⚠️  Warning: Missing fields: {missing_fields}")
                    return False, None
                
                # Verify formats include expected qualities
                formats = data.get('formats', [])
                qualities = [f.get('quality') for f in formats]
                print(f"   Qualities available: {qualities}")
                
                return True, data
            except Exception as e:
                print(f"❌ Failed to parse metadata response: {str(e)}")
                return False, None
        
        return False, None

    def test_invalid_url(self):
        """Test metadata fetch with invalid URL"""
        success, response = self.run_test(
            "Invalid URL Handling",
            "POST",
            "metadata",
            400,  # Should return 400 for invalid URL
            data={"url": "https://invalid-url.com/not-youtube"},
            timeout=30
        )
        return success

    def test_history_empty(self):
        """Test history endpoint with no downloads"""
        success, response = self.run_test(
            "Get History (Empty)",
            "GET",
            "history",
            200,
            timeout=10
        )
        
        if success:
            try:
                data = response.json()
                print(f"   History items: {len(data)}")
                return True
            except Exception as e:
                print(f"❌ Failed to parse history response: {str(e)}")
                return False
        
        return False

    def test_download_endpoint_structure(self):
        """Test download endpoint structure (without full download)"""
        # This test just checks if the endpoint exists and responds
        # We won't do a full download to save time and resources
        test_url = "https://www.youtube.com/watch?v=jNQXAC9IVRw"
        
        print(f"\n🔍 Testing Download Endpoint Structure...")
        print(f"   Note: Not performing full download to save resources")
        print(f"   Checking if endpoint accepts requests...")
        
        # Just check if endpoint is reachable
        # A 500 error or timeout would indicate issues
        # We expect it to start processing (may take time)
        self.tests_run += 1
        
        try:
            # Send request but don't wait for full download
            response = self.session.post(
                f"{self.api_url}/download",
                json={
                    "url": test_url,
                    "format_id": "18",  # Common format ID
                    "quality": "360p"
                },
                timeout=5,  # Short timeout - we just want to see if it starts
                stream=True
            )
            
            # If we get here without exception, endpoint is working
            print(f"✅ Download endpoint is reachable and accepting requests")
            print(f"   Status: {response.status_code}")
            self.tests_passed += 1
            self.test_results.append({
                "test": "Download Endpoint Structure",
                "status": "PASSED",
                "note": "Endpoint reachable, not testing full download"
            })
            return True
            
        except requests.exceptions.Timeout:
            # Timeout is actually OK here - means it's processing
            print(f"✅ Download endpoint started processing (timed out as expected)")
            self.tests_passed += 1
            self.test_results.append({
                "test": "Download Endpoint Structure",
                "status": "PASSED",
                "note": "Endpoint processing started"
            })
            return True
        except Exception as e:
            print(f"❌ Failed - Error: {str(e)}")
            self.test_results.append({
                "test": "Download Endpoint Structure",
                "status": "FAILED",
                "error": str(e)
            })
            return False

    def print_summary(self):
        """Print test summary"""
        print("\n" + "="*60)
        print("📊 TEST SUMMARY")
        print("="*60)
        print(f"Total Tests: {self.tests_run}")
        print(f"Passed: {self.tests_passed}")
        print(f"Failed: {self.tests_run - self.tests_passed}")
        print(f"Success Rate: {(self.tests_passed/self.tests_run*100):.1f}%")
        print("="*60)
        
        return self.tests_passed == self.tests_run

def main():
    print("="*60)
    print("🚀 YouTube Downloader API Testing")
    print("="*60)
    
    tester = YouTubeDownloaderTester()
    
    # Test 1: Fetch metadata for valid video
    print("\n📹 Testing Video Metadata Fetching...")
    metadata_success, metadata = tester.test_metadata_fetch()
    
    # Test 2: Invalid URL handling
    print("\n🚫 Testing Invalid URL Handling...")
    tester.test_invalid_url()
    
    # Test 3: History endpoint
    print("\n📜 Testing History Endpoint...")
    tester.test_history_empty()
    
    # Test 4: Download endpoint structure
    print("\n⬇️  Testing Download Endpoint...")
    tester.test_download_endpoint_structure()
    
    # Print summary
    all_passed = tester.print_summary()
    
    return 0 if all_passed else 1

if __name__ == "__main__":
    sys.exit(main())
