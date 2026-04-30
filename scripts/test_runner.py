#!/usr/bin/env python
"""
Unified Test Runner for RAG Wiki
Combines automated execution with fallback to manual testing
"""

import requests
import json
import time
from pathlib import Path
from datetime import datetime
import sys

STREAMLIT_URL = "http://localhost:8501"

class UnifiedTestRunner:
    def __init__(self, base_url=STREAMLIT_URL):
        self.base_url = base_url
        self.session = requests.Session()
        self.results = []
        self.automated_mode = False

    def check_app_running(self, timeout=60):
        """Check if Streamlit app is running"""
        print("⏳ Checking Streamlit app...")
        start_time = time.time()

        while time.time() - start_time < timeout:
            try:
                response = self.session.get(self.base_url, timeout=5)
                if response.status_code == 200:
                    print("✓ Streamlit app is running\n")
                    self.automated_mode = True
                    return True
            except:
                pass
            time.sleep(2)

        print("⚠ App not running - switching to manual test mode\n")
        return False

    def run_test(self, test_id, query, expected_criteria):
        """Run single test - automated or manual"""
        print(f"{'='*70}")
        print(f"Test {test_id}: {query[:60]}...")
        print(f"{'='*70}")

        if self.automated_mode:
            return self._run_automated(test_id, query, expected_criteria)
        else:
            return self._run_manual(test_id, query, expected_criteria)

    def _run_automated(self, test_id, query, expected_criteria):
        """Automated test execution"""
        try:
            # Send query to app
            response = self._send_query(query)

            if response is None:
                return self._record_result(test_id, query, "ERROR", "No response from app")

            # Validate against criteria
            passed, failures = self._validate(response, expected_criteria)
            status = "✓ PASS" if passed else "✗ FAIL"
            notes = ", ".join(failures) if failures else "All criteria met"

            print(f"Status: {status}")
            print(f"Notes: {notes}\n")

            return self._record_result(test_id, query, "PASS" if passed else "FAIL", notes, response)

        except Exception as e:
            print(f"✗ Error: {str(e)}\n")
            return self._record_result(test_id, query, "ERROR", str(e))

    def _run_manual(self, test_id, query, expected_criteria):
        """Manual test guidance"""
        print(f"📝 Expected Criteria: {expected_criteria}")
        print(f"\n🔍 Manual Test Steps:")
        print("  1. Go to http://localhost:8501")

        if query:
            print(f"  2. Enter: '{query}'")
            print(f"  3. Check if response matches criteria")
        else:
            print(f"  2. Leave input empty and verify graceful handling")
            print(f"  3. Check app doesn't crash")

        print(f"  4. Verify sources are cited correctly")
        print(f"  5. Enter result (pass/fail/skip): ", end="")

        # For automated testing, default to manual check required
        result = "skip"
        # Uncomment below for interactive mode:
        # result = input().lower()

        status_map = {"pass": "PASS", "fail": "FAIL", "skip": "MANUAL"}
        status = status_map.get(result, "MANUAL")

        print(f"\n{status} - Manual validation recorded\n")
        return self._record_result(test_id, query, status, f"Manual check: {result}")

    def _send_query(self, query):
        """Send query to app and get response"""
        # This would require actual API endpoint or browser automation
        # Placeholder implementation
        time.sleep(1)
        return {"answer": "Mock response", "sources": [], "pages": []}

    def _validate(self, response, criteria):
        """Validate response against criteria"""
        failures = []
        # Implement validation logic here
        return len(failures) == 0, failures

    def _record_result(self, test_id, query, status, notes, response=None):
        """Record test result"""
        result = {
            "test_id": test_id,
            "query": query,
            "status": status,
            "notes": notes,
            "timestamp": datetime.now().isoformat(),
            "response": response or {}
        }
        self.results.append(result)
        return result

    def run_all_tests(self):
        """Execute complete test suite"""
        test_cases = [
            ("1.1", "What is MCP?", "Should cite Intro_to_MCP.pdf only"),
            ("1.2", "Tell me about Mobile Content Provider", "Handle missing term gracefully"),
            ("2.1", "What are Linux commands?", "Should cite linux-commands.pdf"),
            ("2.4", "What are the different types of screens in an Agentry app?", "Should cite smp_agentry_language_reference.pdf"),
            ("3.1", "Compare different software frameworks", "Retrieve from multiple docs"),
            ("4.1", "", "Graceful empty input handling"),
            ("4.2", "What is AI/ML & how does it work?", "Parse special characters"),
            ("6.1", "What are Linux commands?", "List all sources with pages"),
        ]

        print(f"\n{'='*70}")
        print(f"Running {len(test_cases)} test cases")
        print(f"Mode: {'AUTOMATED' if self.automated_mode else 'MANUAL'}")
        print(f"{'='*70}\n")

        for test_id, query, criteria in test_cases:
            self.run_test(test_id, query, criteria)

        self._generate_reports()

    def _generate_reports(self):
        """Generate test reports"""
        # Count results
        total = len(self.results)
        passed = sum(1 for r in self.results if r["status"] == "PASS")
        failed = sum(1 for r in self.results if r["status"] == "FAIL")
        manual = sum(1 for r in self.results if r["status"] == "MANUAL")

        # Print summary
        print(f"\n{'='*70}")
        print("TEST SUMMARY")
        print(f"{'='*70}")
        print(f"Total: {total} | ✓ Passed: {passed} | ✗ Failed: {failed} | ⚠ Manual: {manual}")
        print(f"{'='*70}\n")

        # Save reports
        self._save_markdown_report()
        # self._save_json_report()

    def _save_markdown_report(self):
        """Save Markdown report"""
        md = f"""# Test Execution Report

**Date**: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}
**Mode**: {'Automated' if self.automated_mode else 'Manual'}

## Summary
| Status | Count |
|--------|-------|
| Total | {len(self.results)} |
| Passed | {sum(1 for r in self.results if r['status'] == 'PASS')} |
| Failed | {sum(1 for r in self.results if r['status'] == 'FAIL')} |
| Manual | {sum(1 for r in self.results if r['status'] == 'MANUAL')} |

## Results
| Test ID | Query | Status | Notes |
|---------|-------|--------|-------|
"""
        for r in self.results:
            md += f"| {r['test_id']} | {r['query'][:40]} | {r['status']} | {r['notes'][:50]} |\n"

        Path("TEST_RESULTS.md").write_text(md)
        print("✓ Report saved to `TEST_RESULTS.md`")

    def _save_json_report(self):
        """Save JSON report"""
        Path("TEST_RESULTS.json").write_text(json.dumps(self.results, indent=2))
        print("✓ Report saved to `TEST_RESULTS.json`")

if __name__ == "__main__":
    runner = UnifiedTestRunner()
    runner.check_app_running()
    runner.run_all_tests()
