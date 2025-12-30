#!/usr/bin/env python3
"""
Comprehensive test script for all available NeuronMCP PostgreSQL tools
Tests all 8 PostgreSQL administration and monitoring tools
"""

import subprocess
import json
import sys
import time
from datetime import datetime

def send_mcp_request(neuronmcp_path, request):
    """Send an MCP request and get response"""
    try:
        request_json = json.dumps(request)
        
        process = subprocess.Popen(
            [neuronmcp_path],
            stdin=subprocess.PIPE,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
            cwd='/home/pge/pge/neurondb/bin/neuronmcp'
        )
        
        stdout, stderr = process.communicate(input=request_json + '\n', timeout=10)
        
        if stdout.strip():
            lines = stdout.strip().split('\n')
            for line in lines:
                try:
                    response = json.loads(line)
                    return response
                except json.JSONDecodeError:
                    continue
        
        return None
        
    except subprocess.TimeoutExpired:
        process.kill()
        return None
    except Exception as e:
        return None

def test_postgresql_tool(neuronmcp_path, tool_name, arguments=None, description=""):
    """Test a PostgreSQL tool"""
    if arguments is None:
        arguments = {}
    
    request = {
        "jsonrpc": "2.0",
        "id": 2,
        "method": "tools/call",
        "params": {
            "name": tool_name,
            "arguments": arguments
        }
    }
    
    print(f"\n{'='*70}")
    print(f"📊 Tool: {tool_name}")
    if description:
        print(f"   {description}")
    print(f"   Arguments: {json.dumps(arguments, indent=2) if arguments else 'None'}")
    print(f"{'='*70}")
    
    start_time = time.time()
    response = send_mcp_request(neuronmcp_path, request)
    duration = time.time() - start_time
    
    if response:
        if 'result' in response:
            print(f"✅ SUCCESS (took {duration:.2f}s)")
            
            # Print result content
            if 'content' in response['result']:
                for content in response['result']['content']:
                    if content.get('type') == 'text':
                        try:
                            data = json.loads(content['text'])
                            print(f"\n📋 Result:")
                            print(json.dumps(data, indent=2))
                        except:
                            print(f"\n📋 Result (text):")
                            print(content['text'])
            return True
        elif 'error' in response:
            print(f"❌ ERROR")
            print(f"   Message: {response['error'].get('message', 'Unknown error')}")
            if 'data' in response['error']:
                print(f"   Details: {response['error']['data']}")
            return False
    else:
        print(f"❌ NO RESPONSE")
        return False

def main():
    """Main test function"""
    neuronmcp_path = '/home/pge/pge/neurondb/bin/neuronmcp/neuronmcp'
    
    print("\n" + "="*70)
    print(" "*15 + "NeuronMCP PostgreSQL Tools")
    print(" "*20 + "Complete Test Suite")
    print("="*70)
    print(f"Test started: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print(f"PostgreSQL Version: 18.1")
    print(f"Database: neurondb")
    print("="*70)
    
    tests = [
        {
            "name": "postgresql_version",
            "args": {},
            "desc": "Get PostgreSQL server version and build information"
        },
        {
            "name": "postgresql_stats",
            "args": {
                "include_database_stats": True,
                "include_table_stats": True,
                "include_connection_stats": True,
                "include_performance_stats": True
            },
            "desc": "Get comprehensive server statistics"
        },
        {
            "name": "postgresql_databases",
            "args": {"include_system": False},
            "desc": "List all user databases"
        },
        {
            "name": "postgresql_databases",
            "args": {"include_system": True},
            "desc": "List all databases including system databases"
        },
        {
            "name": "postgresql_connections",
            "args": {},
            "desc": "Get detailed connection information"
        },
        {
            "name": "postgresql_settings",
            "args": {},
            "desc": "Get all PostgreSQL configuration settings"
        },
        {
            "name": "postgresql_settings",
            "args": {"pattern": "max_"},
            "desc": "Get configuration settings matching 'max_'"
        },
        {
            "name": "postgresql_extensions",
            "args": {},
            "desc": "List all installed PostgreSQL extensions"
        },
        {
            "name": "postgresql_locks",
            "args": {},
            "desc": "Get current lock information"
        },
        {
            "name": "postgresql_replication",
            "args": {},
            "desc": "Get replication status"
        }
    ]
    
    passed = 0
    failed = 0
    
    for test in tests:
        if test_postgresql_tool(
            neuronmcp_path, 
            test["name"], 
            test["args"], 
            test["desc"]
        ):
            passed += 1
        else:
            failed += 1
        time.sleep(0.3)  # Brief pause between tests
    
    print("\n" + "="*70)
    print(" "*25 + "TEST SUMMARY")
    print("="*70)
    print(f"Total tests:  {passed + failed}")
    print(f"✅ Passed:    {passed}")
    print(f"❌ Failed:    {failed}")
    print(f"Success rate: {(passed/(passed+failed)*100):.1f}%")
    print("="*70)
    print(f"Test ended: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print("="*70)
    
    return 0 if failed == 0 else 1

if __name__ == "__main__":
    sys.exit(main())

