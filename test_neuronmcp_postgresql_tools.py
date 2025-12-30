#!/usr/bin/env python3
"""
Test script for NeuronMCP PostgreSQL tools
Tests various PostgreSQL administration and monitoring tools via MCP protocol
"""

import subprocess
import json
import sys
import time

def send_mcp_request(neuronmcp_path, request):
    """Send an MCP request and get response"""
    try:
        # Convert request to JSON
        request_json = json.dumps(request)
        
        # Start neuronmcp process
        process = subprocess.Popen(
            [neuronmcp_path],
            stdin=subprocess.PIPE,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
            cwd='/home/pge/pge/neurondb/bin/neuronmcp'
        )
        
        # Send request and get response
        stdout, stderr = process.communicate(input=request_json + '\n', timeout=10)
        
        # Print stderr for debugging
        if stderr:
            print(f"STDERR: {stderr}", file=sys.stderr)
        
        # Parse response
        if stdout.strip():
            lines = stdout.strip().split('\n')
            # Try to parse each line as JSON
            for line in lines:
                try:
                    response = json.loads(line)
                    return response
                except json.JSONDecodeError:
                    continue
        
        return None
        
    except subprocess.TimeoutExpired:
        process.kill()
        print("Request timed out", file=sys.stderr)
        return None
    except Exception as e:
        print(f"Error: {e}", file=sys.stderr)
        return None

def initialize_mcp(neuronmcp_path):
    """Initialize MCP connection"""
    init_request = {
        "jsonrpc": "2.0",
        "id": 1,
        "method": "initialize",
        "params": {
            "protocolVersion": "2024-11-05",
            "capabilities": {
                "roots": {
                    "listChanged": True
                },
                "sampling": {}
            },
            "clientInfo": {
                "name": "test-client",
                "version": "1.0.0"
            }
        }
    }
    
    print("🔌 Initializing MCP connection...")
    response = send_mcp_request(neuronmcp_path, init_request)
    if response and 'result' in response:
        print("✅ MCP initialized successfully")
        return True
    else:
        print("❌ Failed to initialize MCP")
        print(f"Response: {response}")
        return False

def test_postgresql_tool(neuronmcp_path, tool_name, arguments=None):
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
    
    print(f"\n📊 Testing tool: {tool_name}")
    print(f"   Arguments: {json.dumps(arguments)}")
    
    response = send_mcp_request(neuronmcp_path, request)
    
    if response:
        if 'result' in response:
            print(f"✅ Success!")
            # Print result content
            if 'content' in response['result']:
                for content in response['result']['content']:
                    if content.get('type') == 'text':
                        try:
                            data = json.loads(content['text'])
                            print(f"   Result: {json.dumps(data, indent=2)[:500]}...")
                        except:
                            print(f"   Result: {content['text'][:500]}...")
            return True
        elif 'error' in response:
            print(f"❌ Error: {response['error'].get('message', 'Unknown error')}")
            return False
    else:
        print(f"❌ No response received")
        return False

def main():
    """Main test function"""
    neuronmcp_path = '/home/pge/pge/neurondb/bin/neuronmcp/neuronmcp'
    
    print("=" * 70)
    print("NeuronMCP PostgreSQL Tools Test Suite")
    print("=" * 70)
    
    # Test various PostgreSQL tools
    tests = [
        ("postgresql_version", {}),
        ("postgresql_stats", {
            "include_database_stats": True,
            "include_table_stats": True,
            "include_connection_stats": True,
            "include_performance_stats": True
        }),
        ("postgresql_databases", {"include_system": False}),
        ("postgresql_tables", {"schema_name": "public"}),
        ("postgresql_connections", {}),
        ("postgresql_settings", {"setting_name": "max_connections"}),
        ("postgresql_extensions", {}),
        ("postgresql_active_queries", {}),
    ]
    
    passed = 0
    failed = 0
    
    for tool_name, arguments in tests:
        if test_postgresql_tool(neuronmcp_path, tool_name, arguments):
            passed += 1
        else:
            failed += 1
        time.sleep(0.5)  # Brief pause between tests
    
    print("\n" + "=" * 70)
    print(f"Test Results: {passed} passed, {failed} failed")
    print("=" * 70)
    
    return 0 if failed == 0 else 1

if __name__ == "__main__":
    sys.exit(main())

