#!/usr/bin/env python3
"""
Test script to check available PostgreSQL tools in NeuronMCP
"""

import subprocess
import json
import sys

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

def list_tools(neuronmcp_path):
    """List all available tools"""
    request = {
        "jsonrpc": "2.0",
        "id": 1,
        "method": "tools/list",
        "params": {}
    }
    
    response = send_mcp_request(neuronmcp_path, request)
    
    if response and 'result' in response:
        return response['result'].get('tools', [])
    
    return []

def main():
    neuronmcp_path = '/home/pge/pge/neurondb/bin/neuronmcp/neuronmcp'
    
    print("=" * 70)
    print("Available PostgreSQL Tools in NeuronMCP")
    print("=" * 70)
    
    tools = list_tools(neuronmcp_path)
    
    # Filter PostgreSQL tools
    postgresql_tools = [t for t in tools if t['name'].startswith('postgresql_')]
    
    print(f"\nTotal tools: {len(tools)}")
    print(f"PostgreSQL tools: {len(postgresql_tools)}\n")
    
    print("Available PostgreSQL Tools:")
    print("-" * 70)
    
    for tool in sorted(postgresql_tools, key=lambda x: x['name']):
        print(f"\n📊 {tool['name']}")
        if 'description' in tool:
            print(f"   Description: {tool['description'][:100]}...")
        if 'inputSchema' in tool and 'properties' in tool['inputSchema']:
            params = list(tool['inputSchema']['properties'].keys())
            if params:
                print(f"   Parameters: {', '.join(params)}")
    
    print("\n" + "=" * 70)
    
    return 0

if __name__ == "__main__":
    sys.exit(main())

