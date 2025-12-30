#!/usr/bin/env python3
"""
Test script to verify the truncation fix for NeuronMCP
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
        
        stdout, stderr = process.communicate(input=request_json + '\n', timeout=30)
        
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

def test_sampling_with_tool_call():
    """Test sampling/createMessage with a question that requires a tool call"""
    neuronmcp_path = '/home/pge/pge/neurondb/bin/neuronmcp/neuronmcp'
    
    print("="*70)
    print("Testing NeuronMCP Truncation Fix")
    print("="*70)
    
    # Test with a question that should trigger a tool call
    request = {
        "jsonrpc": "2.0",
        "id": 1,
        "method": "sampling/createMessage",
        "params": {
            "messages": [
                {
                    "role": "user",
                    "content": "What version of PostgreSQL are we running?"
                }
            ],
            "model": "llama3:70b",
            "temperature": 0.7,
            "maxTokens": 8192
        }
    }
    
    print("\n📤 Sending request:")
    print(f"   Question: {request['params']['messages'][0]['content']}")
    print(f"   Model: {request['params']['model']}")
    print(f"   Max Tokens: {request['params']['maxTokens']}")
    
    print("\n⏳ Waiting for response (this may take 10-20 seconds)...")
    
    response = send_mcp_request(neuronmcp_path, request)
    
    if response:
        if 'result' in response:
            print("\n✅ SUCCESS - Response received!")
            result = response['result']
            
            if isinstance(result, dict):
                content = result.get('content', '')
                print(f"\n📋 Response content:")
                print(f"   Length: {len(content)} characters")
                print(f"   Content: {content[:200]}...")
                
                # Check if response contains complete tool call
                if 'TOOL_CALL:' in content and '{' in content:
                    # Try to extract and parse the JSON
                    try:
                        json_start = content.index('{')
                        json_part = content[json_start:]
                        # Try to parse it
                        tool_call = json.loads(json_part)
                        print(f"\n✅ TOOL CALL COMPLETE!")
                        print(f"   Tool: {tool_call.get('name', 'N/A')}")
                        print(f"   Arguments: {tool_call.get('arguments', {})}")
                        return True
                    except json.JSONDecodeError as e:
                        print(f"\n❌ TOOL CALL INCOMPLETE/TRUNCATED")
                        print(f"   Error: {e}")
                        print(f"   Content after TOOL_CALL: {content[content.index('TOOL_CALL:'):]}")
                        return False
                else:
                    print(f"\n✅ Direct answer (no tool call needed)")
                    return True
            else:
                print(f"\n📋 Raw result: {result}")
                return True
        elif 'error' in response:
            print(f"\n❌ Error: {response['error'].get('message', 'Unknown')}")
            return False
    else:
        print(f"\n❌ No response received")
        return False

if __name__ == "__main__":
    success = test_sampling_with_tool_call()
    sys.exit(0 if success else 1)

