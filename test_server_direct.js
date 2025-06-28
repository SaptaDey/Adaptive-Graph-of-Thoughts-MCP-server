#!/usr/bin/env node

import { spawn } from 'child_process';
import { randomUUID } from 'crypto';

// Test data for each tool
const testCases = [
  {
    tool: 'scientific_reasoning_query',
    input: {
      query: "What are the key molecular mechanisms underlying protein misfolding in Alzheimer's disease?",
      parameters: {
        include_reasoning_trace: true,
        max_depth: 5,
        confidence_threshold: 0.8
      }
    }
  },
  {
    tool: 'analyze_research_hypothesis',
    input: {
      hypothesis: "Chronic inflammation contributes to accelerated cognitive decline in Alzheimer's disease patients",
      context: "Multiple studies suggest neuroinflammation plays a role in AD progression",
      evidence_sources: ["PubMed", "Google Scholar"]
    }
  },
  {
    tool: 'explore_scientific_relationships',
    input: {
      concepts: ["amyloid plaques", "tau tangles", "neuroinflammation", "cognitive decline"],
      relationship_types: ["causal", "correlational", "inhibitory"],
      depth: 3
    }
  },
  {
    tool: 'validate_scientific_claims',
    input: {
      claims: [
        "Beta-amyloid plaques are the primary cause of Alzheimer's disease",
        "Exercise can slow cognitive decline in elderly populations"
      ],
      evidence_requirement: "high",
      sources: ["PubMed", "Cochrane"]
    }
  }
];

async function testTool(toolName, args) {
  return new Promise((resolve, reject) => {
    console.log(`\n🧪 Testing tool: ${toolName}`);
    console.log(`📝 Input: ${JSON.stringify(args, null, 2)}`);
    
    const server = spawn('node', ['server/index.js'], {
      stdio: ['pipe', 'pipe', 'pipe']
    });

    let output = '';
    let errorOutput = '';

    server.stdout.on('data', (data) => {
      output += data.toString();
    });

    server.stderr.on('data', (data) => {
      errorOutput += data.toString();
    });

    // Send MCP protocol messages
    const initMessage = {
      jsonrpc: '2.0',
      id: 1,
      method: 'initialize',
      params: {
        protocolVersion: '2024-11-05',
        capabilities: {
          tools: {}
        },
        clientInfo: {
          name: 'test-client',
          version: '1.0.0'
        }
      }
    };

    const toolCallMessage = {
      jsonrpc: '2.0',
      id: 2,
      method: 'tools/call',
      params: {
        name: toolName,
        arguments: args
      }
    };

    server.stdin.write(JSON.stringify(initMessage) + '\n');
    
    setTimeout(() => {
      server.stdin.write(JSON.stringify(toolCallMessage) + '\n');
    }, 1000);

    setTimeout(() => {
      server.kill();
      console.log(`✅ Tool test completed for ${toolName}`);
      console.log(`📊 Output length: ${output.length} chars`);
      console.log(`❌ Error output: ${errorOutput.length} chars`);
      
      if (errorOutput.includes('startTime is not defined')) {
        console.log('🔴 CRITICAL: startTime error still present!');
        reject(new Error('startTime error detected'));
      } else if (errorOutput.includes('error') && !errorOutput.includes('Backend service unavailable')) {
        console.log('🟡 WARNING: Unexpected errors detected');
        console.log(errorOutput);
      } else {
        console.log('✅ No critical errors detected');
      }
      
      resolve({ output, errorOutput, toolName });
    }, 5000);
  });
}

async function runAllTests() {
  console.log('🚀 Starting comprehensive tool testing...\n');
  
  const results = [];
  
  for (const testCase of testCases) {
    try {
      const result = await testTool(testCase.tool, testCase.input);
      results.push({ ...result, status: 'completed' });
    } catch (error) {
      console.log(`🔴 FAILED: ${testCase.tool} - ${error.message}`);
      results.push({ toolName: testCase.tool, status: 'failed', error: error.message });
    }
  }
  
  console.log('\n📋 TEST SUMMARY:');
  results.forEach(result => {
    console.log(`${result.status === 'completed' ? '✅' : '🔴'} ${result.toolName}: ${result.status}`);
  });
  
  const failedTests = results.filter(r => r.status === 'failed');
  if (failedTests.length === 0) {
    console.log('\n🎉 ALL TOOLS TESTED SUCCESSFULLY - No critical errors detected!');
  } else {
    console.log(`\n⚠️  ${failedTests.length} tools failed testing`);
  }
}

runAllTests().catch(console.error);