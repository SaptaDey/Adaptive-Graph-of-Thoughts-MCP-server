#!/usr/bin/env node

import { spawn } from 'child_process';

async function testStandaloneTools() {
  console.log('🧪 Testing Standalone ASR-GoT Tools...\n');
  
  const testCases = [
    {
      name: "Scientific Reasoning Query",
      request: {
        jsonrpc: "2.0",
        id: 1,
        method: "tools/call",
        params: {
          name: "scientific_reasoning_query",
          arguments: {
            query: "What are the molecular mechanisms of protein folding in neurodegenerative diseases?",
            parameters: {
              include_reasoning_trace: true,
              max_depth: 3,
              confidence_threshold: 0.8
            }
          }
        }
      }
    },
    {
      name: "Analyze Research Hypothesis",
      request: {
        jsonrpc: "2.0",
        id: 2,
        method: "tools/call",
        params: {
          name: "analyze_research_hypothesis",
          arguments: {
            hypothesis: "Machine learning can improve diagnostic accuracy in dermatology",
            context: "Recent advances in computer vision and medical imaging"
          }
        }
      }
    },
    {
      name: "Explore Scientific Relationships",
      request: {
        jsonrpc: "2.0",
        id: 3,
        method: "tools/call",
        params: {
          name: "explore_scientific_relationships",
          arguments: {
            concepts: ["immunology", "machine learning", "personalized medicine"],
            depth: 2
          }
        }
      }
    },
    {
      name: "Validate Scientific Claims",
      request: {
        jsonrpc: "2.0",
        id: 4,
        method: "tools/call",
        params: {
          name: "validate_scientific_claims",
          arguments: {
            claims: ["The extension now works in standalone mode", "ASR-GoT provides comprehensive scientific reasoning"],
            evidence_requirement: "medium"
          }
        }
      }
    }
  ];

  for (const testCase of testCases) {
    console.log(`🔬 Testing: ${testCase.name}`);
    
    const server = spawn('node', ['server/index.js'], {
      stdio: ['pipe', 'pipe', 'pipe']
    });

    let output = '';
    let errorOutput = '';
    let responseReceived = false;

    server.stdout.on('data', (data) => {
      output += data.toString();
      
      // Look for JSON-RPC response
      const lines = output.split('\n');
      for (const line of lines) {
        if (line.trim() && line.includes('"jsonrpc"') && line.includes('"result"')) {
          try {
            const response = JSON.parse(line);
            if (response.result && response.result.content) {
              console.log(`✅ ${testCase.name}: SUCCESS`);
              console.log(`📊 Response length: ${JSON.stringify(response.result).length} chars`);
              responseReceived = true;
              
              // Parse the tool response to check for ASR-GoT framework
              const toolResponse = JSON.parse(response.result.content[0].text);
              if (toolResponse.framework === "ASR-GoT" || toolResponse.analysis_type || toolResponse.exploration_type || toolResponse.validation_type) {
                console.log(`🧬 ASR-GoT Framework: ACTIVE`);
              }
              
              break;
            }
          } catch (e) {
            // Continue parsing
          }
        }
      }
    });

    server.stderr.on('data', (data) => {
      errorOutput += data.toString();
    });

    // Send initialization
    const initRequest = {
      jsonrpc: "2.0",
      id: 0,
      method: "initialize",
      params: {
        protocolVersion: "2024-11-05",
        capabilities: {},
        clientInfo: { name: "test-client", version: "1.0.0" }
      }
    };

    server.stdin.write(JSON.stringify(initRequest) + '\n');
    
    // Wait a moment then send the tool request
    setTimeout(() => {
      server.stdin.write(JSON.stringify(testCase.request) + '\n');
    }, 1000);

    // Wait for response or timeout
    await new Promise(resolve => {
      setTimeout(() => {
        server.kill();
        if (!responseReceived) {
          console.log(`❌ ${testCase.name}: TIMEOUT or ERROR`);
          if (errorOutput.includes('ECONNRESET')) {
            console.log(`🔴 CRITICAL: Still getting ECONNRESET errors!`);
          } else if (errorOutput) {
            console.log(`⚠️  Error: ${errorOutput.substring(0, 100)}...`);
          }
        }
        resolve();
      }, 5000);
    });

    console.log('');
  }

  console.log('🎯 Standalone Tool Testing Complete!');
}

testStandaloneTools().catch(console.error);