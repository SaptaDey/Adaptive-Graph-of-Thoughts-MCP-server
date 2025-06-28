#!/usr/bin/env node

/**
 * Direct MCP Server Testing Script
 * Tests the Adaptive Graph of Thoughts MCP server directly
 */

import { spawn } from 'child_process';

class MCPTester {
  constructor() {
    this.serverProcess = null;
    this.tests = [];
  }

  async startServer() {
    console.log('🚀 Starting MCP Server...');
    
    this.serverProcess = spawn('node', ['server/index.js'], {
      stdio: ['pipe', 'pipe', 'pipe'],
      env: {
        ...process.env,
        NEO4J_PASSWORD: 'test_password',
        LOG_LEVEL: 'ERROR'
      }
    });

    // Set up communication
    this.serverProcess.stdout.on('data', (data) => {
      const responses = data.toString().split('\n').filter(line => line.trim());
      responses.forEach(response => {
        if (response.trim()) {
          try {
            const parsed = JSON.parse(response);
            this.handleResponse(parsed);
          } catch (e) {
            console.log('Server output:', response);
          }
        }
      });
    });

    this.serverProcess.stderr.on('data', (data) => {
      // Ignore stderr (logs) during testing
    });

    // Wait a moment for server to start
    await new Promise(resolve => setTimeout(resolve, 1000));
  }

  sendRequest(request) {
    return new Promise((resolve, reject) => {
      const timeout = setTimeout(() => {
        reject(new Error('Request timeout'));
      }, 5000);

      this.pendingRequest = { resolve, reject, timeout };
      
      this.serverProcess.stdin.write(JSON.stringify(request) + '\n');
    });
  }

  handleResponse(response) {
    if (this.pendingRequest) {
      clearTimeout(this.pendingRequest.timeout);
      this.pendingRequest.resolve(response);
      this.pendingRequest = null;
    }
  }

  async runTest(name, testFn) {
    console.log(`\n🧪 Test: ${name}`);
    try {
      await testFn();
      console.log(`✅ PASSED: ${name}`);
      this.tests.push({ name, status: 'PASSED' });
    } catch (error) {
      console.log(`❌ FAILED: ${name} - ${error.message}`);
      this.tests.push({ name, status: 'FAILED', error: error.message });
    }
  }

  async testInitialize() {
    const request = {
      jsonrpc: '2.0',
      method: 'initialize',
      params: {
        protocolVersion: '2025-06-18',
        capabilities: {},
        clientInfo: {
          name: 'test-client',
          version: '1.0.0'
        }
      },
      id: 1
    };

    const response = await this.sendRequest(request);
    
    if (response.error) {
      throw new Error(`Initialize error: ${response.error.message}`);
    }
    
    if (!response.result) {
      throw new Error('No result in initialize response');
    }
    
    if (!response.result.capabilities) {
      throw new Error('No capabilities in initialize response');
    }
    
    console.log(`   Server: ${response.result.serverInfo.name} v${response.result.serverInfo.version}`);
    console.log(`   Protocol: ${response.result.protocolVersion}`);
  }

  async testToolsList() {
    const request = {
      jsonrpc: '2.0',
      method: 'tools/list',
      params: {},
      id: 2
    };

    const response = await this.sendRequest(request);
    
    if (!response.result || !response.result.tools) {
      throw new Error('No tools in response');
    }
    
    const tools = response.result.tools;
    if (tools.length !== 4) {
      throw new Error(`Expected 4 tools, got ${tools.length}`);
    }
    
    const expectedTools = [
      'scientific_reasoning_query',
      'analyze_research_hypothesis', 
      'explore_scientific_relationships',
      'validate_scientific_claims'
    ];
    
    for (const expectedTool of expectedTools) {
      if (!tools.find(t => t.name === expectedTool)) {
        throw new Error(`Missing tool: ${expectedTool}`);
      }
    }
    
    console.log(`   Found all ${tools.length} expected tools`);
  }

  async testPromptsList() {
    const request = {
      jsonrpc: '2.0',
      method: 'prompts/list',
      params: {},
      id: 3
    };

    const response = await this.sendRequest(request);
    
    if (!response.result || !response.result.prompts) {
      throw new Error('No prompts in response');
    }
    
    const prompts = response.result.prompts;
    if (prompts.length !== 3) {
      throw new Error(`Expected 3 prompts, got ${prompts.length}`);
    }
    
    const expectedPrompts = [
      'analyze_research_question',
      'hypothesis_generator',
      'literature_synthesis'
    ];
    
    for (const expectedPrompt of expectedPrompts) {
      if (!prompts.find(p => p.name === expectedPrompt)) {
        throw new Error(`Missing prompt: ${expectedPrompt}`);
      }
    }
    
    console.log(`   Found all ${prompts.length} expected prompts`);
  }

  async testToolCall() {
    const request = {
      jsonrpc: '2.0',
      method: 'tools/call',
      params: {
        name: 'scientific_reasoning_query',
        arguments: {
          query: 'Test scientific query for MCP validation'
        }
      },
      id: 4
    };

    const response = await this.sendRequest(request);
    
    if (response.error) {
      throw new Error(`Tool call failed: ${response.error.message}`);
    }
    
    if (!response.result) {
      throw new Error('No result in tool call response');
    }
    
    if (!response.result.content || !Array.isArray(response.result.content)) {
      throw new Error('Tool response missing content array');
    }
    
    const content = response.result.content[0];
    if (content.type !== 'text' || !content.text) {
      throw new Error('Tool response missing text content');
    }
    
    // Verify the response contains expected ASR-GoT structure
    const result = JSON.parse(content.text);
    if (!result.framework || result.framework !== 'ASR-GoT') {
      throw new Error('Tool response missing ASR-GoT framework identifier');
    }
    
    if (!result.stages || !result.metadata) {
      throw new Error('Tool response missing required ASR-GoT structure');
    }
    
    console.log('   Tool executed successfully with ASR-GoT framework response');
  }

  async testPromptGet() {
    const request = {
      jsonrpc: '2.0',
      method: 'prompts/get',
      params: {
        name: 'analyze_research_question',
        arguments: {
          research_question: 'How does machine learning impact scientific research?',
          domain: 'Computer Science'
        }
      },
      id: 5
    };

    const response = await this.sendRequest(request);
    
    if (!response.result) {
      throw new Error('No result in prompt response');
    }
    
    if (!response.result.messages || response.result.messages.length === 0) {
      throw new Error('No messages in prompt response');
    }
    
    const message = response.result.messages[0];
    if (!message.content || !message.content.text) {
      throw new Error('No text content in prompt message');
    }
    
    if (!message.content.text.includes('Adaptive Graph of Thoughts framework')) {
      throw new Error('Prompt content does not match expected template');
    }
    
    console.log('   Prompt generated correctly with expected template');
  }

  async testHypothesisGeneratorPrompt() {
    const request = {
      jsonrpc: '2.0',
      method: 'prompts/get',
      params: {
        name: 'hypothesis_generator',
        arguments: {
          problem_statement: 'How do neural networks learn complex patterns?',
          constraints: 'Focus on biological plausibility'
        }
      },
      id: 6
    };

    const response = await this.sendRequest(request);
    
    if (!response.result) {
      throw new Error('No result in hypothesis_generator prompt response');
    }
    
    if (!response.result.messages || response.result.messages.length === 0) {
      throw new Error('No messages in hypothesis_generator prompt response');
    }
    
    const message = response.result.messages[0];
    if (!message.content || !message.content.text) {
      throw new Error('No text content in hypothesis_generator prompt message');
    }
    
    const text = message.content.text;
    if (!text.includes('testable hypotheses')) {
      throw new Error('Prompt content does not match expected template');
    }
    
    if (!text.includes('confidence scoring')) {
      throw new Error('Prompt content does not match expected template');
    }
    
    console.log('   Hypothesis generator prompt generated correctly with expected template');
  }

  async runAllTests() {
    console.log('🧪 MCP Server Functionality Tests');
    console.log('==================================');

    try {
      await this.startServer();
      
      await this.runTest('Initialize Server', () => this.testInitialize());
      await this.runTest('List Tools', () => this.testToolsList());
      await this.runTest('List Prompts', () => this.testPromptsList());
      await this.runTest('Call Tool (ASR-GoT Framework)', () => this.testToolCall());
      await this.runTest('Get Prompt', () => this.testPromptGet());
      await this.runTest('Hypothesis Generator Prompt Fix', () => this.testHypothesisGeneratorPrompt());
      
    } finally {
      if (this.serverProcess) {
        this.serverProcess.kill();
      }
    }

    console.log('\n📊 Test Summary');
    console.log('================');
    
    const passed = this.tests.filter(t => t.status === 'PASSED').length;
    const failed = this.tests.filter(t => t.status === 'FAILED').length;
    
    console.log(`✅ Passed: ${passed}`);
    console.log(`❌ Failed: ${failed}`);
    console.log(`📋 Total: ${this.tests.length}`);
    
    if (failed === 0) {
      console.log('\n🎉 All MCP functionality tests PASSED!');
      console.log('✅ Server is ready for production use');
    } else {
      console.log('\n⚠️  Some tests failed - review above for details');
    }
    
    return failed === 0;
  }
}

// Run tests if called directly
if (import.meta.url === `file://${process.argv[1]}`) {
  const tester = new MCPTester();
  tester.runAllTests().then(success => {
    process.exit(success ? 0 : 1);
  }).catch(error => {
    console.error('Test runner error:', error);
    process.exit(1);
  });
}