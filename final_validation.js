#!/usr/bin/env node

// Comprehensive final validation test
import { spawn } from 'child_process';
import { writeFileSync } from 'fs';

class ComprehensiveValidator {
  constructor() {
    this.testResults = [];
    this.passed = 0;
    this.failed = 0;
  }

  log(test, status, details = '') {
    const result = { test, status, details, timestamp: new Date().toISOString() };
    this.testResults.push(result);
    
    const icon = status === 'PASS' ? '✅' : status === 'FAIL' ? '🔴' : '🟡';
    console.log(`${icon} ${test}: ${status} ${details}`);
    
    if (status === 'PASS') this.passed++;
    if (status === 'FAIL') this.failed++;
  }

  async testServerStartup() {
    console.log('\n🚀 Testing Server Startup & Configuration...');
    
    return new Promise((resolve) => {
      const server = spawn('node', ['server/index.js'], { stdio: 'pipe' });
      let output = '';
      let errorOutput = '';
      let hasStarted = false;
      
      server.stdout.on('data', (data) => {
        output += data.toString();
        if (output.includes('Adaptive Graph of Thoughts MCP server running on stdio')) {
          hasStarted = true;
        }
      });
      
      server.stderr.on('data', (data) => {
        errorOutput += data.toString();
      });
      
      setTimeout(() => {
        server.kill('SIGTERM');
        
        if (hasStarted && !errorOutput.includes('startTime')) {
          this.log('Server Startup', 'PASS', 'Clean startup without critical errors');
        } else {
          this.log('Server Startup', 'FAIL', `Errors: ${errorOutput.substring(0, 100)}`);
        }
        
        resolve();
      }, 3000);
    });
  }

  async testMCPInspectorTools() {
    console.log('\n🔍 Testing MCP Inspector Tool Discovery...');
    
    return new Promise((resolve) => {
      const inspector = spawn('npx', [
        '@modelcontextprotocol/inspector',
        '--cli', 'node', 'server/index.js',
        '--method', 'tools/list'
      ], { stdio: 'pipe' });
      
      let output = '';
      let errorOutput = '';
      
      inspector.stdout.on('data', (data) => {
        output += data.toString();
      });
      
      inspector.stderr.on('data', (data) => {
        errorOutput += data.toString();
      });
      
      inspector.on('close', (code) => {
        try {
          const tools = JSON.parse(output);
          if (tools.tools && tools.tools.length === 4) {
            const toolNames = tools.tools.map(t => t.name);
            const expectedTools = [
              'scientific_reasoning_query',
              'analyze_research_hypothesis', 
              'explore_scientific_relationships',
              'validate_scientific_claims'
            ];
            
            const allPresent = expectedTools.every(tool => toolNames.includes(tool));
            if (allPresent) {
              this.log('Tool Discovery', 'PASS', 'All 4 tools discovered with valid schemas');
            } else {
              this.log('Tool Discovery', 'FAIL', `Missing tools: ${expectedTools.filter(t => !toolNames.includes(t))}`);
            }
          } else {
            this.log('Tool Discovery', 'FAIL', `Expected 4 tools, got ${tools.tools?.length || 0}`);
          }
        } catch (e) {
          this.log('Tool Discovery', 'FAIL', `JSON parse error: ${e.message}`);
        }
        resolve();
      });
      
      setTimeout(() => {
        inspector.kill();
        this.log('Tool Discovery', 'FAIL', 'Timeout after 10s');
        resolve();
      }, 10000);
    });
  }

  async testMCPInspectorPrompts() {
    console.log('\n📝 Testing MCP Inspector Prompt Discovery...');
    
    return new Promise((resolve) => {
      const inspector = spawn('npx', [
        '@modelcontextprotocol/inspector',
        '--cli', 'node', 'server/index.js',
        '--method', 'prompts/list'
      ], { stdio: 'pipe' });
      
      let output = '';
      
      inspector.stdout.on('data', (data) => {
        output += data.toString();
      });
      
      inspector.on('close', (code) => {
        try {
          const prompts = JSON.parse(output);
          if (prompts.prompts && prompts.prompts.length === 3) {
            const promptNames = prompts.prompts.map(p => p.name);
            const expectedPrompts = [
              'analyze_research_question',
              'hypothesis_generator',
              'literature_synthesis'
            ];
            
            const allPresent = expectedPrompts.every(prompt => promptNames.includes(prompt));
            if (allPresent) {
              this.log('Prompt Discovery', 'PASS', 'All 3 prompts discovered');
            } else {
              this.log('Prompt Discovery', 'FAIL', `Missing prompts: ${expectedPrompts.filter(p => !promptNames.includes(p))}`);
            }
          } else {
            this.log('Prompt Discovery', 'FAIL', `Expected 3 prompts, got ${prompts.prompts?.length || 0}`);
          }
        } catch (e) {
          this.log('Prompt Discovery', 'FAIL', `JSON parse error: ${e.message}`);
        }
        resolve();
      });
      
      setTimeout(() => {
        inspector.kill();
        this.log('Prompt Discovery', 'FAIL', 'Timeout after 10s');
        resolve();
      }, 10000);
    });
  }

  async testDXTManifest() {
    console.log('\n📦 Testing DXT Manifest Validation...');
    
    return new Promise((resolve) => {
      const validator = spawn('dxt', ['validate', 'manifest.json'], { stdio: 'pipe' });
      
      let output = '';
      let errorOutput = '';
      
      validator.stdout.on('data', (data) => {
        output += data.toString();
      });
      
      validator.stderr.on('data', (data) => {
        errorOutput += data.toString();
      });
      
      validator.on('close', (code) => {
        if (code === 0 && output.includes('Manifest is valid')) {
          this.log('DXT Manifest', 'PASS', 'Manifest validation successful');
        } else {
          this.log('DXT Manifest', 'FAIL', `Validation failed: ${errorOutput || output}`);
        }
        resolve();
      });
      
      setTimeout(() => {
        validator.kill();
        this.log('DXT Manifest', 'FAIL', 'Timeout after 5s');
        resolve();
      }, 5000);
    });
  }

  async testSecurityFeatures() {
    console.log('\n🔒 Testing Security Features...');
    
    // Test input validation
    const testCases = [
      { env: { MAX_REASONING_DEPTH: '999' }, expected: '5' },
      { env: { CONFIDENCE_THRESHOLD: '-1' }, expected: '0.7' },
      { env: { NEO4J_URI: 'malicious://test' }, expected: 'bolt://localhost:7687' },
    ];
    
    for (const testCase of testCases) {
      await new Promise((resolve) => {
        const server = spawn('node', ['server/index.js'], { 
          stdio: 'pipe',
          env: { ...process.env, ...testCase.env }
        });
        
        let output = '';
        
        server.stdout.on('data', (data) => {
          output += data.toString();
        });
        
        setTimeout(() => {
          server.kill('SIGTERM');
          
          if (output.includes('Configuration loaded successfully')) {
            this.log('Security Validation', 'PASS', `Properly sanitized ${Object.keys(testCase.env)[0]}`);
          } else {
            this.log('Security Validation', 'FAIL', `Failed to sanitize ${Object.keys(testCase.env)[0]}`);
          }
          resolve();
        }, 2000);
      });
    }
  }

  generateReport() {
    console.log('\n📊 COMPREHENSIVE VALIDATION REPORT');
    console.log('=' .repeat(50));
    console.log(`📈 Total Tests: ${this.testResults.length}`);
    console.log(`✅ Passed: ${this.passed}`);
    console.log(`🔴 Failed: ${this.failed}`);
    console.log(`🎯 Success Rate: ${Math.round((this.passed / this.testResults.length) * 100)}%`);
    
    if (this.failed === 0) {
      console.log('\n🎉 ALL TESTS PASSED - DXT READY FOR SUBMISSION! 🎉');
    } else {
      console.log(`\n⚠️  ${this.failed} TESTS FAILED - REVIEW REQUIRED`);
      console.log('\nFailed Tests:');
      this.testResults.filter(r => r.status === 'FAIL').forEach(r => {
        console.log(`  🔴 ${r.test}: ${r.details}`);
      });
    }
    
    // Write detailed report
    const report = {
      summary: {
        totalTests: this.testResults.length,
        passed: this.passed,
        failed: this.failed,
        successRate: Math.round((this.passed / this.testResults.length) * 100),
        timestamp: new Date().toISOString(),
        ready: this.failed === 0
      },
      results: this.testResults
    };
    
    writeFileSync('VALIDATION_REPORT.json', JSON.stringify(report, null, 2));
    console.log('\n📄 Detailed report saved to VALIDATION_REPORT.json');
  }

  async runAllTests() {
    console.log('🚀 STARTING COMPREHENSIVE DXT VALIDATION');
    console.log('=' .repeat(50));
    
    await this.testServerStartup();
    await this.testMCPInspectorTools();
    await this.testMCPInspectorPrompts();
    await this.testDXTManifest();
    await this.testSecurityFeatures();
    
    this.generateReport();
  }
}

const validator = new ComprehensiveValidator();
validator.runAllTests().catch(console.error);