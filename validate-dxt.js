#!/usr/bin/env node

/**
 * DXT Validation Script
 * Tests the Adaptive Graph of Thoughts DXT extension
 */

import { spawn } from 'child_process';
import { readFileSync } from 'fs';
import { join, dirname } from 'path';
import { fileURLToPath } from 'url';

const __filename = fileURLToPath(import.meta.url);
const __dirname = dirname(__filename);

class DXTValidator {
  constructor() {
    this.serverProcess = null;
    this.results = {
      manifestValid: false,
      serverStarts: false,
      mcpHandshake: false,
      toolsAvailable: false,
      errors: []
    };
  }

  log(message, level = 'INFO') {
    const timestamp = new Date().toISOString();
    const prefix = level === 'ERROR' ? '❌' : level === 'WARN' ? '⚠️' : level === 'SUCCESS' ? '✅' : 'ℹ️';
    console.log(`${prefix} [${timestamp}] ${message}`);
  }

  async validateManifest() {
    this.log('Validating manifest.json...');
    
    try {
      const manifestPath = join(__dirname, 'manifest.json');
      const manifestContent = readFileSync(manifestPath, 'utf8');
      const manifest = JSON.parse(manifestContent);
      
      // Check required fields
      const requiredFields = ['dxt_version', 'name', 'version', 'description', 'author', 'server'];
      const missingFields = requiredFields.filter(field => !manifest[field]);
      
      if (missingFields.length > 0) {
        throw new Error(`Missing required fields: ${missingFields.join(', ')}`);
      }
      
      // Check server configuration
      if (!manifest.server.entry_point || !manifest.server.mcp_config) {
        throw new Error('Invalid server configuration in manifest');
      }
      
      // Check tools array
      if (!Array.isArray(manifest.tools) || manifest.tools.length === 0) {
        throw new Error('No tools defined in manifest');
      }
      
      this.results.manifestValid = true;
      this.log('Manifest validation successful', 'SUCCESS');
      return true;
    } catch (error) {
      this.results.errors.push(`Manifest validation failed: ${error.message}`);
      this.log(`Manifest validation failed: ${error.message}`, 'ERROR');
      return false;
    }
  }

  async startServer() {
    this.log('Starting DXT server...');
    
    return new Promise((resolve) => {
      const serverPath = join(__dirname, 'server', 'index.js');
      this.serverProcess = spawn('node', [serverPath], {
        stdio: ['pipe', 'pipe', 'pipe'],
        env: {
          ...process.env,
          LOG_LEVEL: 'ERROR', // Reduce noise during testing
          NEO4J_PASSWORD: process.env.NEO4J_PASSWORD || 'test', // Provide default for testing
        }
      });

      let serverOutput = '';
      let serverReady = false;

      // Capture server output
      this.serverProcess.stdout?.on('data', (data) => {
        serverOutput += data.toString();
      });

      this.serverProcess.stderr?.on('data', (data) => {
        const output = data.toString();
        serverOutput += output;
        
        // Check if server is ready
        if (output.includes('MCP server running') || output.includes('Server initialization complete')) {
          serverReady = true;
        }
      });

      this.serverProcess.on('error', (error) => {
        this.results.errors.push(`Server start failed: ${error.message}`);
        this.log(`Server start failed: ${error.message}`, 'ERROR');
        resolve(false);
      });

      // Give server time to start
      setTimeout(() => {
        if (this.serverProcess && !this.serverProcess.killed) {
          this.results.serverStarts = true;
          this.log('Server started successfully', 'SUCCESS');
          resolve(true);
        } else {
          this.results.errors.push('Server failed to start within timeout');
          this.log('Server failed to start within timeout', 'ERROR');
          resolve(false);
        }
      }, 5000);
    });
  }

  async testMCPHandshake() {
    this.log('Testing MCP handshake...');
    
    if (!this.serverProcess) {
      this.log('Server not running for handshake test', 'ERROR');
      return false;
    }

    return new Promise((resolve) => {
      let responseData = '';
      let handshakeComplete = false;

      // Listen for server responses
      this.serverProcess.stdout?.on('data', (data) => {
        responseData += data.toString();
        
        // Check for initialize response
        try {
          const lines = responseData.split('\n').filter(line => line.trim());
          for (const line of lines) {
            if (line.trim()) {
              const response = JSON.parse(line);
              if (response.id === 1 && response.result) {
                handshakeComplete = true;
                this.results.mcpHandshake = true;
                this.log('MCP handshake successful', 'SUCCESS');
                resolve(true);
                return;
              }
            }
          }
        } catch (error) {
          // Ignore JSON parse errors for non-JSON output
        }
      });

      // Send initialize request
      const initRequest = {
        jsonrpc: '2.0',
        method: 'initialize',
        params: {
          protocolVersion: '0.1.0',
          capabilities: {}
        },
        id: 1
      };

      this.serverProcess.stdin?.write(JSON.stringify(initRequest) + '\n');

      // Timeout after 3 seconds
      setTimeout(() => {
        if (!handshakeComplete) {
          this.results.errors.push('MCP handshake timeout');
          this.log('MCP handshake timeout', 'ERROR');
          resolve(false);
        }
      }, 3000);
    });
  }

  async testToolsAvailable() {
    this.log('Testing tools availability...');
    
    if (!this.serverProcess) {
      this.log('Server not running for tools test', 'ERROR');
      return false;
    }

    return new Promise((resolve) => {
      let responseData = '';
      let toolsFound = false;

      // Listen for server responses
      this.serverProcess.stdout?.on('data', (data) => {
        responseData += data.toString();
        
        // Check for tools list response
        try {
          const lines = responseData.split('\n').filter(line => line.trim());
          for (const line of lines) {
            if (line.trim()) {
              const response = JSON.parse(line);
              if (response.id === 2 && response.result?.tools) {
                const tools = response.result.tools;
                if (Array.isArray(tools) && tools.length > 0) {
                  toolsFound = true;
                  this.results.toolsAvailable = true;
                  this.log(`Found ${tools.length} tools available`, 'SUCCESS');
                  tools.forEach(tool => {
                    this.log(`  - ${tool.name}: ${tool.description}`);
                  });
                  resolve(true);
                  return;
                }
              }
            }
          }
        } catch (error) {
          // Ignore JSON parse errors for non-JSON output
        }
      });

      // Send tools list request
      const toolsRequest = {
        jsonrpc: '2.0',
        method: 'tools/list',
        params: {},
        id: 2
      };

      this.serverProcess.stdin?.write(JSON.stringify(toolsRequest) + '\n');

      // Timeout after 3 seconds
      setTimeout(() => {
        if (!toolsFound) {
          this.results.errors.push('Tools list request timeout');
          this.log('Tools list request timeout', 'ERROR');
          resolve(false);
        }
      }, 3000);
    });
  }

  async cleanup() {
    this.log('Cleaning up...');
    
    if (this.serverProcess && !this.serverProcess.killed) {
      this.serverProcess.kill('SIGTERM');
      
      // Wait for graceful shutdown
      await new Promise(resolve => {
        setTimeout(() => {
          if (this.serverProcess && !this.serverProcess.killed) {
            this.serverProcess.kill('SIGKILL');
          }
          resolve();
        }, 2000);
      });
    }
    
    this.log('Cleanup complete');
  }

  async run() {
    this.log('🧪 Starting DXT Validation');
    this.log('===========================');

    try {
      // Validate manifest
      if (!await this.validateManifest()) {
        return this.generateReport();
      }

      // Start server
      if (!await this.startServer()) {
        return this.generateReport();
      }

      // Test MCP handshake
      if (!await this.testMCPHandshake()) {
        return this.generateReport();
      }

      // Test tools availability
      await this.testToolsAvailable();

    } catch (error) {
      this.results.errors.push(`Validation error: ${error.message}`);
      this.log(`Validation error: ${error.message}`, 'ERROR');
    } finally {
      await this.cleanup();
    }

    return this.generateReport();
  }

  generateReport() {
    this.log('📊 Validation Report');
    this.log('===================');
    
    const checks = [
      { name: 'Manifest Valid', passed: this.results.manifestValid },
      { name: 'Server Starts', passed: this.results.serverStarts },
      { name: 'MCP Handshake', passed: this.results.mcpHandshake },
      { name: 'Tools Available', passed: this.results.toolsAvailable },
    ];

    checks.forEach(check => {
      const status = check.passed ? '✅ PASS' : '❌ FAIL';
      this.log(`${status} ${check.name}`);
    });

    if (this.results.errors.length > 0) {
      this.log('\n🚨 Errors encountered:');
      this.results.errors.forEach((error, index) => {
        this.log(`${index + 1}. ${error}`, 'ERROR');
      });
    }

    const passedChecks = checks.filter(c => c.passed).length;
    const totalChecks = checks.length;
    
    this.log(`\n📈 Summary: ${passedChecks}/${totalChecks} checks passed`);
    
    if (passedChecks === totalChecks) {
      this.log('🎉 All validations passed! The DXT extension is ready for use.', 'SUCCESS');
      return true;
    } else {
      this.log('⚠️ Some validations failed. Please address the errors above.', 'WARN');
      return false;
    }
  }
}

// Run validation if called directly
if (import.meta.url === `file://${process.argv[1]}`) {
  const validator = new DXTValidator();
  
  validator.run().then(success => {
    process.exit(success ? 0 : 1);
  }).catch(error => {
    console.error('❌ Validation failed with error:', error);
    process.exit(1);
  });
}

export { DXTValidator };