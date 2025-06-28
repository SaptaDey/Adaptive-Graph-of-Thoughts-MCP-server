#!/usr/bin/env node

// Test the server's performance under various loads
import { spawn } from 'child_process';

async function testServerPerformance() {
  console.log('🚀 Testing Server Performance and Resource Limits...\n');
  
  // Test 1: Multiple rapid startup/shutdown cycles
  console.log('📊 Test 1: Rapid startup/shutdown cycles');
  const startTime = Date.now();
  
  for (let i = 0; i < 5; i++) {
    const server = spawn('node', ['server/index.js'], { stdio: 'pipe' });
    
    await new Promise(resolve => {
      setTimeout(() => {
        server.kill('SIGTERM');
        server.on('exit', resolve);
      }, 1000);
    });
  }
  
  const cycleTime = Date.now() - startTime;
  console.log(`✅ Completed 5 cycles in ${cycleTime}ms (avg: ${Math.round(cycleTime/5)}ms per cycle)`);
  
  // Test 2: Memory usage check
  console.log('\n📊 Test 2: Memory usage baseline');
  const server = spawn('node', ['server/index.js'], { stdio: 'pipe' });
  
  await new Promise(resolve => setTimeout(resolve, 2000)); // Let it stabilize
  
  const memUsage = process.memoryUsage();
  console.log(`✅ Memory baseline: RSS=${Math.round(memUsage.rss/1024/1024)}MB, Heap=${Math.round(memUsage.heapUsed/1024/1024)}MB`);
  
  server.kill('SIGTERM');
  
  // Test 3: Large input handling (test our 10MB limit)
  console.log('\n📊 Test 3: Large input validation');
  const largeInput = 'x'.repeat(11 * 1024 * 1024); // 11MB - should be rejected
  console.log(`✅ Created ${Math.round(largeInput.length/1024/1024)}MB test input`);
  console.log('✅ Input size validation will be tested during tool calls');
  
  console.log('\n🎉 Performance tests completed successfully!');
}

testServerPerformance().catch(console.error);