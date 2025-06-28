#!/usr/bin/env node

// Test input validation directly
import { z } from 'zod';

// Import schemas from our server (simulated)
const ScientificReasoningQuerySchema = z.object({
  query: z.string().describe('The scientific question or research query to analyze'),
  parameters: z.object({
    include_reasoning_trace: z.boolean().default(true).describe('Include detailed reasoning steps'),
    include_graph_state: z.boolean().default(false).describe('Include graph state information'),
    max_depth: z.number().min(1).max(10).optional().describe('Override maximum reasoning depth'),
    confidence_threshold: z.number().min(0).max(1).optional().describe('Override confidence threshold'),
  }).optional(),
});

const AnalyzeResearchHypothesisSchema = z.object({
  hypothesis: z.string().describe('The research hypothesis to analyze'),
  context: z.string().optional().describe('Additional context or background information'),
  evidence_sources: z.array(z.string()).optional().describe('Specific evidence sources to consider'),
});

// Test cases
const testCases = [
  {
    name: 'Valid Scientific Query',
    schema: ScientificReasoningQuerySchema,
    input: {
      query: "What are the key molecular mechanisms underlying protein misfolding in Alzheimer's disease?",
      parameters: {
        include_reasoning_trace: true,
        max_depth: 5,
        confidence_threshold: 0.8
      }
    },
    expectSuccess: true
  },
  {
    name: 'Invalid Max Depth (too high)',
    schema: ScientificReasoningQuerySchema,
    input: {
      query: "Test query",
      parameters: { max_depth: 15 }
    },
    expectSuccess: false
  },
  {
    name: 'Invalid Confidence Threshold (negative)',
    schema: ScientificReasoningQuerySchema,
    input: {
      query: "Test query",
      parameters: { confidence_threshold: -0.5 }
    },
    expectSuccess: false
  },
  {
    name: 'Missing Required Query',
    schema: ScientificReasoningQuerySchema,
    input: {
      parameters: { max_depth: 3 }
    },
    expectSuccess: false
  },
  {
    name: 'Valid Hypothesis Analysis',
    schema: AnalyzeResearchHypothesisSchema,
    input: {
      hypothesis: "Chronic inflammation contributes to accelerated cognitive decline in Alzheimer's disease patients",
      context: "Multiple studies suggest neuroinflammation plays a role in AD progression",
      evidence_sources: ["PubMed", "Google Scholar"]
    },
    expectSuccess: true
  }
];

console.log('🧪 Testing Input Validation...\n');

let passed = 0;
let failed = 0;

testCases.forEach((testCase, index) => {
  try {
    const result = testCase.schema.parse(testCase.input);
    if (testCase.expectSuccess) {
      console.log(`✅ Test ${index + 1} PASSED: ${testCase.name}`);
      passed++;
    } else {
      console.log(`🔴 Test ${index + 1} FAILED: ${testCase.name} - Expected validation to fail but it passed`);
      failed++;
    }
  } catch (error) {
    if (!testCase.expectSuccess) {
      console.log(`✅ Test ${index + 1} PASSED: ${testCase.name} - Validation correctly failed`);
      passed++;
    } else {
      console.log(`🔴 Test ${index + 1} FAILED: ${testCase.name} - ${error.message}`);
      failed++;
    }
  }
});

console.log(`\n📊 VALIDATION TEST RESULTS:`);
console.log(`✅ Passed: ${passed}`);
console.log(`🔴 Failed: ${failed}`);
console.log(`🎯 Success Rate: ${Math.round((passed / (passed + failed)) * 100)}%`);

if (failed === 0) {
  console.log('\n🎉 ALL VALIDATION TESTS PASSED!');
} else {
  console.log(`\n⚠️  ${failed} validation tests failed`);
}