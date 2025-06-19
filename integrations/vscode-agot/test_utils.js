const assert = require('assert');
const { parseNdjson } = require('./utils');
const sample = '{"a":1}\n{"b":2}\n';
const parsed = parseNdjson(sample);
assert.deepStrictEqual(parsed, [{a:1},{b:2}]);
console.log('OK');
