/**
 * Parses a string containing newline-delimited JSON (NDJSON) and returns an array of parsed objects.
 * @param {string} text - The NDJSON-formatted string to parse.
 * @return {Object[]} An array of objects parsed from each non-empty line of the input.
 */
function parseNdjson(text) {
  return text.trim().split(/\r?\n/).filter(Boolean).map((line) => JSON.parse(line));
}
module.exports = { parseNdjson };
