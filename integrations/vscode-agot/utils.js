function parseNdjson(text) {
  return text.trim().split(/\r?\n/).filter(Boolean).map((line) => JSON.parse(line));
}
module.exports = { parseNdjson };
