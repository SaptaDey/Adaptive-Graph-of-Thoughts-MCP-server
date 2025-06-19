const fetch = require('node-fetch');

module.exports = async ({selectionText}) => {
  const res = await fetch('http://localhost:8000/nlq', {
    method: 'POST',
    headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify({ question: selectionText })
  });
  const data = await res.json();
  return data.summary || JSON.stringify(data);
};
