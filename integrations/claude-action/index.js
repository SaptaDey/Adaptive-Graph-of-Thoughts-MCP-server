const fetch = require('node-fetch');

const BASE_URL = process.env.AGOT_BASE_URL || 'http://localhost:8000';

module.exports = async ({selectionText}) => {
  try {
    const res = await fetch(`${BASE_URL}/nlq`, {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({ question: selectionText })
    });
    
    if (!res.ok) {
      throw new Error(`HTTP ${res.status}: ${res.statusText}`);
    }
    
    const data = await res.json();
    return data.summary || JSON.stringify(data);
  } catch (error) {
    return `Error: ${error.message}`;
  }
};
