
// server.js
require('dotenv').config();
const express = require('express');
const OpenAI = require('openai');

const app = express();
app.use(express.json({ limit: '1mb' })); // parse JSON safely

// OpenAI client setup
// - Standard OpenAI: only OPENAI_API_KEY is required.
// - Azure OpenAI: also set OPENAI_BASE_URL and OPENAI_API_VERSION.
//   Example OPENAI_BASE_URL:
//   https://<resource>.openai.azure.com/openai/deployments/<deployment>?api-version=2024-02-15-preview
const client = new OpenAI({
  apiKey: process.env.OPENAI_API_KEY,
  baseURL: process.env.OPENAI_BASE_URL || undefined
});

/**
 * POST /embed
 * Request body:
 * {
 *   "ticketNumber": "TEST-123",
 *   "text": "The server is overheating",
 *   "model": "text-embedding-3-small" // optional; for Azure, use your deployment name
 * }
 *
 * Response:
 * {
 *   "ok": true,
 *   "ticketNumber": "TEST-123",
 *   "model": "text-embedding-3-small",
 *   "dims": 1536,
 *   "embedding": [ ... ]
 * }
 */
app.post('/embed', async (req, res) => {
  try {
    const { ticketNumber, text, model } = req.body;

    if (!text || typeof text !== 'string') {
      return res.status(400).json({ error: 'text (string) is required' });
    }

    const chosenModel =
      model ||
      process.env.OPENAI_MODEL ||          // optional override via env
      'text-embedding-3-small';            // default for OpenAI

    const result = await client.embeddings.create({
      model: chosenModel,
      input: text
    });

    const embedding = result?.data?.[0]?.embedding;
    if (!Array.isArray(embedding)) {
      return res.status(500).json({ error: 'No embedding returned from provider' });
    }

    return res.status(200).json({
      ok: true,
      ticketNumber: ticketNumber ?? null,
      model: chosenModel,
      dims: embedding.length,
      embedding
    });
  } catch (err) {
    console.error('Embed error:', err);
    const status = err.status ?? 500;
    return res.status(status).json({
      error: err.message || 'Embedding failed',
      details: err.response?.data || undefined
    });
  }
});

// Azure App Service injects PORT; fallback to 8080 locally
const port = process.env.PORT || 8080;
app.listen(port, () => {
  console.log(`Server running on port ${port}`);
});
