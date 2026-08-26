const express = require('express');
const { Pool } = require('pg');
const OpenAI = require('openai');

const app = express();
app.use(express.json({ limit: '10mb' }));

// -------------------------------------------------------
// PostgreSQL
// -------------------------------------------------------
const pool = new Pool({
  connectionString: process.env.DATABASE_URL,
  ssl: { rejectUnauthorized: false }
});

// -------------------------------------------------------
// OpenAI
// -------------------------------------------------------
console.log('OPENAI_BASE_URL=', process.env.OPENAI_BASE_URL);
console.log('OPENAI_MODEL=', process.env.OPENAI_MODEL);

const openai = new OpenAI({
  apiKey: process.env.OPENAI_API_KEY,
  baseURL: process.env.OPENAI_BASE_URL || undefined
});

const MODEL = process.env.OPENAI_MODEL || 'text-embedding-3-small';

// -------------------------------------------------------
// Limits
// -------------------------------------------------------
const MAX_CHARS_TEXT = 8000;

// -------------------------------------------------------
// Helpers
// -------------------------------------------------------
function safeTruncate(value, maxChars) {
  if (!value || typeof value !== 'string') return '';
  return value.length > maxChars ? value.slice(0, maxChars) : value;
}

function toPgVector(arr) {
  if (!Array.isArray(arr)) {
    throw new Error('Embedding is not an array');
  }
  return `[${arr.join(',')}]`;
}

// ====================================================================
// POST /embed
// ====================================================================
app.post('/embed', async (req, res) => {
  try {
    const { ticketNumber, text } = req.body;

    if (!ticketNumber) {
      return res.status(400).json({ error: 'ticketNumber is required' });
    }

    const embedText = (text || '').trim();
    if (!embedText) {
      return res.status(400).json({ error: 'text is required' });
    }

    // ✅ Text exists only in memory
    const safeText = safeTruncate(embedText, MAX_CHARS_TEXT);

    // ✅ Create embedding
    const embedResult = await openai.embeddings.create({
      model: MODEL,
      input: safeText
    });

    const embedding = embedResult.data[0].embedding;
    const pgVector = toPgVector(embedding);

    // ✅ Vector‑only persistence
    const sql = `
      INSERT INTO ticket_embeddings (
        ticket_number,
        embedding,
        created_at
      )
      VALUES ($1, $2::vector, NOW())
      ON CONFLICT (ticket_number)
      DO UPDATE SET
        embedding  = EXCLUDED.embedding,
        created_at = NOW()
      RETURNING id;
    `;

    const result = await pool.query(sql, [
      ticketNumber,
      pgVector
    ]);

    return res.status(200).json({
      ok: true,
      ticketNumber,
      id: result.rows[0].id,
      dims: embedding.length
    });

  } catch (err) {
    console.error('ERROR /embed:', err);
    return res.status(500).json({ error: err.message });
  }
});

// ====================================================================
// POST /match
// ====================================================================
app.post('/match', async (req, res) => {
  try {
    const { text, limit } = req.body;

    if (!text || typeof text !== 'string') {
      return res.status(400).json({ error: 'text is required' });
    }

    const topK = Math.min(Number(limit) || 5, 20);
    const safeText = safeTruncate(text.trim(), MAX_CHARS_TEXT);

    if (!safeText) {
      return res.status(400).json({ error: 'text is empty after truncation' });
    }

    // ✅ Embed query text
    const embedResult = await openai.embeddings.create({
      model: MODEL,
      input: safeText
    });

    const queryEmbedding = embedResult.data[0].embedding;
    const pgVector = toPgVector(queryEmbedding);

    const sql = `
      SELECT
        ticket_number,
        1 - (embedding <=> $1::vector) AS similarity
      FROM ticket_embeddings
      WHERE embedding IS NOT NULL
      ORDER BY embedding <=> $1::vector
      LIMIT $2;
    `;

    const result = await pool.query(sql, [pgVector, topK]);

    return res.status(200).json({
      ok: true,
      matches: result.rows
    });

  } catch (err) {
    console.error('ERROR /match:', err);
    return res.status(500).json({ error: err.message });
  }
});

// ====================================================================
// Health
// ====================================================================
app.get('/health', (req, res) => {
  res.status(200).send('OK');
});

// Root
app.get('/', (req, res) => {
  res.send('EmbeddingPlus API running');
});

// ====================================================================
// Start server (Azure-safe)
// ====================================================================
const port = process.env.PORT || 8080;
app.listen(port, () => {
  console.log(`EmbeddingPlus API running on port ${port}`);
});
