// server.js
require('dotenv').config();
const express = require('express');
const { Pool } = require('pg');
const OpenAI = require('openai');

const app = express();
app.use(express.json({ limit: '2mb' }));

// ---------------------------------------------------------
// PostgreSQL Connection (Azure Flexible Server + pgvector)
// ---------------------------------------------------------
const pool = new Pool({
  connectionString: process.env.DATABASE_URL,
  ssl: { rejectUnauthorized: false }
});

// ---------------------------------------------------------
// OpenAI Client (supports OpenAI & Azure OpenAI)
// ---------------------------------------------------------
const client = new OpenAI({
  apiKey: process.env.OPENAI_API_KEY,
  baseURL: process.env.OPENAI_BASE_URL || undefined
});

// ---------------------------------------------------------
// POST /embed → Generate embedding + Insert into PostgreSQL
// ---------------------------------------------------------
app.post('/embed', async (req, res) => {
  try {
    const { ticketNumber, summary, model } = req.body;

    if (!summary || typeof summary !== 'string') {
      return res.status(400).json({ error: 'summary (string) is required' });
    }

    const chosenModel =
      model ||
      process.env.OPENAI_MODEL ||
      'text-embedding-3-small';

    // Generate embedding
    const result = await client.embeddings.create({
      model: chosenModel,
      input: summary
    });

    const embedding = result?.data?.[0]?.embedding;

    if (!Array.isArray(embedding)) {
      return res.status(500).json({
        error: 'Embedding array missing from provider response'
      });
    }

    // Insert into PostgreSQL using pgvector `::vector` cast
    const sql = `
      INSERT INTO ticket_embeddings (ticket_number, summary, embedding)
      VALUES ($1, $2, $3::vector)
      RETURNING id;
    `;

    const db = await pool.query(sql, [
      ticketNumber ?? null,
      summary,
      embedding
    ]);

    res.status(200).json({
      ok: true,
      id: db.rows[0].id,
      ticketNumber: ticketNumber ?? null,
      dims: embedding.length,
      model: chosenModel
    });

  } catch (err) {
    console.error('❌ Embed error:', err);
    res.status(500).json({
      error: err.message,
      details: err.response?.data
    });
  }
});

// ---------------------------------------------------------
// POST /match → Find similar tickets (vector search)
// ---------------------------------------------------------
app.post('/match', async (req, res) => {
  try {
    const { summary, model } = req.body;

    if (!summary || summary.length < 5) {
      return res.status(400).json({
        error: 'summary must be at least 5 characters'
      });
    }

    const chosenModel =
      model ||
      process.env.OPENAI_MODEL ||
      'text-embedding-3-small';

    // Generate embedding for query text
    const result = await client.embeddings.create({
      model: chosenModel,
      input: summary
    });

    const queryEmbedding = result.data[0].embedding;

    // Vector similarity search (<=> distance operator)
    const sql = `
      SELECT
        ticket_number,
        summary,
        embedding <=> $1 AS distance
      FROM ticket_embeddings
      ORDER BY embedding <=> $1
      LIMIT 5;
    `;

    const db = await pool.query(sql, [queryEmbedding]);

    res.status(200).json({
      ok: true,
      count: db.rows.length,
      matches: db.rows
    });

  } catch (err) {
    console.error('❌ Match error:', err);
    res.status(500).json({
      error: err.message,
      details: err.response?.data
    });
  }
});

// ---------------------------------------------------------
// Health Check + Default Route
// ---------------------------------------------------------
app.get('/health', (_req, res) => res.status(200).send('OK'));
app.get('/', (_req, res) => res.send('EmbeddingPlus API is running'));

// ---------------------------------------------------------
// Start Server
// ---------------------------------------------------------
const port = process.env.PORT || 8080;
app.listen(port, () => {
  console.log(`🚀 Server running on port ${port}`);
});
