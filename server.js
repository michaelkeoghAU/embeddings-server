// server.js
require('dotenv').config();
const express = require('express');
const { Pool } = require('pg');
const OpenAI = require('openai');

const app = express();
app.use(express.json({ limit: '2mb' }));

// ---------------------------------------------------------
// PostgreSQL Connection (Using Azure App Settings: PGHOST,...)
// ---------------------------------------------------------
const pool = new Pool({
  host: process.env.PGHOST,
  user: process.env.PGUSER,
  password: process.env.PGPASSWORD,
  database: process.env.PGDATABASE,
  port: parseInt(process.env.PGPORT || "5432", 10),
  ssl: { rejectUnauthorized: false }
});

// ---------------------------------------------------------
// OpenAI Client (OpenAI or Azure OpenAI compatible)
// ---------------------------------------------------------
const client = new OpenAI({
  apiKey: process.env.OPENAI_API_KEY,
  baseURL: process.env.OPENAI_BASE_URL || undefined
});

// ---------------------------------------------------------
// POST /embed → Create embedding + store in PostgreSQL
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
      return res.status(500).json({ error: 'No embedding returned from provider' });
    }

    // Insert into Postgres
    const sql = `
      INSERT INTO ticket_embeddings (ticket_number, summary, embedding)
      VALUES ($1, $2, $3)
      RETURNING id;
    `;

    await pool.query(sql, [
      ticketNumber ?? null,
      summary,
      embedding
    ]);

    res.status(200).json({
      ok: true,
      ticketNumber: ticketNumber ?? null,
      dims: embedding.length,
      model: chosenModel,
      embedding
    });

  } catch (err) {
    console.error('Embed error:', err);
    res.status(500).json({
      error: err.message,
      details: err.response?.data
    });
  }
});

// ---------------------------------------------------------
// POST /match → Similarity search using pgvector
// ---------------------------------------------------------
app.post('/match', async (req, res) => {
  try {
    const { summary, model } = req.body;

    if (!summary || summary.length < 5) {
      return res.status(400).json({ error: 'summary must be at least 5 characters' });
    }

    const chosenModel =
      model ||
      process.env.OPENAI_MODEL ||
      'text-embedding-3-small';

    // Create query embedding
    const result = await client.embeddings.create({
      model: chosenModel,
      input: summary
    });

    const queryEmbedding = result.data[0].embedding;

    // Search for similar embeddings
    const searchSQL = `
      SELECT
        ticket_number,
        summary,
        embedding <=> $1 AS distance
      FROM ticket_embeddings
      ORDER BY embedding <=> $1
      LIMIT 5;
    `;

    const matches = await pool.query(searchSQL, [queryEmbedding]);

    res.json({
      ok: true,
      count: matches.rows.length,
      matches: matches.rows
    });

  } catch (err) {
    console.error('Match error:', err);
    res.status(500).json({
      error: err.message,
      details: err.response?.data
    });
  }
});

// ---------------------------------------------------------
// Health + Default Route
// ---------------------------------------------------------
app.get('/health', (_req, res) => res.status(200).send('OK'));
app.get('/', (_req, res) => res.send('EmbeddingPlus API is running'));

// ---------------------------------------------------------
// Start Server
// ---------------------------------------------------------
const port = process.env.PORT || 8080;
app.listen(port, () => {
  console.log(`Server running on port ${port}`);
});
