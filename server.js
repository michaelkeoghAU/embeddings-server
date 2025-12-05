// server.js
require('dotenv').config();
const express = require('express');
const { Pool } = require('pg');
const OpenAI = require('openai');

const app = express();
app.use(express.json({ limit: '2mb' }));

// ---------------------------------------------
// PostgreSQL Connection (Azure Flexible Server)
// ---------------------------------------------
const pool = new Pool({
  connectionString: process.env.DATABASE_URL,
  ssl: { rejectUnauthorized: false }
});

// ---------------------------------------------
// OpenAI Client Setup (OpenAI or Azure OpenAI)
// ---------------------------------------------
const client = new OpenAI({
  apiKey: process.env.OPENAI_API_KEY,
  baseURL: process.env.OPENAI_BASE_URL || undefined
});

// ---------------------------------------------
// POST /embed  → create embedding + store in DB
// ---------------------------------------------
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

    // Create embedding
    const result = await client.embeddings.create({
      model: chosenModel,
      input: summary
    });

    const embedding = result?.data?.[0]?.embedding;
    if (!Array.isArray(embedding)) {
      return res.status(500).json({ error: 'No embedding returned from provider' });
    }

    // Insert into PostgreSQL
    const insertSQL = `
      INSERT INTO ticket_embeddings (ticket_number, summary, embedding)
      VALUES ($1, $2, $3)
      RETURNING id;
    `;

    await pool.query(insertSQL, [
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

// ---------------------------------------------
// POST /match  → embedding similarity search
// ---------------------------------------------
app.post('/match', async (req, res) => {
  try {
    const { summary, model } = req.body;

    if (!summary || summary.length < 5) {
      return res.status(400).json({ error: 'summary must be at least 5 characters' });
    }

    // Step 1: Embed the incoming text
    const chosenModel =
      model ||
      process.env.OPENAI_MODEL ||
      'text-embedding-3-small';

    const result = await client.embeddings.create({
      model: chosenModel,
      input: summary
    });

    const queryEmbedding = result.data[0].embedding;

    // Step 2: Vector similarity search using pgvector <=> operator
    const searchSQL = `
      SELECT
        ticket_number,
        summary,
        embedding <=> $1 AS distance
      FROM ticket_embeddings
      ORDER BY embedding <=> $1
      LIMIT 5;
    `;

    const dbResult = await pool.query(searchSQL, [queryEmbedding]);

    res.json({
      ok: true,
      count: dbResult.rows.length,
      matches: dbResult.rows
    });

  } catch (err) {
    console.error('Match error:', err);
    res.status(500).json({
      error: err.message,
      details: err.response?.data
    });
  }
});

// ---------------------------------------------
// Health Check + Root Route
// ---------------------------------------------
app.get('/health', (_req, res) => res.status(200).send('OK'));
app.get('/', (_req, res) => res.send('EmbeddingPlus API is running'));

// ---------------------------------------------
// Server Startup
// ---------------------------------------------
const port = process.env.PORT || 8080;
app.listen(port, () => {
  console.log(`Server running on port ${port}`);
});
