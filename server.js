// server.js
require('dotenv').config();
const express = require('express');
const { Pool } = require('pg');
const OpenAI = require('openai');

const app = express();
app.use(express.json({ limit: '2mb' }));

// PostgreSQL
const pool = new Pool({
  connectionString: process.env.DATABASE_URL,
  ssl: { rejectUnauthorized: false }
});

// OpenAI
const client = new OpenAI({
  apiKey: process.env.OPENAI_API_KEY,
  baseURL: process.env.OPENAI_BASE_URL || undefined
});

// -------------------------
// /embed
// -------------------------
app.post('/embed', async (req, res) => {
  try {
    const { ticketNumber, summary } = req.body;

   console.log("MODEL IN USE before:", process.env.OPENAI_MODEL);
 
    const embed = await client.embeddings.create({
      model: process.env.OPENAI_MODEL || 'text-embedding-3-small',
      input: summary
    });

   console.log("MODEL IN USE after:", process.env.OPENAI_MODEL);

    const embedding = embed.data[0].embedding;

    const sql = `
      INSERT INTO ticket_embeddings (ticket_number, summary, embedding)
      VALUES ($1, $2, $3::vector)
      RETURNING id;
    `;

    const result = await pool.query(sql, [
      ticketNumber,
      summary,
      embedding
    ]);

    res.json({ ok: true, id: result.rows[0].id });

  } catch (err) {
    res.status(500).json({ error: err.message });
  }
});

// -------------------------
// /match
// -------------------------
app.post('/match', async (req, res) => {
  try {
    const { summary } = req.body;

    const embed = await client.embeddings.create({
      model: process.env.OPENAI_MODEL || 'text-embedding-3-small',
      input: summary
    });

    const queryEmbedding = embed.data[0].embedding;

    const sql = `
      SELECT ticket_number, summary, embedding <=> $1 AS distance
      FROM ticket_embeddings
      ORDER BY embedding <=> $1
      LIMIT 5;
    `;

    const result = await pool.query(sql, [queryEmbedding]);

    res.json({ ok: true, matches: result.rows });

  } catch (err) {
    res.status(500).json({ error: err.message });
  }
});

// -------------------------
app.get('/health', (req, res) => res.send('OK'));
app.get('/', (req, res) => res.send('EmbeddingPlus API is running'));

const port = process.env.PORT || 8080;
app.listen(port, () => console.log(`Server running on ${port}`));
