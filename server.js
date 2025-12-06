// -------------------------------------------------------
// server.js  (FINAL FIXED VERSION)
// -------------------------------------------------------

require('dotenv').config();
const express = require('express');
const { Pool } = require('pg');
const OpenAI = require('openai');

const app = express();
app.use(express.json({ limit: '2mb' }));

// -------------------------------------------------------
// PostgreSQL
// -------------------------------------------------------
const pool = new Pool({
  connectionString: process.env.DATABASE_URL,
  ssl: { rejectUnauthorized: false }
});

// -------------------------------------------------------
// OpenAI Client (OpenAI or Azure OpenAI compatible)
// -------------------------------------------------------
const client = new OpenAI({
  apiKey: process.env.OPENAI_API_KEY,
  baseURL: process.env.OPENAI_BASE_URL || undefined
});

const DEFAULT_MODEL = process.env.OPENAI_MODEL || 'text-embedding-3-small';

// -------------------------------------------------------
// Utility: Convert JS array → pgvector literal
// -------------------------------------------------------
function toPgVector(arr) {
  if (!Array.isArray(arr)) throw new Error("Embedding is not an array!");
  return `[${arr.join(",")}]`;  // required pgvector format
}

// -------------------------------------------------------
// POST /embed
// -------------------------------------------------------
app.post('/embed', async (req, res) => {
  try {
    const { ticketNumber, summary } = req.body;

    if (!summary || summary.length < 3) {
      return res.status(400).json({ error: "summary must be provided" });
    }

    console.log("MODEL IN USE:", DEFAULT_MODEL);

    // ----------- Create embedding -----------
    const result = await client.embeddings.create({
      model: DEFAULT_MODEL,
      input: summary
    });

    const embedding = result.data[0].embedding;

    if (!Array.isArray(embedding)) {
      console.error("Embedding returned was NOT an array:", embedding);
      return res.status(500).json({ error: "Invalid embedding format returned" });
    }

    console.log("Embedding length:", embedding.length);

    // ----------- Format into pgvector syntax -----------
    const pgVector = toPgVector(embedding);

    // ----------- Insert into DB -----------
    const sql = `
      INSERT INTO ticket_embeddings (ticket_number, summary, embedding)
      VALUES ($1, $2, $3::vector)
      RETURNING id;
    `;

    const resultInsert = await pool.query(sql, [
      ticketNumber ?? null,
      summary,
      pgVector
    ]);

    res.json({
      ok: true,
      id: resultInsert.rows[0].id,
      dims: embedding.length,
      model: DEFAULT_MODEL
    });

  } catch (err) {
    console.error("ERROR in /embed:", err);
    res.status(500).json({
      error: err.message,
      details: err.stack
    });
  }
});

// -------------------------------------------------------
// POST /match
// -------------------------------------------------------
app.post('/match', async (req, res) => {
  try {
    const { summary } = req.body;

    if (!summary) {
      return res.status(400).json({ error: "summary text is required" });
    }

    // Embed the query text
    const result = await client.embeddings.create({
      model: DEFAULT_MODEL,
      input: summary
    });

    const queryEmbedding = result.data[0].embedding;
    const pgVector = toPgVector(queryEmbedding);

    // Perform vector similarity search (<=> operator)
    const searchSQL = `
      SELECT 
        ticket_number,
        summary,
        embedding <=> $1::vector AS distance
      FROM ticket_embeddings
      WHERE embedding IS NOT NULL
      ORDER BY distance ASC
      LIMIT 5;
    `;

    const matches = await pool.query(searchSQL, [pgVector]);

    res.json({
      ok: true,
      count: matches.rows.length,
      results: matches.rows
    });

  } catch (err) {
    console.error("ERROR in /match:", err);
    res.status(500).json({
      error: err.message,
      details: err.stack
    });
  }
});

// -------------------------------------------------------
// Health Check
// -------------------------------------------------------
app.get('/health', (req, res) => {
  res.status(200).send("OK");
});

// -------------------------------------------------------
// Root
// -------------------------------------------------------
app.get('/', (req, res) => {
  res.send("EmbeddingPlus API is running");
});

// -------------------------------------------------------
// Start Server
// -------------------------------------------------------
const port = process.env.PORT || 8080;
app.listen(port, () => console.log(`Server running on ${port}`));
