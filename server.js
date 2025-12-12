// -------------------------------------------------------
// server.js — FINAL VERSION WITH NOTES SUPPORT
// -------------------------------------------------------

require('dotenv').config();
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
// OpenAI Client
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
  return `[${arr.join(",")}]`;
}

// ====================================================================
// POST /embed — UPSERT summary + notes (combined embedding)
// ====================================================================
app.post('/embed', async (req, res) => {
  try {
    const { ticketNumber, summary, notes } = req.body;

    const cleanSummary = (summary || "").trim();
    const cleanNotes = (notes || "").trim();

    // Require at least 10 chars of meaningful summary
    if (!cleanSummary || cleanSummary.length < 10) {
      return res.status(400).json({
        error: "summary must be at least 10 characters",
        providedLength: cleanSummary.length
      });
    }

    // Combined embedding text
    const combinedText = cleanSummary + "\n\nNotes:\n" + cleanNotes;

    console.log("Embedding ticket:", ticketNumber);
    console.log("Summary length:", cleanSummary.length);
    console.log("Notes length:", cleanNotes.length);

    // Generate embedding for combined text
    const result = await client.embeddings.create({
      model: DEFAULT_MODEL,
      input: combinedText
    });

    const embedding = result.data[0].embedding;
    const pgVector = toPgVector(embedding);

    // Store summary + notes separately, store embed for combined field
    const sql = `
      INSERT INTO ticket_embeddings (ticket_number, summary, notes, embedding, created_at)
      VALUES ($1, $2, $3, $4::vector, NOW())
      ON CONFLICT (ticket_number)
      DO UPDATE SET
          summary   = EXCLUDED.summary,
          notes     = EXCLUDED.notes,
          embedding = EXCLUDED.embedding,
          created_at = NOW()
      RETURNING id;
    `;

    const resultInsert = await pool.query(sql, [
      ticketNumber,
      cleanSummary,
      cleanNotes,
      pgVector
    ]);

    res.json({
      ok: true,
      id: resultInsert.rows[0].id,
      dims: embedding.length
    });

  } catch (err) {
    console.error("ERROR in /embed:", err);
    res.status(500).json({ error: err.message, details: err.stack });
  }
});

// ====================================================================
// POST /match — nearest neighbours (uses summary+notes embedding)
// ====================================================================
app.post('/match', async (req, res) => {
  try {
    const { summary, notes } = req.body;

    const cleanSummary = (summary || "").trim();
    const cleanNotes = (notes || "").trim();

    if (!cleanSummary || cleanSummary.length < 10) {
      return res.status(400).json({
        error: "summary must be at least 10 characters",
        providedLength: cleanSummary.length
      });
    }

    const combinedText = cleanSummary + "\n\nNotes:\n" + cleanNotes;

    const result = await client.embeddings.create({
      model: DEFAULT_MODEL,
      input: combinedText
    });

    const queryEmbedding = result.data[0].embedding;
    const pgVector = toPgVector(queryEmbedding);

    const searchSQL = `
      SELECT 
        ticket_number,
        summary,
        notes,
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
    res.status(500).json({ error: err.message, details: err.stack });
  }
});

// ====================================================================
// Health Check
// ====================================================================
app.get('/health', (req, res) => {
  res.status(200).send("OK");
});

// ====================================================================
// Root
// ====================================================================
app.get('/', (req, res) => {
  res.send("EmbeddingPlus API is running");
});

// ====================================================================
// Start Server
// ====================================================================
const port = process.env.PORT || 8080;
app.listen(port, () => console.log(`Server running on ${port}`));
