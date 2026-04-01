// -------------------------------------------------------
// server.js — PRODUCTION (Power Automate compatible)
// Contract: { ticketNumber, text }
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
// OpenAI
// -------------------------------------------------------
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
    throw new Error("Embedding is not an array");
  }
  return `[${arr.join(",")}]`;
}

// ====================================================================
// POST /embed — embeds EXACT text sent by Power Automate
// ====================================================================
app.post('/embed', async (req, res) => {
  try {
    const { ticketNumber, text } = req.body;

    if (!ticketNumber) {
      return res.status(400).json({ error: "ticketNumber is required" });
    }

    const cleanText = (text || "").trim();
    if (!cleanText) {
      return res.status(400).json({ error: "text is required" });
    }

    const safeText = safeTruncate(cleanText, MAX_CHARS_TEXT);

    const embedResult = await openai.embeddings.create({
      model: MODEL,
      input: safeText
    });

    const embedding = embedResult.data[0].embedding;
    const pgVector = toPgVector(embedding);

    const sql = `
      INSERT INTO ticket_embeddings (
        ticket_number,
        notes,
        embedding,
        created_at
      )
      VALUES ($1, $2, $3::vector, NOW())
      ON CONFLICT (ticket_number)
      DO UPDATE SET
        notes      = EXCLUDED.notes,
        embedding  = EXCLUDED.embedding,
        created_at = NOW()
      RETURNING id;
    `;

    const result = await pool.query(sql, [
      ticketNumber,
      safeText,
      pgVector
    ]);

    res.json({
      ok: true,
      ticketNumber,
      id: result.rows[0].id,
      dims: embedding.length
    });

  } catch (err) {
    console.error("ERROR /embed:", err);
    res.status(500).json({ error: err.message });
  }
});

// ====================================================================
// POST /match — similarity search using same text contract
// ====================================================================
app.post('/match', async (req, res) => {
  try {
    const { text } = req.body;

    const cleanText = (text || "").trim();
    if (!cleanText) {
      return res.status(400).json({ error: "text is required" });
    }

    const safeText = safeTruncate(cleanText, MAX_CHARS_TEXT);

    const embedResult = await openai.embeddings.create({
      model: MODEL,
      input: safeText
    });

    const embedding = embedResult.data[0].embedding;
    const pgVector = toPgVector(embedding);

    const sql = `
      SELECT
        ticket_number,
        notes,
        embedding <=> $1::vector AS distance
      FROM ticket_embeddings
      ORDER BY distance ASC
      LIMIT 5;
    `;

    const matches = await pool.query(sql, [pgVector]);

    res.json({
      ok: true,
      count: matches.rows.length,
      results: matches.rows
    });

  } catch (err) {
    console.error("ERROR /match:", err);
    res.status(500).json({ error: err.message });
  }
});

// ====================================================================
// Health
// ====================================================================
app.get('/health', (req, res) => {
  res.status(200).send("OK");
});

// Root
app.get('/', (req, res) => {
  res.send("EmbeddingPlus API running");
});

// ====================================================================
// Start server (Azure-safe)
// ====================================================================
const port = process.env.PORT || 8080;
app.listen(port, () => {
  console.log(`EmbeddingPlus API running on port ${port}`);
});
