// -------------------------------------------------------
// server.js  (FINAL VERSION WITH BULK INGEST SUPPORT)
// -------------------------------------------------------

require('dotenv').config();
const express = require('express');
const { Pool } = require('pg');
const OpenAI = require('openai');
const fetch = require('node-fetch');   // REQUIRED for CW + local API calls

const app = express();
app.use(express.json({ limit: '5mb' }));

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

    // ----------- UPSERT -----------
    const sql = `
      INSERT INTO ticket_embeddings (ticket_number, summary, embedding, created_at)
      VALUES ($1, $2, $3::vector, NOW())
      ON CONFLICT (ticket_number)
      DO UPDATE SET
          summary = EXCLUDED.summary,
          embedding = EXCLUDED.embedding,
          created_at = NOW()
      RETURNING id;
    `;

    const resultInsert = await pool.query(sql, [
      ticketNumber,
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
// NEW: POST /ingest-all-closed
// One-off ingestion of ALL historical closed tickets
// -------------------------------------------------------
app.post('/ingest-all-closed', async (req, res) => {
  try {
    let page = 1;
    let inserted = 0;
    let skipped = 0;

    const boards = ["SMB1", "SMB2", "SMB4", "Escalations", "Pia"];

    console.log("🚀 Starting historical closed-ticket ingestion...");

    while (true) {
      const url =
        "https://api-aus.myconnectwise.net/v4_6_release/apis/3.0/service/tickets?" +
        `pageSize=1000&page=${page}&conditions=closedFlag=true AND (` +
        boards.map(b => `board/name="${b}"`).join(" OR ") +
        ")";

      console.log(`➡ Fetching page ${page}`);

      const response = await fetch(url, {
        headers: {
          clientId: process.env.CW_CLIENT_ID,
          Authorization:
            "Basic " +
            Buffer.from(
              `${process.env.CW_PUBLIC_KEY}:${process.env.CW_PRIVATE_KEY}`
            ).toString("base64")
        }
      });

      const tickets = await response.json();

      if (!Array.isArray(tickets) || tickets.length === 0) {
        console.log("📭 No more tickets returned — complete.");
        break;
      }

      for (const t of tickets) {

        // ---------- Duplicate check BEFORE OpenAI ----------
        const exists = await pool.query(
          `SELECT 1 FROM ticket_embeddings WHERE ticket_number = $1 LIMIT 1`,
          [t.id]
        );

        if (exists.rowCount > 0) {
          skipped++;
          continue;
        }

        // ---------- Send to /embed (internal API call) ----------
        const embedRes = await fetch("http://localhost:8080/embed", {
          method: "POST",
          headers: { "Content-Type": "application/json" },
          body: JSON.stringify({
            ticketNumber: t.id,
            summary: t.summary || ""
          })
        });

        if (embedRes.ok) {
          inserted++;
        }
      }

      page++;
    }

    res.json({
      ok: true,
      inserted,
      skipped
    });

  } catch (err) {
    console.error("ERROR in /ingest-all-closed:", err);
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
