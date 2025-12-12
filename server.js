// -------------------------------------------------------
// server.js — FINAL VERSION (supports NOTES + embeddings)
// -------------------------------------------------------

require('dotenv').config();
const express = require('express');
const { Pool } = require('pg');
const OpenAI = require('openai');

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
// Utility: Convert JS embedding array → pgvector literal
// -------------------------------------------------------
function toPgVector(arr) {
  if (!Array.isArray(arr)) throw new Error("Embedding is not an array");
  return `[${arr.join(",")}]`;
}

// -------------------------------------------------------
// POST /embed — UPSERT with summary + notes
// -------------------------------------------------------
app.post('/embed', async (req, res) => {
  try {
    const { ticketNumber, summary, notes } = req.body;

    const cleanSummary = (summary || "").trim();
    const cleanNotes = (notes || "").trim();

    if (!cleanSummary || cleanSummary.length < 10) {
      return res.status(400).json({
        error: "summary must be at least 10 characters",
        providedLength: cleanSummary.length
      });
    }

    // Combine text for embedding
    const textForEmbedding =
      cleanNotes.length > 0
        ? `${cleanSummary}\n\nNOTES:\n${cleanNotes}`
        : cleanSummary;

    console.log(`Embedding text length = ${textForEmbedding.length}`);

    // Generate embedding
    const result = await client.embeddings.create({
      model: DEFAULT_MODEL,
      input: textForEmbedding
    });

    const embedding = result.data[0].embedding;
    const pgVector = toPgVector(embedding);

    // Insert or update in DB
    const sql = `
      INSERT INTO ticket_embeddings (ticket_number, summary, notes, embedding, created_at)
      VALUES ($1, $2, $3, $4::vector, NOW())
      ON CONFLICT (ticket_number)
      DO UPDATE SET
          summary = EXCLUDED.summary,
          notes = EXCLUDED.notes,
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
      dims: embedding.length,
      notesIncluded: cleanNotes.length > 0
    });

  } catch (err) {
    console.error("ERROR in /embed:", err);
    res.status(500).json({ error: err.message, details: err.stack });
  }
});

// -------------------------------------------------------
// POST /match — top 5 nearest embeddings
// -------------------------------------------------------
app.post('/match', async (req, res) => {
  try {
    const { summary } = req.body;

    const cleanSummary = (summary || "").trim();
    if (!cleanSummary || cleanSummary.length < 10) {
      return res.status(400).json({
        error: "summary must be at least 10 characters",
        providedLength: cleanSummary.length
      });
    }

    const result = await client.embeddings.create({
      model: DEFAULT_MODEL,
      input: cleanSummary
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

// -------------------------------------------------------
// POST /ingest-all-closed — Historical ingestion from CW
// -------------------------------------------------------
app.post('/ingest-all-closed', async (req, res) => {
  try {
    const limit = parseInt(req.query.limit || "0", 10);
    let processed = 0;
    let inserted = 0;
    let skipped = 0;
    let shortSummaries = 0;

    let page = 1;
    const boards = ["SMB1", "SMB2", "SMB4", "Escalations", "Pia"];

    console.log("🚀 Starting historical closed-ticket ingestion...");

    while (true) {
      const url =
        "https://api-aus.myconnectwise.net/v4_6_release/apis/3.0/service/tickets?" +
        `pageSize=1000&page=${page}&conditions=` +
        `closedFlag=true AND status/name!="Closed (Cancelled)" AND (` +
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

      const text = await response.text();
      let tickets;

      try {
        tickets = JSON.parse(text);
      } catch (err) {
        console.error("CW returned NON-JSON:", text);
        return res.status(500).json({ error: "CW returned invalid JSON", raw: text });
      }

      if (!Array.isArray(tickets) || tickets.length === 0) {
        console.log("📭 No more tickets — complete.");
        break;
      }

      for (const t of tickets) {
        const summary = (t.summary || "").trim();

        if (!summary || summary.length < 10) {
          shortSummaries++;
        } else {
          const exists = await pool.query(
            `SELECT 1 FROM ticket_embeddings WHERE ticket_number = $1 LIMIT 1`,
            [t.id]
          );

          if (exists.rowCount > 0) {
            skipped++;
          } else {
            const embedRes = await fetch("http://localhost:8080/embed", {
              method: "POST",
              headers: { "Content-Type": "application/json" },
              body: JSON.stringify({
                ticketNumber: t.id,
                summary
              })
            });

            if (embedRes.ok) inserted++;
          }
        }

        processed++;
        if (limit > 0 && processed >= limit) {
          return res.json({
            ok: true,
            inserted,
            skipped,
            shortSummaries,
            note: `Stopped early after processing ${limit} tickets`
          });
        }
      }

      page++;
    }

    res.json({
      ok: true,
      inserted,
      skipped,
      shortSummaries
    });

  } catch (err) {
    console.error("ERROR in /ingest-all-closed:", err);
    res.status(500).json({ error: err.message, details: err.stack });
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
