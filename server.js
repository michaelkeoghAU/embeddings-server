// server.js
require('dotenv').config();
const express = require('express');
const OpenAI = require('openai');
const { Pool } = require('pg');

const app = express();
app.use(express.json({ limit: '1mb' })); // Parse JSON safely

//---------------------------------------------
// OpenAI Client Setup (Supports Azure + OpenAI)
//---------------------------------------------
const client = new OpenAI({
  apiKey: process.env.OPENAI_API_KEY,
  baseURL: process.env.OPENAI_BASE_URL || undefined
});

//---------------------------------------------
// PostgreSQL Connection (Azure PG + SSL)
//---------------------------------------------
const pool = new Pool({
  host: process.env.PGHOST,
  user: process.env.PGUSER,
  password: process.env.PGPASSWORD,
  database: process.env.PGDATABASE,
  port: process.env.PGPORT || 5432,
  ssl: { rejectUnauthorized: false }
});

//====================================================
// POST /embed → Generate embeddings (no DB insert here)
//====================================================
app.post('/embed', async (req, res) => {
  try {
    const { ticketNumber, text, model } = req.body;

    if (!text || typeof text !== 'string') {
      return res.status(400).json({ error: 'text (string) is required' });
    }

    const chosenModel =
      model ||
      process.env.OPENAI_MODEL ||
      'text-embedding-3-small'; // Default for OpenAI

    const result = await client.embeddings.create({
      model: chosenModel,
      input: text
    });

    const embedding = result?.data?.[0]?.embedding;
    if (!Array.isArray(embedding)) {
      return res.status(500).json({ error: 'No embedding returned from provider' });
    }

    return res.status(200).json({
      ok: true,
      ticketNumber: ticketNumber ?? null,
      model: chosenModel,
      dims: embedding.length,
      embedding
    });
  } catch (err) {
    console.error('Embed error:', err);
    const status = err.status ?? 500;
    return res.status(status).json({
      error: err.message || 'Embedding failed',
      details: err.response?.data || undefined
    });
  }
});

//====================================================
// POST /match → Embedding → pgvector search → AI Note
//====================================================
app.post('/match', async (req, res) => {
  try {
    const { ticketNumber, text, topN } = req.body;

    if (!ticketNumber || !text) {
      return res.status(400).json({
        error: "ticketNumber and text are required"
      });
    }

    const limit = Math.min(topN || 5, 20); // Safety limit

    //----------------------------------------------------
    // 1. Generate embedding for incoming ticket text
    //----------------------------------------------------
    const embedResponse = await client.embeddings.create({
      model: process.env.OPENAI_MODEL || "text-embedding-3-large",
      input: text
    });

    const embedding = embedResponse.data[0].embedding;
    const embeddingVector = `[${embedding.join(",")}]`;

    //----------------------------------------------------
    // 2. Vector Similarity Search using pgvector
    //----------------------------------------------------
    const sql = `
      SELECT 
        ticket_number,
        subject,
        body,
        embedding <-> $1::vector AS distance
      FROM ticket_embeddings
      WHERE ticket_number <> $2
      ORDER BY embedding <-> $1::vector
      LIMIT $3;
    `;

    const { rows } = await pool.query(sql, [
      embeddingVector,
      ticketNumber,
      limit
    ]);

    //----------------------------------------------------
    // 3. Convert pgvector distances → similarity scores
    //----------------------------------------------------
    const matches = rows.map(r => ({
      ticketNumber: r.ticket_number,
      subject: r.subject,
      text: r.body,
      distance: Number(r.distance),
      similarity: 1 / (1 + Number(r.distance))
    }));

    //----------------------------------------------------
    // 4. Build match summary for the LLM
    //----------------------------------------------------
    const summary = matches
      .map(
        (m, i) =>
          `${i + 1}. Ticket ${m.ticketNumber} (${(m.similarity * 100).toFixed(
            1
          )}% similar)\n${m.text}`
      )
      .join("\n\n");

    //----------------------------------------------------
    // 5. Default internal note if no matches found
    //----------------------------------------------------
    let internalNote = "No historical matches found.";

    //----------------------------------------------------
    // 6. Let GPT generate a ConnectWise internal note
    //----------------------------------------------------
    if (matches.length > 0) {
      const aiResponse = await client.chat.completions.create({
        model: "gpt-4o-mini",
        messages: [
          {
            role: "system",
            content:
              "You are a senior MSP technician. Write short, structured internal notes for ConnectWise tickets."
          },
          {
            role: "user",
            content: `
Current Ticket (${ticketNumber}):
"${text}"

Historical Matches:
${summary}

Write a ConnectWise internal note including:
- Likely root cause
- 2–4 troubleshooting steps
- References to matched ticket numbers
- Keep it concise (5–8 lines max)
`
          }
        ]
      });

      internalNote = aiResponse.choices[0].message.content.trim();
    }

    //----------------------------------------------------
    // 7. Return result to CW / Power Automate
    //----------------------------------------------------
    return res.json({
      ok: true,
      ticketNumber,
      matchCount: matches.length,
      matches,
      internalNote
    });

  } catch (err) {
    console.error("MATCH ERROR:", err);
    return res.status(500).json({ error: err.message });
  }
});

//====================================================
// Root / Health Check Routes
//====================================================
app.get('/', (_req, res) => res.send('EmbeddingPlus server is up'));
app.get('/health', (_req, res) => res.status(200).send('OK'));

//====================================================
// Azure App Service listens on injected PORT
//====================================================
const port = process.env.PORT || 8080;
app.listen(port, () => {
  console.log(`Server running on port ${port}`);
});
