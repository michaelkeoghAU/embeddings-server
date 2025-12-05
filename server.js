import express from "express";
import pkg from "pg";
import OpenAI from "openai";

const { Pool } = pkg;

const app = express();
app.use(express.json({ limit: "2mb" }));

// Postgres
const pool = new Pool({
  connectionString: process.env.POSTGRES_URL,
  ssl: { rejectUnauthorized: false }
});

// OpenAI
const openai = new OpenAI({ apiKey: process.env.OPENAI_API_KEY });

/* ---------------- EMBED ENDPOINT ---------------- */

app.post("/embed", async (req, res) => {
  try {
    const { ticket_number, summary } = req.body;

    if (!ticket_number || !summary) {
      return res.status(400).json({ error: "ticket_number and summary required" });
    }

    // Create embedding
    const embeddingResponse = await openai.embeddings.create({
      model: "text-embedding-3-small",
      input: summary
    });

    const vector = embeddingResponse.data[0].embedding;

    // Insert OR update existing ticket
    await pool.query(
      `INSERT INTO ticket_embeddings (ticket_number, summary, embedding)
       VALUES ($1, $2, $3)
       ON CONFLICT (ticket_number)
       DO UPDATE SET summary = EXCLUDED.summary,
                     embedding = EXCLUDED.embedding`,
      [ticket_number, summary, vector]
    );

    return res.json({ ok: true });

  } catch (err) {
    console.error("SERVER ERROR:", err);
    return res.status(500).json({ error: "server error" });
  }
});

/* ---------------- START SERVER ---------------- */

app.listen(3000, () => {
  console.log("Server running on port 3000");
});
