// server.js
require("dotenv").config();
const express = require("express");
const cors = require("cors");
const { Pool } = require("pg");
const OpenAI = require("openai");

const app = express();
app.use(express.json({ limit: "2mb" }));
app.use(cors());

// PostgreSQL
const pool = new Pool({
    connectionString: process.env.DATABASE_URL,
    ssl: { rejectUnauthorized: false }
});

// OpenAI
const openai = new OpenAI({
    apiKey: process.env.OPENAI_API_KEY
});

// Health check
app.get("/", (req, res) => {
    res.send("EmbeddingPlus API running");
});

// ----------------------
// EMBED (no description)
// ----------------------
app.post("/embed", async (req, res) => {
    try {
        const { ticket_number, summary } = req.body;

        if (!ticket_number || !summary) {
            return res.status(400).json({ error: "ticket_number and summary are required" });
        }

        // Text to embed = summary only
        const textToEmbed = summary.trim();

        // Create embedding
        const embedResponse = await openai.embeddings.create({
            model: "text-embedding-3-large",
            input: textToEmbed
        });

        const vector = embedResponse.data[0].embedding;
        const vectorJson = JSON.stringify(vector);

        // Store in DB
        const sql = `
            INSERT INTO ticket_embeddings (ticket_number, summary, embedding)
            VALUES ($1, $2, $3)
            ON CONFLICT (ticket_number)
            DO UPDATE SET summary = EXCLUDED.summary,
                          embedding = EXCLUDED.embedding
            RETURNING id;
        `;

        const result = await pool.query(sql, [
            ticket_number.toString(),
            summary,
            vectorJson
        ]);

        res.json({
            success: true,
            ticket_number,
            record_id: result.rows[0].id
        });

    } catch (err) {
        console.error("ERROR in /embed:", err);
        res.status(500).json({ error: "Internal server error", details: err.message });
    }
});

// Start server
const port = process.env.PORT || 3000;
app.listen(port, () => {
    console.log(`EmbeddingPlus API running on port ${port}`);
});
