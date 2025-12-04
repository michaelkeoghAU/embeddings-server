require('dotenv').config();
const express = require('express');
const cors = require('cors');
const { Client } = require('pg');
const OpenAI = require('openai');

const app = express();
app.use(express.json());
app.use(cors());

// --------------------------------------
// PostgreSQL
// --------------------------------------
const pg = new Client({
    host: process.env.PGHOST,
    user: process.env.PGUSER,
    password: process.env.PGPASSWORD,
    database: process.env.PGDATABASE,
    port: process.env.PGPORT,
    ssl: { rejectUnauthorized: false }
});

pg.connect()
    .then(() => console.log("Connected to Postgres"))
    .catch(err => console.error("Postgres connection error:", err));

// --------------------------------------
// OpenAI
// --------------------------------------
const openai = new OpenAI({
    apiKey: process.env.OPENAI_API_KEY
});

// --------------------------------------
// Helpers
// --------------------------------------
function toPgVector(arr) {
    // Convert JS array → pgvector literal
    return `[${arr.join(',')}]`;
}

// --------------------------------------
// Health Check
// --------------------------------------
app.get('/health', (req, res) => {
    res.status(200).send("OK");
});

// --------------------------------------
// EMBED: Store a ticket in pgvector DB
// --------------------------------------
app.post('/embed', async (req, res) => {
    try {
        const { ticketNumber, text } = req.body;

        if (!ticketNumber || !text) {
            return res.status(400).json({ error: "ticketNumber and text are required" });
        }

        // Create embedding
        const response = await openai.embeddings.create({
            model: process.env.OPENAI_MODEL,
            input: text
        });

        const embedding = response.data[0].embedding;
        const vectorString = toPgVector(embedding);

        // Insert into DB
        await pg.query(
            `INSERT INTO ticket_embeddings (ticket_number, text, embedding)
             VALUES ($1, $2, $3::vector)`,
            [ticketNumber, text, vectorString]
        );

        res.json({
            status: "success",
            ticket: ticketNumber
        });

    } catch (err) {
        console.error("Error:", err);
        res.status(500).json({ error: err.message });
    }
});

// --------------------------------------
// ADVISOR: Find top 3 similar tickets
// --------------------------------------
app.post('/advisor', async (req, res) => {
    try {
        const { ticketNumber, summary, description } = req.body;

        if (!ticketNumber || !summary || !description) {
            return res.status(400).json({
                error: "ticketNumber, summary and description are required"
            });
        }

        // Combine summary and description
        const combinedText =
`Summary:
${summary}

Description:
${description}`;

        // Create embedding for the *open* ticket
        const embedRes = await openai.embeddings.create({
            model: process.env.OPENAI_MODEL,
            input: combinedText
        });

        const queryEmbedding = embedRes.data[0].embedding;
        const queryVector = toPgVector(queryEmbedding);

        // Query top 3 similar historical tickets
        const result = await pg.query(
            `SELECT ticket_number, text
             FROM ticket_embeddings
             ORDER BY embedding <-> $1::vector
             LIMIT 3`,
            [queryVector]
        );

        const matches = result.rows;

        // Build internal note for ConnectWise
        let note = `AI Suggested Related Tickets

These tickets appear most similar to this issue based on historical ConnectWise data:

`;

        matches.forEach((row, i) => {
            note += `${i + 1}. Ticket ${row.ticket_number}\n`;
            note += `${row.text}\n\n`;
        });

        // Response for Power Automate
        res.json({
            ticket: ticketNumber,
            matchCount: matches.length,
            matches,
            internalNote: note
        });

    } catch (err) {
        console.error("Error:", err);
        res.status(500).json({ error: err.message });
    }
});

// --------------------------------------
// START SERVER
// --------------------------------------
const port = process.env.PORT || 8080;
app.listen(port, () => {
    console.log(`Server running on port ${port}`);
});
