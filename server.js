
// server.js
require('dotenv').config();
const express = require('express');
const cors = require('cors');

const app = express();
app.use(cors());
app.use(express.json());

// Health check
app.get('/health', (_req, res) => res.status(200).send('OK'));

// Example root route (optional)
app.get('/', (_req, res) => res.send('EmbeddingPlus server is up'));

// IMPORTANT: use Azure's assigned port
const port = process.env.PORT || 8080;
app.listen(port, () => {
  console.log(`Server running on port ${port}`);
});
