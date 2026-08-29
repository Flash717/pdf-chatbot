// Minimal static file server, only for local development / demoing the
// widget. In production you'd typically serve public/widget.js from
// whatever already hosts your website (or a CDN) -- it's a plain static
// JS file with no build step or server-side dependency of its own.
const express = require("express");
const path = require("path");

const app = express();
const PORT = process.env.PORT || 3000;

app.use(express.static(path.join(__dirname, "public")));

app.listen(PORT, () => {
  console.log(`Demo page:   http://localhost:${PORT}/demo.html`);
  console.log(`Widget file: http://localhost:${PORT}/widget.js`);
});
