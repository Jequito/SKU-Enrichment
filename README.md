# SKU Product Enrichment

Bulk enrich SKU / product codes with descriptions, specifications and metadata.
Searches DuckDuckGo (no API key), fetches product pages directly, and extracts
structured data using your choice of LLM.

## Features

- 📂 Upload CSV or Excel file with any column layout
- 🔍 **No search API** — uses DuckDuckGo via the `ddgs` library
- 🌐 **No fetch API** — direct `httpx` + `trafilatura` content extraction
- 🧬 **Cascade search** — tries Manufacturer Code, SKU, Brand, and Product Name
  combinations to handle the case where SKU and manufacturer code differ
- ✅ **Content validation** — flags rows where the fetched page doesn't contain
  any of your identifier codes (suggests the wrong product was matched)
- 📐 **JSON-LD hint** — extracts structured product data from page HTML and
  passes it to the LLM as trusted metadata
- ⚡ Concurrent processing with configurable workers (1–20)
- ⏸ **Working pause/stop** — runs one batch per Streamlit rerun so the buttons
  always respond
- 🤖 Extracts with OpenAI, Gemini, or Claude
- 📊 Live table updates as batches complete
- ⬇️ Download as CSV or colour-coded Excel

## Quick Start

```bash
pip install -r requirements.txt
streamlit run app.py
```

Open http://localhost:8501 in your browser.

## API Keys Required

Only an LLM key — no search or fetch keys.

| Key | Where to get | Cost |
|-----|-------------|------|
| Gemini (recommended) | aistudio.google.com | Generous free tier — gemini-2.0-flash |
| OpenAI               | platform.openai.com | Pay per use — gpt-4o-mini cheapest |
| Claude               | console.anthropic.com | Pay per use — claude-haiku cheapest |

## Identifier Cascade

For each row, the tool tries up to five queries in order, stopping at the first
that returns results matching your identifier codes:

1. `"{Manufacturer Code}"` quoted
2. `"{SKU}"` quoted
3. `"{Manufacturer Code}" {Brand}`
4. `"{SKU}" {Brand}`
5. `{Product Name} {Brand}` — last-resort wide search

A "match" means the SKU or manufacturer code appears in the result's URL,
title, or snippet. After fetching, content is also checked for those codes —
if neither appears in the page text, the row is flagged `REVIEW_NEEDED`.

## Rate Limit Notes

DuckDuckGo throttles. The hard caps are not officially documented, but in
practice:

- 3–5 concurrent workers is reliable
- 6–10 workers usually works with a 1–2 second delay between batches
- Above 10 workers you'll see frequent rate-limit errors

If you hit a limit the run auto-pauses and you can resume after waiting a few
minutes — already-completed work is preserved.

## Output Fields

| Field | Description |
|-------|-------------|
| product_name | Full cleaned product title |
| brand | Brand or manufacturer |
| short_description | 1–2 sentence summary |
| long_description | Full 3–6 sentence description |
| specifications | Pipe-separated key:value specs |
| category | Product category |
| model_number | Manufacturer MPN |
| barcode | EAN/UPC/GTIN |
| country_of_origin | Country of manufacture |
| source_url | Primary source URL |
| confidence_score | High / Medium / Low |
| review_flag | VERIFIED / REVIEW_NEEDED / NOT_FOUND / BLOCKED / RATE_LIMITED / ERROR |

## Project Structure

```
sku-enrichment/
├── app.py                   # Streamlit UI + one-batch-per-rerun pipeline driver
├── requirements.txt
├── README.md
└── src/
    ├── search_client.py     # DuckDuckGo search + httpx fetch + trafilatura
    ├── content_cleaner.py   # JSON-LD extraction
    ├── extractors.py        # OpenAI / Gemini / Claude
    ├── file_handler.py      # CSV + Excel I/O
    └── pipeline.py          # Per-product orchestration + concurrency
```
