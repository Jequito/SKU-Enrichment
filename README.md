# SKU Product Enrichment

Bulk enrich SKU / product codes with descriptions, specifications and metadata.
Searches Google via SerpAPI, fetches product pages directly, and extracts
structured data using your choice of LLM.

## Features

- 📂 Upload CSV or Excel file with any column layout
- 🔍 **Google search via SerpAPI** — proper exact-match support, comprehensive index
- 🌐 **Direct page fetching** via `httpx` + `trafilatura` content extraction
- 🧬 **Two-column primary/fallback search** — pick any two columns, exact-match each
- ✅ **Content validation** — flags rows where the fetched page doesn't contain either identifier
- 📐 **JSON-LD hint** — extracts structured product data and passes it to the LLM as trusted metadata
- ⚡ Concurrent processing with configurable workers
- ⏸ **Working pause/stop** — runs one batch per Streamlit rerun
- 🤖 Extracts with OpenAI, Gemini, or Claude
- 💥 **Error diagnostic panel** — shows distinct LLM error messages grouped by frequency
- ⬇️ Download as CSV or colour-coded Excel

## Quick Start

```bash
pip install -r requirements.txt
streamlit run app.py
```

Open http://localhost:8501 in your browser.

## SerpAPI cost

SerpAPI is the only search backend. Get a key at
**https://serpapi.com/manage-api-key**.

| Plan | Cost | Searches/month | Approx SKUs/month |
|------|------|----------------|-------------------|
| Free | $0 | 100 | 50–100 |
| Developer | $75/mo | 5,000 | 2,500–5,000 |
| Production | $150/mo | 15,000 | 7,500–15,000 |

Per-row cost depends on whether the primary search finds a match:

- **Primary column finds it** → 1 query charged
- **Primary returns nothing, fallback finds it** → 2 queries charged
- **Both return nothing** → SerpAPI doesn't bill empty results, so this typically costs 1 (the cheaper of the two outcomes)

Most rows cost 1 query. Rows where the primary column is missing or unindexable cost 2.

## Search strategy

For each row, the tool runs an exact-quoted Google search on your chosen
**Primary** column. If that returns zero results, it falls through to the
**Fallback** column. There's no wide-net retry — exact match only — because
on Google a wider query usually pulls in pages about *different* products,
which then poisons the LLM extraction.

You map both columns in the sidebar. Typical mapping:

- **Primary** = manufacturer's Product Code / MPN
- **Fallback** = your internal SKU

You can flip them if your data is the other way round.

After fetching, the tool checks that the page content contains either the
Primary or the Fallback value. If neither does, the row is flagged
`REVIEW_NEEDED` because Google may have surfaced an adjacent product.

## API Keys Required

Two keys total: one for SerpAPI, one for whichever LLM you want.

| Key | Where to get | Cost |
|-----|-------------|------|
| **SerpAPI**  | serpapi.com/manage-api-key | 100 free/month, then $75/mo for 5k |
| Gemini (recommended LLM)  | aistudio.google.com   | Generous free tier |
| OpenAI                    | platform.openai.com   | Pay per use — `gpt-5.4-mini` cheapest |
| Claude                    | console.anthropic.com | Pay per use — `claude-haiku-4-5-20251001` cheapest |

## Diagnosing failures

When rows come back with `ERROR` in the review_flag, the diagnostic panel
above the results table shows the actual error messages from the LLM
provider, grouped by frequency. Common ones:

- **AuthenticationError / invalid x-api-key** — wrong or expired LLM key
- **SerpAPI key rejected (401)** — paste the key from serpapi.com/manage-api-key
- **SerpAPI quota exhausted** — top up or upgrade plan
- **model_not_found** — the model identifier no longer exists
- **Gemini safety filter blocked** — page content tripped Gemini's filter; switch to Claude or OpenAI

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
    ├── search_client.py     # SerpAPI search + httpx fetch + trafilatura
    ├── content_cleaner.py   # JSON-LD extraction
    ├── extractors.py        # OpenAI / Gemini / Claude
    ├── file_handler.py      # CSV + Excel I/O
    └── pipeline.py          # Per-product orchestration + concurrency
```
