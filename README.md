# SKU Product Enrichment

Bulk enrich SKU / product codes with descriptions, specifications and metadata.
Searches Google via Jina AI, fetches product pages, cleans content, and extracts structured data using your choice of LLM.

## Features

- 📂 **Upload** CSV or Excel file with any column layout
- 🔍 **Searches** Google exactly for each SKU using Jina AI (`s.jina.ai`)
- 🌏 **Country targeting** — Australia default, 10 countries supported (via `&gl=` and `&location=` params — not search keywords)
- 📄 **Fetches** product pages via Jina Reader (`r.jina.ai`) — clean markdown
- 🧹 **Content cleaning** — strips mega menus, footers, cookie notices before LLM sees the page
- 📐 **JSON-LD hint** — injects structured page metadata into the LLM prompt as trusted data for higher accuracy
- ⚡ **Concurrent processing** — configurable workers (1–20) for large batches
- 🤖 **Extracts** with OpenAI, Gemini, or Claude
- 📊 **Live table** updates as batches complete
- ⬇️ **Download** enriched results as CSV or colour-coded Excel

## Quick Start

```bash
git clone <your-repo-url>
cd sku-enrichment
pip install -r requirements.txt
streamlit run app.py
```

Open http://localhost:8501 in your browser.

## API Keys Required

| Key | Where to get | Cost |
|-----|-------------|------|
| **LLM key (choose one)** | | |
| OpenAI | platform.openai.com | Pay per use — gpt-4o-mini cheapest |
| Gemini | aistudio.google.com | Free tier + pay per use — gemini-2.0-flash cheapest |
| Claude | console.anthropic.com | Pay per use — claude-haiku cheapest |
| **Jina API Key** | jina.ai/reader | Free tier — optional but strongly recommended for large batches |

## Cost Estimates at 10,000 SKUs

| Model | Estimated Cost |
|-------|---------------|
| gemini-2.0-flash | ~$1–3 |
| claude-haiku-4-5 | ~$5–8 |
| gpt-4o-mini | ~$8–12 |
| gpt-4o | ~$100+ |

Jina fetching is free with your API key regardless of volume.

## Large Batch Guidance

- Run **locally** for batches over 5,000 SKUs — Streamlit Cloud sessions may time out
- Use **5–10 concurrent workers** for best speed/stability balance
- Estimated runtime at 5 workers: ~2–3 hours for 10k SKUs
- Keep browser tab active — progress lives in the session

## Jina AI Settings

| Setting | Description |
|---------|-------------|
| API Key | Higher rate limits — free at jina.ai/reader |
| Search Country | Country-targeted Google results via `&gl=` and `&location=` params |
| URLs per SKU | 1–5 pages to cross-reference (2 recommended) |
| Max chars per page | Applied after content cleaning (4,000 default) |
| Request timeout | Per-request timeout in seconds |
| Page return format | `markdown` recommended for LLM extraction |
| Bypass cache | Always fetch fresh content |
| Retry on few results | Retry with `specifications` appended if < 2 results |
| Delay between fetches | Pause between URL fetches within one SKU |

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
| review_flag | VERIFIED / REVIEW_NEEDED / NOT_FOUND / ERROR |

## Project Structure

```
sku-enrichment/
├── app.py                   # Streamlit UI
├── requirements.txt
├── README.md
└── src/
    ├── jina_client.py       # Jina search + fetch
    ├── content_cleaner.py   # Nav stripping + JSON-LD extraction
    ├── extractors.py        # OpenAI / Gemini / Claude
    ├── file_handler.py      # CSV + Excel I/O
    └── pipeline.py          # Per-SKU orchestration + concurrency
```
