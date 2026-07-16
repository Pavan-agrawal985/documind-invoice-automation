# DocuMind: Invoice Processing Automation

AI-powered comparative system for invoice extraction and automated decision-making.
Final-year B.Tech project — compares rule-based, classical ML, and LLM-based extraction
methods on invoice documents, then automates approval/flagging decisions.

## Project Structure
- `data/` — raw and processed invoice data (SROIE dataset)
- `extraction/` — 4 extraction method modules (rule-based, classical ML, LLM prompting, fine-tuned)
- `decision/` — automated decision/anomaly-flagging logic
- `app/` — Flask dashboard (upload, review, correct)
- `evaluation/` — comparison scripts and metrics
- `notebooks/` — exploration and experiments
- `thesis/` — write-up chapters

## Setup
1. `python -m venv venv && source venv/bin/activate`
2. `pip install -r requirements.txt`
3. `python -m spacy download en_core_web_sm`
4. Set up MySQL and configure `.env` with DB credentials
5. `python app/routes.py` to run the Flask dashboard

## Status
Phase 0: Setup — in progress
