# Data classification and model use

Kepler splits data into four classes. Pasting customer data into an unapproved model is a serious violation.

## Classes

| Level | Name | Examples | External LLM |
| --- | --- | --- | --- |
| L0 | Public | Website, job posts | Allowed |
| L1 | Internal | This handbook, architecture diagrams | Approved enterprise gateway only |
| L2 | Confidential | Quotes, unpublished roadmaps | Forbidden unless redacted and VP-approved |
| L3 | Restricted | Customer PII, camera frames, warehouse maps | **Forbidden** |

## Approved model channels

- Production Q&A: company gateway to **GPT-4o** (logs retained 30 days, prompts redacted)
- Drafts and evaluation: GPT-4o-mini
- Forbidden: personal ChatGPT, personal Claude, random web wrappers

## Retention

Retrieval and answer logs are kept **90** days for hallucination audits and compliance sampling. Raw customer video does not enter the knowledge base.
