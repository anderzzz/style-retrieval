# Style Reconstruction Method Performance Analysis

*Generated: 2026-01-08 16:23:09*

*Based on 10 samples × 2 runs = 20 judgments per LLM*

---


## Mistral Reconstructions - Judge Evaluation Summary

| Method | Mean Rank | Std Dev | % 1st | % Top-2 | % Last |
|--------|-----------|---------|-------|---------|--------|
| Fewshot              | 1.90 | 0.72 |  30.0% |  80.0% |   0.0% |
| Agent Statistical    | 1.95 | 0.94 |  40.0% |  70.0% |   5.0% |
| Author               | 2.40 | 1.10 |  30.0% |  45.0% |  15.0% |
| Generic              | 3.75 | 0.55 |   0.0% |   5.0% |  80.0% |


## OpenAI Reconstructions - Judge Evaluation Summary

| Method | Mean Rank | Std Dev | % 1st | % Top-2 | % Last |
|--------|-----------|---------|-------|---------|--------|
| Agent Statistical    | 1.50 | 0.83 |  65.0% |  90.0% |   5.0% |
| Fewshot              | 1.85 | 0.81 |  35.0% |  85.0% |   5.0% |
| Author               | 3.00 | 0.65 |   0.0% |  20.0% |  20.0% |
| Generic              | 3.65 | 0.59 |   0.0% |   5.0% |  70.0% |


## Qwen Reconstructions - Judge Evaluation Summary

| Method | Mean Rank | Std Dev | % 1st | % Top-2 | % Last |
|--------|-----------|---------|-------|---------|--------|
| Agent Statistical    | 1.75 | 1.02 |  55.0% |  80.0% |  10.0% |
| Fewshot              | 2.10 | 0.79 |  20.0% |  75.0% |   5.0% |
| Author               | 2.45 | 1.00 |  25.0% |  40.0% |  10.0% |
| Generic              | 3.70 | 0.57 |   0.0% |   5.0% |  75.0% |


## Kimi Reconstructions - Judge Evaluation Summary

| Method | Mean Rank | Std Dev | % 1st | % Top-2 | % Last |
|--------|-----------|---------|-------|---------|--------|
| Agent Statistical    | 1.75 | 0.79 |  40.0% |  90.0% |   5.0% |
| Fewshot              | 1.90 | 0.97 |  45.0% |  70.0% |   5.0% |
| Author               | 2.75 | 1.07 |  15.0% |  40.0% |  30.0% |
| Generic              | 3.60 | 0.50 |   0.0% |   0.0% |  60.0% |


---


## Mean Rank by Reconstruction LLM

*(Lower is better: 1.0 = always ranked 1st, 4.0 = always ranked last)*


| Method | Mistral | OpenAI | Qwen | Kimi | Overall |
|--------|---------|--------|------|------|---------|
| Agent Statistical    |   1.95 |   1.50 |   1.75 |   1.75 |    1.74 |
| Fewshot              |   1.90 |   1.85 |   2.10 |   1.90 |    1.94 |
| Author               |   2.40 |   3.00 |   2.45 |   2.75 |    2.65 |
| Generic              |   3.75 |   3.65 |   3.70 |   3.60 |    3.67 |


---


## Key Findings


- **Best Overall Method**: Agent Statistical (mean rank: 1.74)
- **Worst Overall Method**: Generic (mean rank: 3.67)
- **Total Judgments Analyzed**: 80 (across all 4 LLMs)

### Best Method per LLM:

- **Mistral**: Fewshot (1.90)
- **OpenAI**: Agent Statistical (1.50)
- **Qwen**: Agent Statistical (1.75)
- **Kimi**: Agent Statistical (1.75)