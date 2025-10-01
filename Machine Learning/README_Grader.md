# Grader Config (Post-Training)

A lightweight grading pipeline that:
1) **Ingests** an input prompt and any attached files (rubric + optional references).
2) **Generates** a candidate response with any LLM in the OpenAI API.
3) **Grades** the candidate against the rubric using any LLM in the OpenAI API.
4) **Exports** a **PDF** report with the verdict and rationale.

> Implemented in the notebook `Grader_Config_RLHF.ipynb` using `openai` and `reportlab`.

---

## Flow

<img width="500" height="900" alt="Grader_Flow" src="https://github.com/user-attachments/assets/fb376472-8aaa-4e6d-8abb-098a0e46aee1" />

---

## Quickstart

### 0) Environment
- Python 3.10+
- Set your OpenAI API key:
  ```bash
  export OPENAI_API_KEY="sk-..."
  ```

### 1) Install deps
Minimal set inferred from the notebook:
```bash
pip install openai reportlab pandas
```

### 2) Run the notebook
Open `Grader_Config_RLHF.ipynb` and execute **top to bottom**.  
At the end, the pipeline produces a **PDF** with the **verdict** and **rationale**.

---

## Inputs & Outputs

**Inputs**
- **Prompt file**: the user/system prompt to evaluate.
- **Rubric file**: grading criteria and weights.
- **Reference files (optional)**: ground-truth or context docs.
- **Model config**: model name (e.g., `gpt-4o`), weights (e.g., `exact_weight`, `rubric_weight`), API key.

**Outputs**
- **PDF report** (path printed by the notebook): includes
  - Candidate answer
  - Verdict (e.g., Pass/Fail or score)
  - Rationale (rubric-aligned)
  - Optional references cited

---

## Programmatic use (from Python)

The notebook defines helper functions such as `run_pipeline(...)` and `regrade_pipeline(...)`.
A typical call looks like this (argument names may vary slightly based on the notebook):
```python
from pathlib import Path
from Grader_Config_RLHF import run_pipeline  # if exported as .py, or use %run in a notebook

pdf_path = run_pipeline(
    prompt_file=Path("inputs/prompt.txt"),
    rubric_file=Path("inputs/rubric.json"),
    ref_files=[Path("inputs/reference.pdf")],   # optional
    grader_model="gpt-4o",
    api_key_text=os.environ["OPENAI_API_KEY"],
    exact_weight=0.5,
    rubric_weight=0.5,
)
print("PDF saved to:", pdf_path)
```

> Tip: If you prefer a script, export the notebook to a module (`File → Save and Export As → Python`) and import its functions, or wrap the pipeline call in a small CLI script.

---

## Notes
- Large/binary attachments should be kept outside git or tracked with Git LFS.
- The PDF is built with `reportlab`; you can customize branding, page size, or fonts in the PDF builder section of the notebook.
- For re-grading, use `regrade_pipeline(...)` to supply a prior candidate response and override settings without re-generating.

---

## Troubleshooting
- **Path errors**: ensure `prompt_file`, `rubric_file`, and any `ref_files` exist.
- **Auth errors**: verify `OPENAI_API_KEY` is set and has model access.
- **Missing fonts** (on some systems): install a standard TTF or adjust the PDF builder to use built-in fonts.
