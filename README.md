---
title: QuickBite Late Order Check
emoji: 🚚
colorFrom: blue
colorTo: green
sdk: gradio
sdk_version: 6.5.1
python_version: 3.11
app_file: app.py
pinned: false
short_description: Will this order arrive late?
---

# QuickBite Late Order Check

## What it does

The dispatcher enters order-time attributes and sees the predicted chance of late delivery,
its complementary on-time chance, and whether to offer a $4 voucher. This application loads
**the user's saved Lab 05 pipeline and threshold**, not a substitute model.

**Space URL: https://huggingface.co/spaces/kawsarbuet/lab06-ship-it

## How to run it locally

```bash
conda env create -f environment.yml
conda activate lab06-ship-it
python -m gradio app.py
```

For alternative local startup: `python app.py`.

## Decisions

The input fields are generated **directly** from the feature lists in
`models/model_info.json`. The archived Lab 05 model uses only **five** features;
therefore the following are the actual enforceable rules:

| Field | Allowed values | Reason |
| --- | --- | --- |
| `distance_km` | 0.1–30, numeric | Limit to Lab 06's stated training-range support. |
| `items` | 1–20, whole number | Fractional items and out-of-range orders are invalid. |
| `hour` | 10–22, whole number | Only supported order placement hours are accepted. |
| `weather` | `clear`, `rain`, `snow` (exact case) | Prevent unseen categories silently encoded as all zero. |
| `day_type` | `weekday`, `weekend` (exact case) | Same unseen-category protection. |

**Lab 05 / Lab 06 mismatch:** The Lab 06 handout also specifies `prep_minutes`
(optional), `courier_months`, and `vehicle`. However **none is present in the saved
Lab 05 model's `features` list**. Adding them to the app would create fake inputs that
could not affect predictions and violate the requirement to derive fields from
`model_info.json`. To comply fully with the 8-field version, retrain and export a
compatible Lab 05 model with those features (only if they are known at order time),
then replace the model and metadata. The validator already supports all eight rules
when the new feature names appear in metadata; the page will add them automatically.

The saved pipeline uses `handle_unknown="ignore"`; a raw input of `"Snow"` would
be treated as unknown rather than as `"snow"`, potentially producing a misleading
probability. `predict_order` refuses `"Snow"` with `ValueError` before inference.
**Do not infer a particular probability difference without rerunning the model.**

The model was trained with scikit-learn **1.8.0**, according to its saved metadata.
The older Lab 05 ZIP's `requirements.txt` incorrectly specifies **1.6.1**.
This Lab 06 project uses **1.8.0** to match the metadata and local tested load.

A late probability **at or above 0.25** triggers a $4 voucher; this threshold is
read from `model_info.json`, not embedded in inference logic.

## Result

Existing Lab 05 held-out test results from `models/model_info.json` (750 cases):

| Metric | Lab 05 result |
| --- | ---: |
| Threshold | 0.25 |
| Accuracy | 0.8093 |
| Precision | 0.3898 |
| Recall | 0.3932 |
| F1 | 0.3915 |
| ROC AUC | 0.7477 |
| Confusion matrix | TN 561, FP 72, FN 71, TP 46 |
| Flag nobody cost | $1,755 |
| Flag everybody cost | $3,000 |
| Model policy cost | $1,537 |

The saved model reduced modeled cost by **$218 compared with offering nobody a
voucher** on this test set, and by **$1,463 compared with offering everyone a
voucher**. These are **Lab 05 offline results**, not live deployment evidence.

## Tests

Tests cover harder-vs-easier ranking, threshold boundary, metadata threshold,
validation of numerical bounds and categories, missing fields, load-once
behavior, and deployment file/version agreement. The optional `prep_minutes`
test is explicitly skipped for this five-feature model (it becomes active if the
model is replaced with one containing that feature).

```bash
python -m ruff format --check .
python -m ruff check .
python -m pytest
```

Run these after cloning the committed repository into a clean directory.

## Limitations

- The Lab 05 training data were synthetic QuickBite orders; field deployment
  quality and calibration are not established.
- This model cannot see events such as road closures, unexpected traffic or
  courier breakdowns, and should not make automatic courier or personnel decisions.
- A dispatcher should treat probabilities as guidance, not promised delivery times.
- The current model lacks three Lab 06 example fields. It must **not** be claimed
  to support them without retraining and updated metadata.
- The public Space must contain no private customer records, access tokens, or
  passwords. Upload only trusted model files; joblib uses pickle internally.

## How to publish and submit

1. Create a new repository `lab06-ship-it` and commit all files, including `models/`.
2. Make at least **five meaningful commits** (for example, import model, add
   inference, add tests, add Gradio app, add documentation). Do not fabricate history.
3. Create a public Hugging Face Space with Gradio SDK and Python 3.11; upload/push
   exactly these project files. Use the real Space URL in this README.
4. Verify both links in a signed-out browser, run all three checks from a fresh
   clone, and submit the GitHub and Space URLs in Blackboard.
