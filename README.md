# SHL Grammar Scoring Engine

Research Engineer hiring assessment — spoken grammar scoring (0-5) from 45-60s wav files.

## Data
- `Dataset_Final/train.csv` — 769 files + labels
- `Dataset_Final/test.csv` — 216 files (labels = -1, placeholder)
- `Dataset_Final/train/` and `Dataset_Final/test/` — wav files, 16kHz mono
- `Dataset_Final/sample_submission.csv` — submission format example

Note: train/test reuse the same filenames (e.g. `audio_19.wav` exists in both) but the audio content is different, so always join with the folder path, not just the filename.

## Approach (short)
1. EDA + audio preprocessing (trim silence, normalize)
2. ASR with Whisper-base to get transcripts (grammar lives in the words)
3. Features: linguistic stats + TF-IDF + acoustic stats (MFCC, ZCR, energy)
4. Models: Ridge + LightGBM, 5-fold CV, averaged ensemble. Clip to [0, 5].
5. Metrics: RMSE + Pearson (as per Kaggle eval). Train RMSE is printed in the notebook.

## Run on Kaggle
1. Create a Kaggle notebook with GPU (P100 is fine), upload `Dataset_Final` as input or as a dataset.
2. Upload `SHL_Grammar_Scoring.ipynb` and run all cells.
3. Transcription takes ~1-2h for ~985 files on GPU (cached to `transcripts.csv`, so re-runs are fast).
4. Output: `submission.csv` with `filename,label` for the test set.

## Run locally
```
pip install -r requirements.txt
jupyter notebook SHL_Grammar_Scoring.ipynb
```

## Files
- `SHL_Grammar_Scoring.ipynb` — main notebook (this is the submission notebook, includes report + train RMSE + plots)
- `requirements.txt`
- `Dataset_Final/` — data (not pushed to GitHub if too large, keep local / Kaggle dataset)

## Results (fill after run)
- CV RMSE / Pearson + full-train RMSE are printed in the notebook (section 6).
- See plots: label hist, duration vs score, pred-vs-true, residuals, feature importance.
