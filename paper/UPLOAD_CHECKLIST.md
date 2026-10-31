# Overleaf Upload Checklist — ArecaNut AI IEEE Paper

## Files to upload to Overleaf

| File | Source path in project | Upload as |
|------|------------------------|-----------|
| `main.tex` | `paper\main.tex` | `main.tex` |
| `confusion_matrix.png` | `paper\confusion_matrix.png` | `confusion_matrix.png` |
| `accuracy_loss_plot.png` | `paper\accuracy_loss_plot.png` | `accuracy_loss_plot.png` |

## Missing figures (placeholders in main.tex)

The following figures are referenced in `main.tex` but require you to create and upload them:

### Figure 3: Arecanut image comparison
**What is needed:** 2-4 real dataset images showing different classes side by side
(e.g., Healthy_Leaf vs. yellow leaf disease, Healthy_Trunk vs. Stem_bleeding).

**How to get them:**
1. Open `ml/disease/dataset/Arecanut_dataset/Arecanut_dataset/train/`
2. Pick one image from `Healthy_Leaf/` and one from `yellow leaf disease/`
3. Combine them side by side into a single image file
4. Upload as `arecanut_comparison.png`
5. Update the `\includegraphics` line in main.tex if needed

**Current status:** Not included. The LaTeX source does not contain a figure
environment for this — add one when you have the image.

### Figure 5: Application screenshots
**What is needed:** Screenshots of the React application showing:
- (a) Image upload page
- (b) Prediction result with red/green status badge

**How to get them:**
1. Start the Flask backend: `cd backend && python app.py`
2. Start the React frontend: `cd frontend && npm start`
3. Open http://localhost:3000 in Chrome
4. Navigate to Disease Detection, upload an image, take screenshots
5. Upload as `screenshot_upload.png` and `screenshot_result.png`

**Current status:** Not included. Add figure environments when screenshots
are available.

## Figure 1 and Figure 2: Architecture diagrams
These are already embedded as TikZ vector graphics in `main.tex` — no
image files needed. Figure 1 uses TikZ to draw the system architecture flow.

## Overleaf setup steps

1. Create a new Overleaf project (or open your existing one)
2. Upload all three files listed above to the project root
3. Set the compiler to **pdfLaTeX**
4. Set the main document to `main.tex`
5. Click Recompile

## Second author placeholder

The second author block in `main.tex` contains:
```
\IEEEauthorblockN{[Second Author's Full Name]}
```
Replace `[Second Author's Full Name]` with the actual name before submission.

## Pre-submission verification checklist

- [ ] Replace second author placeholder with actual name
- [ ] Add Figure 3 (arecanut image comparison) with real dataset images
- [ ] Add Figure 5 (application screenshots)
- [ ] Verify all numerical values match evaluation_metrics.json
- [ ] Run plagiarism check (Turnitin or similar)
- [ ] Verify references are accessible (check DOIs/URLs)
- [ ] Align Flask preprocessing with TensorFlow pipeline (or re-evaluate)
- [ ] Test the live application end-to-end before claiming deployment accuracy
- [ ] Check IEEE formatting compliance (page limit, font sizes)
- [ ] Review abstract word count (typically 150-250 words for IEEE)
- [ ] Verify all \cite{} references resolve in the bibliography
