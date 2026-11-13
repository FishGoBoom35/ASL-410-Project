# ASL Sign Recognition: Frontend

React + TypeScript + Vite UI for the CSC 410 ASL project, exported from the Figma Make design.

## Run it

```bash
cd frontend
npm install
npm run dev
```

Then open the URL Vite prints (usually http://localhost:5173).

## Status

UI only. All predictions, confidence scores and camera states are hardcoded demo data. Next step is wiring in the webcam (MediaPipe Hands) and the trained models from `train_baseline.py`.

## Assets still needed

Export these from the Figma file into `public/assets/`:

- `f67a7.png`: UTampa wordmark
- `75cc2.png`: Spartans seal
- `c76e9.svg`, `4a876.svg`: left/right hand landmark overlays (high confidence)
- `1b882.svg`, `6935a.svg`: left/right hand landmark overlays (low confidence)
