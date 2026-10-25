# RedKnot website

Static site (plain HTML/CSS/JS, no build step). KaTeX loads from jsDelivr.

```
website/
├── index.html                      # home
├── blog/index.html                 # post list
├── blog/redknot-explained/         # first post
└── assets/  style.css  charts.js  logo.png
```

Preview locally:

```bash
cd website && python3 -m http.server 8000   # open http://localhost:8000
```

Add a post: copy `blog/redknot-explained/` to a new folder, edit it, and add a card to `blog/index.html`.
Charts are drawn by `assets/charts.js` (`bars`, `split`, `diverge`, `headGrid`); each chart's data sits inline at the bottom of the post.

Deploys to GitHub Pages via `.github/workflows/pages.yml` on every push to `main` that touches `website/`.
