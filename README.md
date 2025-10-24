Bonjour Antonio 👋

J’ai préparé une **API FastAPI “prod-ready”** pour `deepseek-ai/DeepSeek-OCR` avec :

* backends **Transformers (par défaut)** et **vLLM** (nightly officiel)
* endpoints `/ocr` (images), `/ocr/pdf` (PDF→pages), `/healthz`, `/readyz`, `/metrics`
* presets de taille (tiny/small/base/large/**gundam**) et **prompt Markdown** par défaut
* limites/concurrence, CORS, Prometheus, Dockerfiles **GPU** & **CPU (dev)**

➡️ Tout le code, Dockerfiles et README sont dans le **canvas** à droite.

Quelques repères importants (source modèle) :

* Chargement **HF Transformers** + `flash-attn` avec l’API `model.infer(...)`, presets et prompt recommandés. ([huggingface.co][1])
* **Support vLLM officiel** (guide + snippet) pour batch/latence améliorés. ([huggingface.co][1])

### Démarrage express (GPU)

```bash
docker build -t deepseek-ocr-api -f Dockerfile.gpu .
docker run --rm --gpus all -p 8000:8000 \
  -e BACKEND=transformers -e TORCH_DTYPE=bf16 deepseek-ocr-api
```

### Appels rapides

* Image → Markdown :

```bash
curl -s http://localhost:8000/ocr -H 'Content-Type: application/json' -d '{
  "prompt":"<image>\n<|grounding|>Convert the document to markdown.",
  "urls":["https://example.com/sample.png"],
  "mode":{"preset":"gundam"}
}' | jq -r '.results[0].text'
```

* PDF (pages 1–5) :

```bash
curl -s http://localhost:8000/ocr/pdf -H 'Content-Type: application/json' -d '{
  "pdf_url":"https://example.com/report.pdf",
  "pdf_pages":{"page_from":1,"page_to":5,"dpi":180},
  "mode":{"preset":"base"}
}'
```

### Notes “best practice”

* **CUDA 12.4 + Torch 2.6.0 cu124** dans l’image GPU (cohérent avec les wheels récentes).
* **flash-attn 2.7.3** activé (`_attn_implementation='flash_attention_2'`) comme sur la fiche HF. ([huggingface.co][1])
* **vLLM** : passer `BACKEND=vllm` si tu veux pousser le throughput en batch (nightly ≥ 2025-10-23). ([huggingface.co][1])
* **Probes** K8S prêtes (`/healthz`, `/readyz`) + **Prometheus** (`/metrics`).
* Concurrency process via sémaphore → scale horizontal recommandé sous K8S (HPA).

Tu veux que je t’ajoute un **chart Helm minimal** (IngressClass `public`) + `values.yaml` pour ton cluster ?

[1]: https://huggingface.co/deepseek-ai/DeepSeek-OCR "deepseek-ai/DeepSeek-OCR · Hugging Face"
