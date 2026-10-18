# DistilBERT Thematic Analysis Studio

Run `Start Thematic Analysis.bat` from this `Thematic Analysis` folder. The launcher uses the installed Python 3.14 environment, starts the local server at <http://localhost:8050>, and opens the browser after startup. Keep its terminal window open while working.

For manual startup, run `py -3.14 -m pip install -r requirements.txt` and then `py -3.14 app.py` in this directory.

The ONNX classifier is loaded locally from `model/distilbert_onnx` or `ml/distilbert_onnx`. Set `DISTILBERT_MODEL_DIR` to a different directory if needed. The model's tokenizer and ONNX external data file accompany `model.onnx`. If ONNX is unavailable and local `model.safetensors` or `pytorch_model.bin` weights exist, the service can use PyTorch Transformers when `torch` is installed. The status pill explicitly says when neither model loads and heuristic estimates are being used instead.

The studio operates purely via dataset file import, supporting CSV (.csv), Excel (.xlsx), JSON (.json), XML (.xml), and Plain Text (.txt) formats with auto-detected schema column mapping. Uploads are limited to 10 MB and analyses to 5,000 records. Codebook changes and manual theme overrides can be exported; they are kept in memory for the current server session, not saved to a database.

Verify the backend and ONNX inference with `py -3.14 -m unittest discover -s tests -p test_studio.py -v`. With the server running, verify the UI with `py -3.14 tests/test_studio_browser.py`.
