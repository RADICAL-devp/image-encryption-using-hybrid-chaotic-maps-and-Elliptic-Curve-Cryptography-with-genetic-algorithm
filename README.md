
# Image Encryption and Decryption Project

## Modes
- `encrypt`: encrypt one file
- `decrypt`: decrypt one file

## Run
```bash
pip install -r requirements.txt
python main.py --mode encrypt --file path/to/image.jpg
python main.py --mode decrypt --file path/to/encrypted_image.png
```

## Output
Every run creates its own folder inside `output/`, for example:
- `output/encrypt_20260428_213500/`
- `output/decrypt_20260428_213742/`

Each run folder contains:
- the processed image
- the report file
