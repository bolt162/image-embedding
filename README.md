# Image Embedding Similarity with Cohere Embed v4.0

This assignment demonstrates how to generate **image embeddings** and perform **text-to-image semantic similarity search** using the **Cohere Embed v4.0** model.  
Embeddings allow comparison of images and text based on meaning rather than exact content — useful for tasks like search, clustering, and classification.

---

## Learning Objective
Understand how to:
- Represent images as embeddings (numerical vectors)
- Compare embeddings using cosine similarity
- Search for the most semantically similar images based on text queries

<img width="813" height="311" alt="Screenshot 2025-11-07 at 1 39 35 PM" src="https://github.com/user-attachments/assets/249f9d5f-64b2-46d7-a5af-d61c182e9950" />

---

## Files
- `cohere_image_embeddings.py` — Main Python script  
- `README.md` — This documentation file  

---

## Requirements
- **Python 3.11**
- Cohere API key (get one [here](https://dashboard.cohere.com/api-keys))
- Internet connection to download example images

Install dependencies:
```bash
pip install cohere pillow numpy requests
