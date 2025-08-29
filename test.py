# import urllib.request
# import os

# # Create a folder to store the text files
# os.makedirs("gutenberg_texts", exist_ok=True)

# # Gutenberg books with their download links
# books = {
#     "pride_prejudice": "https://www.gutenberg.org/ebooks/1342.txt.utf-8",
#     "moby_dick": "https://www.gutenberg.org/ebooks/2701.txt.utf-8",
#     "frankenstein": "https://www.gutenberg.org/ebooks/84.txt.utf-8",
#     "dracula": "https://www.gutenberg.org/ebooks/345.txt.utf-8",
#     "alice_wonderland": "https://www.gutenberg.org/ebooks/11.txt.utf-8",
#     "tom_sawyer": "https://www.gutenberg.org/ebooks/74.txt.utf-8",
# }

# # Download and save each book
# for name, url in books.items():
#     try:
#         print(f"⬇️ Downloading {name}...")
#         response = urllib.request.urlopen(url)
#         raw_text = response.read().decode("utf-8", errors="ignore")

#         file_path = os.path.join("gutenberg_texts", f"{name}.txt")
#         with open(file_path, "w", encoding="utf-8") as f:
#             f.write(raw_text)

#         print(f"✅ Saved: {file_path}")
#     except Exception as e:
#         print(f"❌ Failed to download {name}: {e}")

from nltk.corpus import gutenberg

with open('data.txt','w',encoding='utf-8') as f:
    f.write(gutenberg.raw('shakespeare-hamlet.txt'))
    