import os
import re
import requests
import pymupdf4llm
import traceback

os.makedirs("papers", exist_ok=True)

with open("blammoe-spec-and-bibliography-v4.md", "r") as f:
    text = f.read()

# Extract arxiv ids
arxiv_ids = re.findall(r'(?:arXiv|arxiv)[\s:]*(\d{4}\.\d{4,5})', text)
# Extract github links
github_links = re.findall(r'github\.com/([^\s\|]+)', text)

print(f"Found {len(set(arxiv_ids))} unique arXiv IDs: {set(arxiv_ids)}")
print(f"Found {len(set(github_links))} unique GitHub links: {set(github_links)}")

headers = {'User-Agent': 'Mozilla/5.0 BLaMMoE-Agent'}

for aid in set(arxiv_ids):
    pdf_path = f"papers/{aid}.pdf"
    md_path = f"papers/{aid}.md"
    if not os.path.exists(md_path):
        print(f"Downloading {aid}...")
        try:
            url = f"https://arxiv.org/pdf/{aid}.pdf"
            r = requests.get(url, headers=headers, timeout=30)
            if r.status_code == 200 and r.headers.get('Content-Type', '').startswith('application/pdf'):
                with open(pdf_path, "wb") as f:
                    f.write(r.content)
                print(f"Converting {aid} to markdown...")
                md_text = pymupdf4llm.to_markdown(pdf_path)
                with open(md_path, "w", encoding='utf-8') as f:
                    f.write(md_text)
                print(f"Successfully converted {aid}")
                
                # Optionally delete PDF to save space
                os.remove(pdf_path)
            else:
                print(f"Failed to download {aid}: HTTP {r.status_code}")
                # Possibly fictional or wrong ID
        except Exception as e:
            print(f"Error processing {aid}: {e}")
            traceback.print_exc()

for repo in set(github_links):
    repo = repo.rstrip(')].,')
    md_path = f"papers/{repo.replace('/', '_')}.md"
    if not os.path.exists(md_path):
        print(f"Fetching GitHub repo readme for {repo}...")
        try:
            url = f"https://raw.githubusercontent.com/{repo}/main/README.md"
            r = requests.get(url, headers=headers, timeout=10)
            if r.status_code != 200:
                url = f"https://raw.githubusercontent.com/{repo}/master/README.md"
                r = requests.get(url, headers=headers, timeout=10)
            if r.status_code == 200:
                with open(md_path, "w", encoding='utf-8') as f:
                    f.write(r.text)
                print(f"Successfully fetched readme for {repo}")
            else:
                print(f"Failed to fetch readme for {repo}: HTTP {r.status_code}")
        except Exception as e:
            print(f"Error processing {repo}: {e}")

print("Done processing all sources.")
