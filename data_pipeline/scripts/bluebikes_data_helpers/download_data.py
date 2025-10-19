# pip install requests beautifulsoup4

import os
import re
import xml.etree.ElementTree as ET
from urllib.parse import urljoin, urlparse

import requests
from bs4 import BeautifulSoup


def _guess_s3_bucket(index_url: str) -> str | None:
    """
    Best-effort bucket extractor for common S3 URLs.
    Supports:
      - https://s3.amazonaws.com/<bucket>/index.html
      - https://<bucket>.s3.amazonaws.com/index.html
    """
    u = urlparse(index_url)
    host = u.netloc
    path = u.path.strip("/")

    # host style: <bucket>.s3.amazonaws.com
    if host.endswith(".s3.amazonaws.com"):
        return host.split(".s3.amazonaws.com")[0] or None

    # path style: s3.amazonaws.com/<bucket>/...
    if host == "s3.amazonaws.com" and path:
        return path.split("/", 1)[0]

    return None


def _list_s3_zip_urls(bucket: str) -> list[str]:
    """
    Use S3's ListObjectsV2 XML API to list all objects and return .zip URLs.
    """
    base = f"https://s3.amazonaws.com/{bucket}"
    params = {"list-type": "2", "max-keys": "1000"}
    urls: list[str] = []
    token = None

    while True:
        if token:
            params["continuation-token"] = token
        resp = requests.get(base, params=params, timeout=30)
        resp.raise_for_status()

        # Parse XML
        root = ET.fromstring(resp.text)
        ns = {"s3": "http://s3.amazonaws.com/doc/2006-03-01/"}

        for c in root.findall("s3:Contents", ns):
            key = c.findtext("s3:Key", default="", namespaces=ns)
            if key.lower().endswith(".zip"):
                urls.append(f"{base}/{key}")

        is_truncated = root.findtext("s3:IsTruncated", default="false", namespaces=ns).lower() == "true"
        if not is_truncated:
            break
        token = root.findtext("s3:NextContinuationToken", default=None, namespaces=ns)

    return urls


def find_zip_links(index_url: str, names: list[str]) -> list[str]:
    """
    Return .zip links that match any of the given names (partial match, case-insensitive).
    Tries HTML first; if none found, falls back to S3 XML listing.
    """
    # 1) Try to scrape HTML <a href="...zip">
    zip_urls: list[str] = []
    try:
        resp = requests.get(index_url, timeout=30)
        resp.raise_for_status()
        soup = BeautifulSoup(resp.text, "html.parser")
        for a in soup.find_all("a", href=True):
            href = a["href"]
            if href.lower().endswith(".zip"):
                zip_urls.append(urljoin(index_url, href))
    except requests.RequestException:
        pass  # will try S3 fallback

    # 2) If HTML had nothing, fall back to S3 ListObjectsV2
    if not zip_urls:
        bucket = _guess_s3_bucket(index_url)
        if bucket:
            try:
                zip_urls = _list_s3_zip_urls(bucket)
            except requests.RequestException:
                pass

    # 3) Filter by names (regex, case-insensitive)
    if not names:
        return zip_urls
    matched: list[str] = []
    for link in zip_urls:
        for name in names:
            if re.search(name, link, re.IGNORECASE):
                matched.append(link)
                break
    return matched


def download_zips(urls: list[str], out_dir: str = "zips") -> list[str]:
    """
    Download each URL in `urls` to `out_dir`. Skips files that already exist.
    Returns list of saved file paths.
    """
    os.makedirs(out_dir, exist_ok=True)
    saved: list[str] = []

    for zurl in urls:
        fname = os.path.join(out_dir, zurl.rsplit("/", 1)[-1])
        if os.path.exists(fname):
            saved.append(fname)
            continue

        with requests.get(zurl, stream=True, timeout=60) as r:
            r.raise_for_status()
            with open(fname, "wb") as f:
                for chunk in r.iter_content(chunk_size=1024 * 64):
                    if chunk:
                        f.write(chunk)
        print(f"Downloaded: {fname}")
        saved.append(fname)

    return saved






if __name__ == "__main__":
    # Example: find and download only 2015 zips from the Hubway (Bluebikes) S3 bucket
    index = "https://s3.amazonaws.com/hubway-data/index.html"
    wanted_names = ["2023","2024","2025"]  # partial names or exact fragments, e.g. "201501"

    zip_urls = find_zip_links(index, wanted_names)
    print("Found:", *zip_urls, sep="\n")

    paths = download_zips(zip_urls, out_dir="bluebikes_zips")
    print("\nSaved files:", *paths, sep="\n")
