import tiktoken
from duckduckgo_search import DDGS
from typing import List, Dict
import requests
import re
from readability import Document
from bs4 import BeautifulSoup

TOKEN_LIMIT = 20000
MIN_TOKENS = 1000


def num_tokens_from_string(string: str) -> int:
    """Returns the number of tokens in a text string."""
    encoding = tiktoken.encoding_for_model("gpt-4o")
    num_tokens = len(encoding.encode(string))
    return num_tokens


def search(query: str, max_results: int = 15) -> List[Dict]:
    results = DDGS().text(query, max_results=max_results)

    def extract_url_content(search_results: List[Dict]) -> List[Dict]:
        processed_results = []
        total_tokens = 0

        for result in search_results:
            url = result["href"]
            title = result["title"]
            try:
                response = requests.get(url, timeout=5)
                if response.status_code == 200:
                    html_content = response.text

                    # 使用 readability 提取主要內容
                    main_content = extract_main_content_readability(html_content)
                    content_tokens = num_tokens_from_string(main_content)

                    # 檢查是否超過 token 限制
                    if total_tokens + content_tokens > TOKEN_LIMIT:
                        if len(processed_results) == 0:
                            # 如果沒有任何結果，仍然返回第一個搜尋結果
                            processed_results.append(
                                {"title": title, "url": url, "content": main_content}
                            )
                        break  # 達到限制則立即返回結果

                    processed_results.append(
                        {"title": title, "url": url, "content": main_content}
                    )
                    total_tokens += content_tokens  # 更新當前的 token 計數

            except Exception as e:
                print(f"Error processing {url}: {e}")

        # 確保搜尋結果的總 tokens 至少 MIN_TOKENS
        if total_tokens < MIN_TOKENS and search_results:
            for result in search_results[len(processed_results):]:
                url = result["href"]
                title = result["title"]
                try:
                    response = requests.get(url, timeout=5)
                    if response.status_code == 200:
                        html_content = response.text

                        # 使用 readability 提取主要內容
                        main_content = extract_main_content_readability(html_content)
                        content_tokens = num_tokens_from_string(main_content)

                        processed_results.append(
                            {"title": title, "url": url, "content": main_content}
                        )
                        total_tokens += content_tokens

                        if total_tokens >= MIN_TOKENS:
                            break  # 達到最小 token 限制則停止
                except Exception as e:
                    print(f"Error processing {url}: {e}")

        return processed_results

    def extract_main_content(soup) -> str:
        main_candidates = [
            soup.find('main'),
            soup.find('article'),
            soup.find(id='content'),
            soup.find(class_='content'),
            soup.find(id='main')
        ]
        for candidate in main_candidates:
            if candidate:
                # 返回純文本，移除多餘空白
                text = candidate.get_text(separator=' ', strip=True)
                text = re.sub(r'\s+', ' ', text)  # 清理文本
                return text

        text = soup.body.get_text(separator=' ', strip=True)
        text = re.sub(r'\s+', ' ', text)
        return text
    
    def extract_main_content_readability(html) -> str:
        """使用 readability-lxml 提取主要內容"""
        doc = Document(html)
        content_html = doc.summary()
        soup = BeautifulSoup(content_html, 'html.parser')
        text = soup.get_text(separator=' ', strip=True)
        text = re.sub(r'\s+', ' ', text)
        return text

    return extract_url_content(results)
