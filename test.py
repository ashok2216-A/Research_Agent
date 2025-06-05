# import requests

# def search_semantic_scholar(query: str, limit: int = 3):
#     base_url = "https://api.semanticscholar.org/graph/v1/paper/search"
    
#     params = {
#         "query": query,
#         "limit": limit,
#         "fields": "title,abstract,authors,year,url"
#     }
    
#     try:
#         response = requests.get(base_url, params=params)
#         response.raise_for_status()
#         results = response.json().get("data", [])
        
#         if not results:
#             print("No results found.")
#             return
        
#         for idx, paper in enumerate(results, 1):
#             print(f"\n Paper {idx}:")
#             print(f"Title     : {paper.get('title')}")
#             print(f"Authors   : {', '.join([a['name'] for a in paper.get('authors', [])])}")
#             print(f"Year      : {paper.get('year')}")
#             print(f"URL       : {paper.get('url')}")
#             print(f"Abstract  : {paper.get('abstract')[:300]}...\n")  # Truncated for readability
            
#     except requests.exceptions.RequestException as e:
#         print(f"Error: {e}")

# # 🔍 Example Usage
# search_semantic_scholar("large language models healthcare", limit=3)



import requests

# Replace with your actual API key
API_KEY = "AIzaSyAO32s3nsYmbvmACUKSY6SDaJ-tNzMif3Q"

def youtube_search(query, max_results=5):
    search_url = "https://www.googleapis.com/youtube/v3/search"
    
    params = {
        "part": "snippet",
        "q": query,
        "type": "video",
        "maxResults": max_results,
        "key": API_KEY
    }
    
    response = requests.get(search_url, params=params)
    if response.status_code != 200:
        print("Failed to fetch data:", response.text)
        return
    
    results = response.json().get("items", [])
    for idx, item in enumerate(results, start=1):
        title = item["snippet"]["title"].encode("ascii", errors="ignore").decode()
        channel = item["snippet"]["channelTitle"].encode("ascii", errors="ignore").decode()
        video_id = item["id"]["videoId"]
        url = f"https://www.youtube.com/watch?v={video_id}"
        
        print(f"\nVideo {idx}")
        print(f"Title   : {title}")
        print(f"Channel : {channel}")
        print(f"URL     : {url}")

# 🔍 Example usage
youtube_search("Narrow AI VS AGI", max_results=3)

