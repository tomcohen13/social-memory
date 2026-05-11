
import json, os, urllib.request                           
                                                                                        
URL = "https://cyborgpsychologylab--internvideo2-stage2-1b-internvideo2-bba4fc.modal.run" 

import asyncio
import requests

class VideoIntern2:

    async def encode(self, video_url: str, text: str):
        import concurrent.futures

        def sync_post():
            body = json.dumps({"video_url": video_url, "text": text})
            headers = {
                "X-API-Key": os.environ["INTERNVIDEO_API_KEY"],
                "Content-Type": "application/json",
            }
            resp = requests.post(URL, data=body, headers=headers, timeout=300)
            resp.raise_for_status()
            return resp.json()
        
        loop = asyncio.get_running_loop()
        return await loop.run_in_executor(None, sync_post)
           
       
