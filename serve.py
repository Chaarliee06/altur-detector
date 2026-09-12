"""Start the service locally or on a host that assigns PORT."""
import os
import uvicorn

if __name__ == "__main__":
    uvicorn.run("app:app", host="0.0.0.0", port=int(os.environ.get("PORT", "8000")),
                workers=1, limit_concurrency=8, timeout_keep_alive=5, access_log=False)
