"""
server/app.py — Required by multi-mode deployment validator.

Exposes the FastAPI application and a callable main() entry point
for the openenv validate check.
"""

import os
import sys

# Ensure project root is on path so src.* imports work
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel

from src.env import SupportEnv
from src.models import Action, ActionType

app = FastAPI(
    title="OpenEnv Customer Support",
    description="AI Customer Support Ticket Resolution Environment",
    version="1.0.0",
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

# Global environment instance
env = SupportEnv()


class ResetRequest(BaseModel):
    task_name: str = "easy"


class StepRequest(BaseModel):
    action_type: str = "reply"
    message: str = ""
    category: str = ""
    escalation_reason: str = ""


class EnvResponse(BaseModel):
    observation: dict | None = None
    reward: dict | None = None
    done: bool = False
    info: dict | None = None
    error: str | None = None


@app.get("/")
async def root():
    return {
        "name": "openenv-support",
        "version": "1.0.0",
        "description": "AI Customer Support Ticket Resolution Environment",
        "tasks": ["easy", "medium", "hard"],
        "endpoints": ["/reset", "/step", "/state", "/health"],
    }


@app.get("/health")
async def health():
    return {"status": "ok"}


@app.post("/reset")
async def reset(request: ResetRequest = ResetRequest()):
    try:
        obs = env.reset(request.task_name)
        return EnvResponse(
            observation=obs.model_dump(),
            done=False,
            info={"task": request.task_name, "status": "reset"},
        )
    except Exception as e:
        raise HTTPException(status_code=400, detail=str(e))


@app.post("/step")
async def step(request: StepRequest):
    try:
        action = Action(
            action_type=ActionType(request.action_type),
            message=request.message,
            category=request.category,
            escalation_reason=request.escalation_reason,
        )
        obs, reward, done, info = env.step(action)
        return EnvResponse(
            observation=obs.model_dump(),
            reward=reward.model_dump(),
            done=done,
            info=info,
        )
    except RuntimeError as e:
        raise HTTPException(status_code=400, detail=str(e))
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/state")
async def state():
    try:
        s = env.state()
        return s.model_dump()
    except RuntimeError as e:
        raise HTTPException(status_code=400, detail=str(e))


@app.post("/close")
async def close():
    env.close()
    return {"status": "closed"}


def main():
    """Entry point — runs the HTTP server. Required by openenv validate."""
    import uvicorn
    port = int(os.getenv("PORT", "7860"))
    uvicorn.run(app, host="0.0.0.0", port=port)


if __name__ == "__main__":
    main()
