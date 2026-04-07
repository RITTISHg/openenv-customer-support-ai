"""HTTP server for the OpenEnv Customer Support environment.

Exposes the environment as a REST API for Hugging Face Space deployment.
The pre-submission validator pings POST /reset expecting HTTP 200.

Endpoints:
    POST /reset        — Reset the environment and return initial observation
    POST /step         — Take an action and return (observation, reward, done, info)
    GET  /state        — Return current ticket state
    GET  /health       — Health check
    GET  /             — Environment info
"""

import os
import sys
import json
import traceback
from typing import Any

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

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


# ── Request / Response models ──

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


# ── Endpoints ──

@app.get("/")
async def root():
    """Return environment info."""
    return {
        "name": "openenv-support",
        "version": "1.0.0",
        "description": "AI Customer Support Ticket Resolution Environment",
        "tasks": ["easy", "medium", "hard"],
        "endpoints": ["/reset", "/step", "/state", "/health"],
    }


@app.get("/health")
async def health():
    """Health check."""
    return {"status": "ok"}


@app.post("/reset")
async def reset(request: ResetRequest = ResetRequest()):
    """Reset the environment and return initial observation."""
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
    """Take an action and return observation, reward, done, info."""
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
    """Return the full internal state for debugging."""
    try:
        s = env.state()
        return s.model_dump()
    except RuntimeError as e:
        raise HTTPException(status_code=400, detail=str(e))


@app.post("/close")
async def close():
    """Close the environment."""
    env.close()
    return {"status": "closed"}


if __name__ == "__main__":
    import uvicorn
    port = int(os.getenv("PORT", "7860"))  # HF Spaces default port
    uvicorn.run(app, host="0.0.0.0", port=port)
