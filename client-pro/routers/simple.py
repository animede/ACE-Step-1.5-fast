"""
シンプルモード ページ配信
"""
from fastapi import APIRouter, Request
from fastapi.responses import HTMLResponse
from fastapi.templating import Jinja2Templates

router = APIRouter(tags=["simple"])
templates = Jinja2Templates(directory="templates")


@router.get("/simple", response_class=HTMLResponse)
async def simple_page(request: Request):
    """シンプルモードページ"""
    return templates.TemplateResponse(request, "simple.html", {"title": "ACE-Step 1.5 — シンプルモード"
    })
