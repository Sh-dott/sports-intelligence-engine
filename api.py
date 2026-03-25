"""
Sports Intelligence Engine -- Web API
FastAPI backend serving match analysis with interactive Plotly dashboard.
"""

import json
from pathlib import Path

from fastapi import FastAPI, Request, HTTPException
from fastapi.responses import HTMLResponse, JSONResponse
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates

from engine.ingestion import load_match_data
from engine.processing import process_match
from engine.detection import DetectionEngine
from engine.statistics import StatisticalAnalyzer
from engine.insights import InsightGenerator, report_to_json
from engine.clustering import PatternAnalyzer
from engine.visualization_plotly import generate_all_plotly
from engine.storage import save_analysis, load_analysis, list_analyses
from engine.providers import get_provider, list_providers

app = FastAPI(title="Sports Intelligence Engine", version="1.0.0")

BASE_DIR = Path(__file__).resolve().parent
app.mount("/static", StaticFiles(directory=str(BASE_DIR / "static")), name="static")
from jinja2 import Environment, FileSystemLoader
_jinja_env = Environment(loader=FileSystemLoader(str(BASE_DIR / "templates")), autoescape=True)
_jinja_env.policies["json.dumps_kwargs"] = {"default": str}

def _tojson_filter(value):
    import json as _json
    return _json.dumps(value, default=str)

_jinja_env.filters["tojson"] = _tojson_filter


def render_template(name: str, request: Request, context: dict = None):
    """Render a Jinja2 template manually to avoid Starlette compatibility issues."""
    template = _jinja_env.get_template(name)
    ctx = context or {}
    ctx["request"] = request
    html = template.render(**ctx)
    return HTMLResponse(content=html)


@app.get("/", response_class=HTMLResponse)
async def home(request: Request):
    """Dashboard home page."""
    try:
        recent = list_analyses()[:10]
    except Exception:
        recent = []
    providers = list_providers()
    return render_template("index.html", request, {
        "recent_analyses": recent,
        "providers": providers,
    })


@app.get("/api/providers")
async def api_providers():
    """List available data providers."""
    return {"providers": list_providers()}


@app.get("/api/competitions")
async def api_competitions(provider: str):
    """List competitions for a provider."""
    try:
        p = get_provider(provider)
        comps = p.list_competitions()
        return JSONResponse(content=comps.to_dict(orient="records"))
    except Exception as e:
        raise HTTPException(status_code=400, detail=str(e))


@app.get("/api/matches")
async def api_matches(provider: str, competition: str, season: str = None):
    """List matches for a competition/season."""
    try:
        p = get_provider(provider)
        matches = p.list_matches(competition_id=competition, season_id=season)
        return JSONResponse(content=matches.to_dict(orient="records"))
    except Exception as e:
        raise HTTPException(status_code=400, detail=str(e))


@app.post("/api/analyze")
async def api_analyze(request: Request):
    """Run full analysis pipeline and return results."""
    body = await request.json()
    provider_name = body.get("provider")
    match_id = body.get("match_id")

    if not provider_name or not match_id:
        raise HTTPException(status_code=400, detail="provider and match_id required")

    try:
        # Fetch data
        provider = get_provider(provider_name)
        df = provider.get_match_events(match_id)
        sport = provider.get_sport()

        # Run pipeline
        processed_df = load_match_data(df, sport=sport)
        ctx = process_match(processed_df)

        engine = DetectionEngine(ctx)
        detected = engine.run_all()

        analyzer = StatisticalAnalyzer(ctx)
        stat_insights = analyzer.run_all()

        generator = InsightGenerator(ctx, detected, stat_insights)
        report = generator.generate_report()
        report_dict = json.loads(report_to_json(report))

        # Generate Plotly charts
        charts = generate_all_plotly(ctx, detected)
        report_dict["charts"] = charts

        # Store result
        analysis_id = save_analysis(report_dict)
        report_dict["analysis_id"] = analysis_id

        return JSONResponse(content=report_dict)

    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/api/analysis/{analysis_id}")
async def api_get_analysis(analysis_id: str):
    """Retrieve stored analysis."""
    data = load_analysis(analysis_id)
    if not data:
        raise HTTPException(status_code=404, detail="Analysis not found")
    return JSONResponse(content=data)


@app.get("/analysis/{analysis_id}", response_class=HTMLResponse)
async def view_analysis(request: Request, analysis_id: str):
    """Render analysis page."""
    data = load_analysis(analysis_id)
    if not data:
        raise HTTPException(status_code=404, detail="Analysis not found")
    return render_template("analysis.html", request, {
        "report": data,
        "analysis_id": analysis_id,
    })


@app.get("/api/recent")
async def api_recent():
    """List recent analyses."""
    return JSONResponse(content=list_analyses()[:20])


@app.post("/api/season")
async def api_season(request: Request):
    """Run season-level multi-match analysis."""
    from engine.multi_match import analyze_season, season_to_json

    body = await request.json()
    provider_name = body.get("provider")
    competition_id = body.get("competition")
    season_id = body.get("season")
    max_matches = body.get("max_matches", 15)

    if not all([provider_name, competition_id, season_id]):
        raise HTTPException(status_code=400, detail="provider, competition, and season required")

    try:
        ctx = analyze_season(provider_name, competition_id, season_id, max_matches=max_matches)
        result = json.loads(season_to_json(ctx))
        return JSONResponse(content=result)
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/debug")
async def debug(request: Request):
    """Debug route to test template rendering."""
    import traceback
    try:
        recent = list_analyses()[:10]
        providers = list_providers()
        return render_template("index.html", request, {
            "recent_analyses": recent,
            "providers": providers,
        })
    except Exception:
        return JSONResponse(content={"error": traceback.format_exc()}, status_code=500)


@app.get("/health")
async def health():
    return {"status": "ok", "providers": list_providers()}
