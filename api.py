"""
Sports Intelligence Engine -- Web API
FastAPI backend serving match analysis with interactive Plotly dashboard.
MongoDB Atlas caching for instant responses.
"""

import json
from pathlib import Path

from fastapi import FastAPI, Request, HTTPException
from fastapi.responses import HTMLResponse, JSONResponse
from fastapi.staticfiles import StaticFiles

from engine.ingestion import load_match_data
from engine.processing import process_match
from engine.detection import DetectionEngine
from engine.statistics import StatisticalAnalyzer
from engine.insights import InsightGenerator, report_to_json
from engine.clustering import PatternAnalyzer
from engine.visualization_plotly import generate_all_plotly
from engine.storage import save_analysis, load_analysis, list_analyses
from engine.providers import get_provider, list_providers
from engine.providers import mongo_cache

app = FastAPI(title="Sports Intelligence Engine", version="2.0.0")

BASE_DIR = Path(__file__).resolve().parent
app.mount("/static", StaticFiles(directory=str(BASE_DIR / "static")), name="static")

from jinja2 import Environment, FileSystemLoader
from markupsafe import Markup
_jinja_env = Environment(loader=FileSystemLoader(str(BASE_DIR / "templates")), autoescape=True)


def _tojson_filter(value):
    import json as _json
    return Markup(_json.dumps(value, default=str))

_jinja_env.filters["tojson"] = _tojson_filter


def render_template(name: str, request: Request, context: dict = None):
    template = _jinja_env.get_template(name)
    ctx = context or {}
    ctx["request"] = request
    return HTMLResponse(content=template.render(**ctx))


# Only expose clean provider names to the UI
UI_PROVIDERS = {
    "football": "Football (All Leagues & Tournaments)",
    "nba": "Basketball (NBA)",
}


@app.get("/", response_class=HTMLResponse)
async def home(request: Request):
    # Use MongoDB for recent analyses
    try:
        recent = mongo_cache.list_cached_analyses(10)
    except Exception:
        try:
            recent = list_analyses()[:10]
        except Exception:
            recent = []

    return render_template("index.html", request, {
        "recent_analyses": recent,
        "providers": UI_PROVIDERS,
    })


@app.get("/api/providers")
async def api_providers():
    return {"providers": UI_PROVIDERS}


@app.get("/api/competitions")
async def api_competitions(provider: str):
    try:
        p = get_provider(provider)
        comps = p.list_competitions()
        return JSONResponse(content=comps.to_dict(orient="records"))
    except Exception as e:
        raise HTTPException(status_code=400, detail=str(e))


@app.get("/api/matches")
async def api_matches(provider: str, competition: str, season: str = None):
    try:
        p = get_provider(provider)
        matches = p.list_matches(competition_id=competition, season_id=season)
        return JSONResponse(content=matches.to_dict(orient="records"))
    except ValueError as e:
        # Return empty list with error info instead of HTTP error (so JS can show it)
        return JSONResponse(content=[{"match_id": 0, "home_team": str(e), "away_team": "", "match_date": "", "score": ""}])
    except Exception as e:
        raise HTTPException(status_code=400, detail=str(e))


@app.post("/api/analyze")
async def api_analyze(request: Request):
    body = await request.json()
    provider_name = body.get("provider")
    match_id = body.get("match_id")

    if not provider_name or not match_id:
        raise HTTPException(status_code=400, detail="provider and match_id required")

    try:
        # Check MongoDB for cached analysis (instant!)
        cached = mongo_cache.get_cached_analysis(provider_name, match_id)
        if cached:
            return JSONResponse(content=cached)

        # Fetch and analyze
        provider = get_provider(provider_name)
        df = provider.get_match_events(match_id)
        sport = provider.get_sport()

        processed_df = load_match_data(df, sport=sport)
        ctx = process_match(processed_df)

        engine = DetectionEngine(ctx)
        detected = engine.run_all()

        analyzer = StatisticalAnalyzer(ctx)
        stat_insights = analyzer.run_all()

        generator = InsightGenerator(ctx, detected, stat_insights)
        report = generator.generate_report()
        report_dict = json.loads(report_to_json(report))

        charts = generate_all_plotly(ctx, detected)
        report_dict["charts"] = charts

        analysis_id = save_analysis(report_dict)
        report_dict["analysis_id"] = analysis_id

        # Cache in MongoDB for instant future loads
        mongo_cache.cache_analysis(provider_name, match_id, report_dict)

        return JSONResponse(content=report_dict)

    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/api/analysis/{analysis_id}")
async def api_get_analysis(analysis_id: str):
    data = load_analysis(analysis_id)
    if not data:
        raise HTTPException(status_code=404, detail="Analysis not found")
    return JSONResponse(content=data)


@app.get("/analysis/{analysis_id}", response_class=HTMLResponse)
async def view_analysis(request: Request, analysis_id: str):
    data = load_analysis(analysis_id)
    if not data:
        raise HTTPException(status_code=404, detail="Analysis not found")
    return render_template("analysis.html", request, {
        "report": data,
        "analysis_id": analysis_id,
    })


@app.get("/api/recent")
async def api_recent():
    try:
        return JSONResponse(content=mongo_cache.list_cached_analyses(20))
    except Exception:
        return JSONResponse(content=list_analyses()[:20])


@app.post("/api/season")
async def api_season(request: Request):
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


@app.get("/health")
async def health():
    return {"status": "ok", "providers": list(UI_PROVIDERS.keys())}
