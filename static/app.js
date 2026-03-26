// Sports Intelligence Engine - Dashboard JS

let competitionsData = [];

async function loadCompetitions() {
    const provider = document.getElementById('provider').value;
    const compSelect = document.getElementById('competition');
    const seasonSelect = document.getElementById('season');
    const matchSelect = document.getElementById('match');

    compSelect.innerHTML = '<option value="">Loading...</option>';
    compSelect.disabled = true;
    seasonSelect.innerHTML = '<option value="">Select season...</option>';
    seasonSelect.disabled = true;
    matchSelect.innerHTML = '<option value="">Select match...</option>';
    matchSelect.disabled = true;
    document.getElementById('analyzeBtn').disabled = true;

    if (!provider) return;

    try {
        const res = await fetch(`/api/competitions?provider=${provider}`);
        competitionsData = await res.json();

        // Get unique competitions
        const comps = {};
        competitionsData.forEach(c => {
            if (!comps[c.competition_id]) {
                comps[c.competition_id] = c.competition_name;
            }
        });

        compSelect.innerHTML = '<option value="">Select competition...</option>';
        Object.entries(comps).sort((a, b) => a[1].localeCompare(b[1])).forEach(([id, name]) => {
            compSelect.innerHTML += `<option value="${id}">${name}</option>`;
        });
        compSelect.disabled = false;
    } catch (e) {
        compSelect.innerHTML = '<option value="">Error loading</option>';
        console.error(e);
    }
}

function loadSeasons() {
    const compId = document.getElementById('competition').value;
    const seasonSelect = document.getElementById('season');
    const matchSelect = document.getElementById('match');

    seasonSelect.innerHTML = '<option value="">Select season...</option>';
    matchSelect.innerHTML = '<option value="">Select match...</option>';
    matchSelect.disabled = true;
    document.getElementById('analyzeBtn').disabled = true;

    if (!compId) return;

    const seasons = competitionsData
        .filter(c => String(c.competition_id) === compId)
        .map(c => ({ id: c.season_id, name: c.season_name }));

    // Deduplicate
    const seen = new Set();
    seasons.forEach(s => {
        if (!seen.has(s.id)) {
            seen.add(s.id);
            seasonSelect.innerHTML += `<option value="${s.id}">${s.name}</option>`;
        }
    });
    seasonSelect.disabled = false;
}

async function loadMatches() {
    const provider = document.getElementById('provider').value;
    const compId = document.getElementById('competition').value;
    const seasonId = document.getElementById('season').value;
    const matchSelect = document.getElementById('match');

    matchSelect.innerHTML = '<option value="">Loading matches...</option>';
    matchSelect.disabled = true;
    document.getElementById('analyzeBtn').disabled = true;

    if (!provider || !compId || !seasonId) return;

    try {
        const res = await fetch(`/api/matches?provider=${provider}&competition=${compId}&season=${seasonId}`);
        const matches = await res.json();

        matchSelect.innerHTML = '<option value="">Select match...</option>';
        matches.forEach(m => {
            const label = `${m.home_team} vs ${m.away_team} (${m.score || ''}) - ${m.match_date || ''}`;
            matchSelect.innerHTML += `<option value="${m.match_id}">${label}</option>`;
        });
        matchSelect.disabled = false;

        matchSelect.onchange = () => {
            document.getElementById('analyzeBtn').disabled = !matchSelect.value;
        };
    } catch (e) {
        matchSelect.innerHTML = `<option value="">Failed to load - try another season</option>`;
        console.error(e);
    }
}

async function analyzeMatch() {
    const provider = document.getElementById('provider').value;
    const matchId = document.getElementById('match').value;
    const btn = document.getElementById('analyzeBtn');
    const loading = document.getElementById('loading');
    const errorDiv = document.getElementById('error');

    if (!provider || !matchId) return;

    btn.disabled = true;
    btn.textContent = 'Analyzing...';
    loading.classList.remove('hidden');
    errorDiv.classList.add('hidden');

    try {
        const res = await fetch('/api/analyze', {
            method: 'POST',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify({ provider, match_id: matchId }),
        });

        if (!res.ok) {
            const err = await res.json();
            throw new Error(err.detail || 'Analysis failed');
        }

        const data = await res.json();
        // Redirect to analysis page
        window.location.href = `/analysis/${data.analysis_id}`;
    } catch (e) {
        errorDiv.textContent = `Error: ${e.message}`;
        errorDiv.classList.remove('hidden');
        btn.disabled = false;
        btn.textContent = 'Analyze Match';
        loading.classList.add('hidden');
    }
}
