"""
Sports Intelligence Engine -- Main Orchestrator
Ties together all modules: ingestion -> processing -> detection -> statistics -> insights -> output.
"""

import json
import sys
import argparse
from pathlib import Path

from engine.ingestion import load_match_data
from engine.processing import process_match
from engine.detection import DetectionEngine
from engine.statistics import StatisticalAnalyzer
from engine.insights import InsightGenerator, report_to_json, report_to_text
from engine.clustering import PatternAnalyzer
from engine.visualization import generate_all_charts
from engine.sample_data import generate_football_match, generate_basketball_match


def analyze_match(source, sport=None, output_dir="output", visualize=True, cluster=True):
    """
    Full analysis pipeline.

    Args:
        source: File path, JSON string, or DataFrame.
        sport: 'football' or 'basketball' (auto-detected if None).
        output_dir: Directory for output files.
        visualize: Whether to generate charts.
        cluster: Whether to run ML clustering.

    Returns:
        MatchReport object.
    """
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)

    # 1. INGEST
    print("[1/6] Ingesting match data...")
    df = load_match_data(source, sport=sport)
    print(f"      Loaded {len(df)} events | Sport: {df.attrs['sport']} | "
          f"Teams: {', '.join(sorted(df['team'].unique()))}")

    # 2. PROCESS
    print("[2/6] Processing and aggregating...")
    ctx = process_match(df)
    print(f"      Duration: {ctx.duration_minutes:.0f} min | "
          f"Players: {ctx.player_stats['player'].nunique() if ctx.player_stats is not None else 0}")

    # 3. DETECT
    print("[3/6] Running detection engine...")
    engine = DetectionEngine(ctx)
    detected = engine.run_all()
    severity_counts = {}
    for e in detected:
        severity_counts[e.severity] = severity_counts.get(e.severity, 0) + 1
    print(f"      Detected {len(detected)} events: {severity_counts}")

    # 4. STATISTICS
    print("[4/6] Running statistical analysis...")
    analyzer = StatisticalAnalyzer(ctx)
    stat_insights = analyzer.run_all()
    print(f"      Generated {len(stat_insights)} statistical insights")

    # 5. ML CLUSTERING (optional)
    cluster_results = []
    if cluster:
        print("[5/6] Running pattern analysis (ML)...")
        pattern_analyzer = PatternAnalyzer(ctx)
        cluster_results = pattern_analyzer.analyze_play_patterns()
        print(f"      Found {len(cluster_results)} cluster patterns")
    else:
        print("[5/6] Skipping ML clustering")

    # 6. GENERATE INSIGHTS & REPORT
    print("[6/6] Generating insights and report...")
    generator = InsightGenerator(ctx, detected, stat_insights)
    report = generator.generate_report()

    # Add clustering results to report
    if cluster_results:
        report_dict = json.loads(report_to_json(report))
        report_dict["ml_patterns"] = [
            {
                "name": cr.name,
                "n_clusters": cr.n_clusters,
                "descriptions": cr.descriptions,
                "silhouette_score": cr.silhouette,
                "feature_importance": cr.feature_importance,
            }
            for cr in cluster_results
        ]
        with open(output_path / "report.json", "w", encoding="utf-8") as f:
            json.dump(report_dict, f, indent=2, default=str)
    else:
        with open(output_path / "report.json", "w", encoding="utf-8") as f:
            f.write(report_to_json(report))

    # Save text report
    text_report = report_to_text(report)
    with open(output_path / "report.txt", "w", encoding="utf-8") as f:
        f.write(text_report)

    # Generate visualizations
    if visualize:
        print("      Generating charts...")
        charts = generate_all_charts(ctx, detected, output_path / "charts")
        print(f"      Created {len(charts)} charts")

    print(f"\n{'='*60}")
    print(text_report)
    print(f"\nOutput saved to: {output_path.resolve()}")

    return report


def handle_provider(args):
    """Handle provider-based commands: list competitions, list matches, analyze."""
    from engine.providers import get_provider

    provider = get_provider(args.provider)

    # List competitions
    if args.list_competitions:
        comps = provider.list_competitions()
        print(f"\n[*] Available competitions ({args.provider}):\n")
        print(comps.to_string(index=False))
        print(f"\nTotal: {len(comps)} competition-seasons")
        print(f"\nUsage: python main.py --provider {args.provider} --competition <ID> --season <ID> --list-matches")
        return

    # List matches
    if args.list_matches:
        if not args.competition:
            print("Error: --competition is required for --list-matches")
            return
        season = args.season if args.season else None
        matches = provider.list_matches(
            competition_id=args.competition,
            season_id=season,
        )
        print(f"\n[*] Matches ({args.provider}):\n")
        print(matches.to_string(index=False))
        print(f"\nTotal: {len(matches)} matches")
        print(f"\nUsage: python main.py --provider {args.provider} --match <MATCH_ID>")
        return

    # Analyze a specific match
    if args.match:
        print(f"\n[*] Sports Intelligence Engine -- {args.provider.title()} Real Data\n")
        print(f"Fetching match {args.match} from {args.provider}...")
        df = provider.get_match_events(args.match)
        sport = provider.get_sport()
        analyze_match(df, sport=sport, output_dir=args.output,
                      visualize=not args.no_viz, cluster=not args.no_ml)
        return

    print(f"Error: Use --list-competitions, --list-matches, or --match with --provider")


def main():
    parser = argparse.ArgumentParser(
        description="Sports Intelligence Engine -- Transform match data into actionable insights",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  -- Demo mode --
  python main.py --demo football                  Run with sample football match
  python main.py --demo basketball                Run with sample basketball match

  -- Real data: StatsBomb (football) --
  python main.py --provider statsbomb --list-competitions
  python main.py --provider statsbomb --competition 43 --season 106 --list-matches
  python main.py --provider statsbomb --match 3869685     # World Cup 2022 Final

  -- Real data: NBA --
  python main.py --provider nba --list-competitions
  python main.py --provider nba --competition nba --season 2024-25 --list-matches
  python main.py --provider nba --match 0042400407        # NBA Finals game

  -- File input --
  python main.py --input match.csv
  python main.py --input match.json --sport basketball --no-viz
        """,
    )
    parser.add_argument("--input", "-i", help="Path to match data file (CSV or JSON)")
    parser.add_argument("--sport", "-s", choices=["football", "basketball"],
                        help="Sport type (auto-detected if not specified)")
    parser.add_argument("--output", "-o", default="output", help="Output directory (default: output)")
    parser.add_argument("--demo", choices=["football", "basketball"],
                        help="Run with sample data for the specified sport")
    parser.add_argument("--no-viz", action="store_true", help="Skip chart generation")
    parser.add_argument("--no-ml", action="store_true", help="Skip ML clustering")
    parser.add_argument("--export-sample", metavar="PATH",
                        help="Export sample data to CSV for inspection")

    # Provider-based commands
    parser.add_argument("--provider", "-p", choices=["statsbomb", "nba"],
                        help="Data provider for real match data")
    parser.add_argument("--competition", "-c", help="Competition/league ID")
    parser.add_argument("--season", help="Season ID")
    parser.add_argument("--match", "-m", help="Match/game ID to analyze")
    parser.add_argument("--list-competitions", action="store_true",
                        help="List available competitions from provider")
    parser.add_argument("--list-matches", action="store_true",
                        help="List matches for a competition/season")
    parser.add_argument("--season-analysis", action="store_true",
                        help="Run multi-match season analysis")
    parser.add_argument("--max-matches", type=int, default=20,
                        help="Max matches to analyze in season mode (default: 20)")

    # Web server
    parser.add_argument("--serve", action="store_true",
                        help="Start web dashboard server")
    parser.add_argument("--port", type=int, default=8000,
                        help="Port for web server (default: 8000)")

    args = parser.parse_args()

    if args.export_sample:
        sport = args.sport or "football"
        if sport == "football":
            df = generate_football_match()
        else:
            df = generate_basketball_match()
        df.to_csv(args.export_sample, index=False)
        print(f"Sample {sport} data exported to {args.export_sample}")
        return

    # Web server mode
    if args.serve:
        import uvicorn
        print(f"\n[*] Sports Intelligence Engine -- Web Dashboard")
        print(f"    Starting server on http://localhost:{args.port}")
        print(f"    Press Ctrl+C to stop\n")
        uvicorn.run("api:app", host="0.0.0.0", port=args.port, reload=False)
        return

    # Season analysis mode
    if hasattr(args, 'season_analysis') and args.season_analysis:
        if not args.provider or not args.competition or not args.season:
            print("Error: --provider, --competition, and --season required for --season-analysis")
            return
        from engine.multi_match import analyze_season, season_to_json
        print(f"\n[*] Sports Intelligence Engine -- Season Analysis\n")
        max_m = args.max_matches if hasattr(args, 'max_matches') else 20
        def on_progress(current, total):
            print(f"  Analyzing match {current}/{total}...")
        ctx = analyze_season(args.provider, args.competition, args.season,
                            max_matches=max_m, progress_callback=on_progress)
        output_path = Path(args.output)
        output_path.mkdir(parents=True, exist_ok=True)
        with open(output_path / "season_report.json", "w", encoding="utf-8") as f:
            f.write(season_to_json(ctx))
        print(f"\n  Analyzed {len(ctx.match_summaries)} matches")
        print(f"  Teams: {len(ctx.team_form)}")
        print(f"\n  League Table (by points):")
        for team, form in ctx.team_form.items():
            print(f"    {team:<25} P:{form['matches']:>2}  W:{form['wins']:>2}  D:{form['draws']:>2}  L:{form['losses']:>2}  "
                  f"GD:{form['goal_difference']:>+3}  Pts:{form['points']:>3}  Form: {form['form']}")
        print(f"\n  Report saved to: {output_path / 'season_report.json'}")
        return

    # Provider mode
    if args.provider:
        handle_provider(args)
        return

    if args.demo:
        print(f"\n[*] Sports Intelligence Engine -- {args.demo.title()} Demo\n")
        if args.demo == "football":
            source = generate_football_match()
        else:
            source = generate_basketball_match()
        analyze_match(source, sport=args.demo, output_dir=args.output,
                      visualize=not args.no_viz, cluster=not args.no_ml)

    elif args.input:
        print(f"\n[*] Sports Intelligence Engine\n")
        analyze_match(args.input, sport=args.sport, output_dir=args.output,
                      visualize=not args.no_viz, cluster=not args.no_ml)
    else:
        parser.print_help()
        print("\nTip: Run 'python main.py --demo football' to see a full demo.")
        print("     Run 'python main.py --provider statsbomb --list-competitions' for real data.")


if __name__ == "__main__":
    main()
