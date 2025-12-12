"""
Monitoring API Routes for Cloud Run Service

Add these routes to your existing Cloud Run Flask app.
Reads drift monitoring reports from GCS and serves them via API.

Usage:
    from monitoring_routes import register_monitoring_routes
    register_monitoring_routes(app)
"""

from flask import Blueprint, jsonify, request
from google.cloud import storage
import json
import os
from datetime import datetime

monitoring_bp = Blueprint('monitoring', __name__, url_prefix='/monitoring')

# Configuration
GCS_BUCKET = os.environ.get("GCS_MODEL_BUCKET", "mlruns234")
MONITORING_PREFIX = os.environ.get("GCS_MONITORING_PREFIX", "monitoring")


def get_gcs_client():
    """Get GCS client."""
    return storage.Client()


def list_reports(client, report_type="json"):
    """List available monitoring reports from GCS."""
    bucket = client.bucket(GCS_BUCKET)
    prefix = f"{MONITORING_PREFIX}/reports/{report_type}/"
    
    blobs = list(bucket.list_blobs(prefix=prefix))
    reports = []
    
    for blob in blobs:
        filename = blob.name.split("/")[-1]
        if filename.endswith(('.json', '.html')):
            parts = filename.replace(".json", "").replace(".html", "").split("_")
            date_str = parts[-1] if parts else None
            reports.append({
                "filename": filename,
                "date": date_str,
                "path": blob.name,
                "updated": blob.updated.isoformat() if blob.updated else None
            })
    
    return sorted(reports, key=lambda x: x["date"] or "", reverse=True)


def get_report_content(client, report_path):
    """Get report content from GCS."""
    bucket = client.bucket(GCS_BUCKET)
    blob = bucket.blob(report_path)
    
    if not blob.exists():
        return None
    
    return blob.download_as_text()


# =============================================================================
# API ENDPOINTS
# =============================================================================

@monitoring_bp.route('/api/status')
def get_status():
    """Get current monitoring status."""
    try:
        client = get_gcs_client()
        reports = list_reports(client, "json")
        
        if not reports:
            return jsonify({
                "status": "NO_DATA",
                "message": "No monitoring reports found. Run drift monitoring first.",
                "lastUpdated": None
            })
        
        # Get latest report
        latest_path = reports[0]["path"]
        content = get_report_content(client, latest_path)
        
        if not content:
            return jsonify({"status": "ERROR", "message": "Could not read report"})
        
        report = json.loads(content)
        
        # Build response
        response = {
            "status": report.get("overall_status", "UNKNOWN"),
            "recommendedAction": report.get("recommended_action", "none"),
            "lastUpdated": report.get("timestamp"),
            "reportDate": report.get("report_date"),
            "alerts": report.get("alerts", []),
            "context": report.get("context", {}),
            
            "dataDrift": {
                "detected": report.get("data_drift", {}).get("dataset_drift", False),
                "driftShare": report.get("data_drift", {}).get("drift_share", 0),
                "driftedFeatures": report.get("data_drift", {}).get("n_drifted_features", 0),
                "totalFeatures": report.get("data_drift", {}).get("n_features_analyzed", 0),
                "topDriftedFeatures": report.get("data_drift", {}).get("drifted_features", [])[:5]
            },
            
            "predictionDrift": {
                "detected": report.get("prediction_drift", {}).get("drift_detected", False),
                "severity": report.get("prediction_drift", {}).get("drift_severity", "none"),
                "meanShiftPct": report.get("prediction_drift", {}).get("drift_metrics", {}).get("mean_shift_pct", 0),
                "stdChangePct": report.get("prediction_drift", {}).get("drift_metrics", {}).get("std_change_pct", 0)
            },
            
            "performance": {
                "status": report.get("performance", {}).get("status", "unknown"),
                "degraded": report.get("performance", {}).get("performance_degraded", False),
                "currentR2": report.get("performance", {}).get("current_metrics", {}).get("r2"),
                "currentMAE": report.get("performance", {}).get("current_metrics", {}).get("mae")
            }
        }
        
        return jsonify(response)
        
    except Exception as e:
        return jsonify({"status": "ERROR", "message": str(e)}), 500


@monitoring_bp.route('/api/history')
def get_history():
    """Get historical monitoring data."""
    try:
        client = get_gcs_client()
        reports = list_reports(client, "json")
        
        limit = request.args.get('limit', 30, type=int)
        history = []
        
        for report_info in reports[:limit]:
            try:
                content = get_report_content(client, report_info["path"])
                if not content:
                    continue
                    
                data = json.loads(content)
                history.append({
                    "date": data.get("report_date"),
                    "timestamp": data.get("timestamp"),
                    "status": data.get("overall_status"),
                    "driftShare": data.get("data_drift", {}).get("drift_share", 0),
                    "driftedFeatures": data.get("data_drift", {}).get("n_drifted_features", 0),
                    "predictionDrift": data.get("prediction_drift", {}).get("drift_detected", False),
                    "alertCount": len(data.get("alerts", []))
                })
            except:
                continue
        
        return jsonify(history)
        
    except Exception as e:
        return jsonify({"error": str(e)}), 500


@monitoring_bp.route('/api/report/<date>')
def get_report(date):
    """Get specific report details."""
    try:
        client = get_gcs_client()
        report_path = f"{MONITORING_PREFIX}/reports/json/monitoring_report_{date}.json"
        
        content = get_report_content(client, report_path)
        if not content:
            return jsonify({"error": "Report not found"}), 404
        
        return jsonify(json.loads(content))
        
    except Exception as e:
        return jsonify({"error": str(e)}), 500


@monitoring_bp.route('/api/features/<date>')
def get_feature_drift(date):
    """Get feature-level drift details."""
    try:
        client = get_gcs_client()
        report_path = f"{MONITORING_PREFIX}/reports/json/monitoring_report_{date}.json"
        
        content = get_report_content(client, report_path)
        if not content:
            return jsonify({"error": "Report not found"}), 404
        
        data = json.loads(content)
        feature_details = data.get("data_drift", {}).get("feature_details", {})
        
        # Sort by drift score
        sorted_features = sorted(
            [
                {
                    "name": name,
                    "driftDetected": details.get("drift_detected", False),
                    "driftScore": details.get("drift_score", 0),
                    "statTest": details.get("stattest_name", ""),
                    "threshold": details.get("stattest_threshold", 0)
                }
                for name, details in feature_details.items()
            ],
            key=lambda x: x["driftScore"],
            reverse=True
        )
        
        return jsonify(sorted_features)
        
    except Exception as e:
        return jsonify({"error": str(e)}), 500


@monitoring_bp.route('/report')
@monitoring_bp.route('/report/<date>')
def serve_html_report(date=None):
    """Serve Evidently HTML report."""
    try:
        client = get_gcs_client()
        
        if date:
            report_path = f"{MONITORING_PREFIX}/reports/html/data_drift_report_{date}.html"
        else:
            # Get latest
            reports = list_reports(client, "html")
            if not reports:
                return "<h1>No reports available</h1>", 404
            report_path = reports[0]["path"]
        
        content = get_report_content(client, report_path)
        if not content:
            return "<h1>Report not found</h1>", 404
        
        return content, 200, {"Content-Type": "text/html"}
        
    except Exception as e:
        return f"<h1>Error: {str(e)}</h1>", 500


@monitoring_bp.route('/api/baseline')
def get_baseline_info():
    """Get current baseline information."""
    try:
        client = get_gcs_client()
        bucket = client.bucket(GCS_BUCKET)
        
        # List baseline metadata files
        prefix = f"{MONITORING_PREFIX}/baselines/"
        blobs = list(bucket.list_blobs(prefix=prefix))
        
        metadata_files = [b for b in blobs if b.name.endswith('_metadata.json')]
        
        if not metadata_files:
            return jsonify({"status": "NO_BASELINE", "message": "No baseline found"})
        
        # Get latest
        latest = sorted(metadata_files, key=lambda x: x.name, reverse=True)[0]
        content = latest.download_as_text()
        metadata = json.loads(content)
        
        return jsonify({
            "version": metadata.get("metadata", {}).get("version"),
            "modelName": metadata.get("metadata", {}).get("model_name"),
            "createdAt": metadata.get("metadata", {}).get("created_at"),
            "baselineSource": metadata.get("metadata", {}).get("baseline_source"),
            "referenceSamples": metadata.get("metadata", {}).get("reference_samples"),
            "features": metadata.get("metadata", {}).get("n_features"),
            "dataSplits": metadata.get("metadata", {}).get("data_splits", {}),
            "performanceBaseline": metadata.get("performance_baseline", {})
        })
        
    except Exception as e:
        return jsonify({"error": str(e)}), 500


# =============================================================================
# REGISTRATION FUNCTION
# =============================================================================

def register_monitoring_routes(app):
    """Register monitoring blueprint with Flask app."""
    app.register_blueprint(monitoring_bp)
    print("  Monitoring routes registered at /monitoring/*")
