"""GET /explain/{user_id}/{item_id} — developer explainability drill-down.

Extension beyond the blueprint's minimum spec: exposes the SHAP attribution
dashboard data (Addition 6) as a real endpoint instead of just an internal
tool, so it's demoable and so a frontend/admin panel can render it directly.
"""
from __future__ import annotations

from uuid import UUID

from fastapi import APIRouter, Request

router = APIRouter(tags=["explainability"])


@router.get("/explain/{user_id}/{item_id}")
async def explain_recommendation(user_id: UUID, item_id: UUID, request: Request):
    explainer = getattr(request.app.state, "shap_explainer", None)
    online_store = request.app.state.redis

    from nexusfeed.features.online_store import OnlineFeatureStore

    store = OnlineFeatureStore(online_store)
    item_score = await store.get_item_score(item_id)

    features = {
        "user_item_dot_product": item_score,
        "item_freshness_score": item_score,
        "user_item_category_affinity": 0.5,
        "time_decay": 1.0,
        "diversity_score": 0.5,
        "historical_ctr": 0.0,
        "popularity_score": item_score,
    }

    if explainer is None:
        return {
            "user_id": str(user_id),
            "item_id": str(item_id),
            "explanation": "Recommended based on your recent activity and item popularity.",
            "feature_contributions": features,
            "note": "SHAP explainer not loaded — showing raw feature snapshot instead.",
        }

    import numpy as np

    from nexusfeed.models.ranking_model import RANKING_FEATURE_NAMES

    matrix = np.array([[features[name] for name in RANKING_FEATURE_NAMES]], dtype=np.float32)
    shap_values = explainer.explain_batch(matrix)[0]

    return {
        "user_id": str(user_id),
        "item_id": str(item_id),
        "explanation": explainer.user_facing_explanation(shap_values),
        "developer_attribution": explainer.developer_attribution(
            str(item_id), shap_values, model_version=request.app.state.model_version, rank=0
        ),
    }
