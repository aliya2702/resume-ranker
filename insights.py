"""
insights.py — Advanced AI-powered candidate analysis for ResumeRank.

Features:
  1. Candidate Clustering (Strong BE / FE / Generalist)
  2. Talent Pool Insights (skill prevalence, experience bands)
  3. JD Quality Analyzer (overloaded JDs, rare combos)
  4. Auto JD Improver (Claude-powered rewrite)
  5. Resume Strength Score (writing quality, structure)
  6. Risk Flags (gaps, hopping, over/under-qualification)
  7. Hidden Gems Detection (low score but unique strength)
  8. Transferable Skills Detection
  9. Ranking Stability Indicator (confidence bands)
"""

import re
import os
import json
import math
import logging
import requests
from typing import List, Dict, Any, Tuple

logger = logging.getLogger(__name__)

ANTHROPIC_API_KEY = os.environ.get("ANTHROPIC_API_KEY", "")
ANTHROPIC_API_URL = "https://api.anthropic.com/v1/messages"
CLAUDE_MODEL      = "claude-sonnet-4-20250514"


# ─────────────────────────────────────────────
# 1. Candidate Clustering
# ─────────────────────────────────────────────

CLUSTER_PROFILES = {
    "Strong Backend Engineer": {
        "required": ["python", "node", "django", "flask", "fastapi", "java", "go", "rest", "graphql"],
        "bonus":    ["docker", "kubernetes", "aws", "postgresql", "redis"],
        "penalty":  ["react", "vue", "angular", "css", "sass"],
    },
    "Frontend & UI Specialist": {
        "required": ["react", "vue", "angular", "javascript", "typescript", "css"],
        "bonus":    ["nextjs", "tailwind", "figma", "sass", "redux"],
        "penalty":  ["django", "flask", "kubernetes", "grpc"],
    },
    "Full-Stack Developer": {
        "required": ["javascript", "react", "node", "rest"],
        "bonus":    ["typescript", "python", "postgresql", "docker"],
        "penalty":  [],
    },
    "DevOps / Cloud Engineer": {
        "required": ["docker", "kubernetes", "aws", "ci/cd", "git"],
        "bonus":    ["azure", "gcp", "terraform", "ansible"],
        "penalty":  ["react", "css", "figma"],
    },
    "Data / ML Engineer": {
        "required": ["python"],
        "bonus":    ["pytorch", "tensorflow", "pandas", "sql", "spark"],
        "penalty":  ["css", "react", "vue"],
    },
    "Generalist / Versatile": {
        "required": [],
        "bonus":    [],
        "penalty":  [],
    },
}


def cluster_candidates(candidates: List[Dict]) -> List[Dict]:
    """
    Assign each candidate to a profile cluster based on their skill set.
    Adds a 'cluster' field to each candidate dict.
    """
    for candidate in candidates:
        all_skills = {
            s
            for skills in candidate.get("matched_skills", {}).values()
            for s in skills
        }
        best_cluster = "Generalist / Versatile"
        best_score = -1

        for cluster_name, profile in CLUSTER_PROFILES.items():
            if cluster_name == "Generalist / Versatile":
                continue
            required_matches = sum(1 for s in profile["required"] if s in all_skills)
            bonus_matches    = sum(1 for s in profile["bonus"]    if s in all_skills)
            penalty_matches  = sum(1 for s in profile["penalty"]  if s in all_skills)

            if not profile["required"] or required_matches == 0:
                continue

            score = (required_matches * 2) + bonus_matches - penalty_matches
            if score > best_score:
                best_score = score
                best_cluster = cluster_name

        candidate["cluster"] = best_cluster

    return candidates


def get_cluster_summary(candidates: List[Dict]) -> Dict[str, List[str]]:
    """Return {cluster_name: [candidate_names]}."""
    summary: Dict[str, List[str]] = {}
    for c in candidates:
        cluster = c.get("cluster", "Generalist / Versatile")
        summary.setdefault(cluster, []).append(c["name"])
    return summary


# ─────────────────────────────────────────────
# 2. Talent Pool Insights
# ─────────────────────────────────────────────

def compute_talent_pool_insights(
    candidates: List[Dict],
    jd_skills: Dict,
) -> Dict[str, Any]:
    """
    Analyze the talent pool and return aggregate insights.
    """
    n = len(candidates)
    if n == 0:
        return {}

    # Skill prevalence across all candidates
    skill_counter: Dict[str, int] = {}
    for c in candidates:
        for skills in c.get("matched_skills", {}).values():
            for s in skills:
                skill_counter[s] = skill_counter.get(s, 0) + 1

    # Top 8 most common skills
    top_skills = sorted(skill_counter.items(), key=lambda x: x[1], reverse=True)[:8]

    # Skills from JD that are rare in the pool (< 20% of candidates)
    jd_flat = [s for skills in jd_skills.values() for s in skills]
    rare_required = [
        s for s in jd_flat
        if skill_counter.get(s, 0) / n < 0.20
    ]

    # Score distribution
    scores = [c["score"] for c in candidates]
    avg_score   = round(sum(scores) / n, 1)
    max_score   = max(scores)
    min_score   = min(scores)

    # Score bands
    excellent = sum(1 for s in scores if s >= 80)
    strong    = sum(1 for s in scores if 65 <= s < 80)
    moderate  = sum(1 for s in scores if 45 <= s < 65)
    weak      = sum(1 for s in scores if s < 45)

    # Coverage: how many required JD skills does the pool collectively cover?
    pool_all_skills = {s for c in candidates for skills in c.get("matched_skills", {}).values() for s in skills}
    jd_skill_set    = set(jd_flat)
    coverage_pct    = round(len(jd_skill_set & pool_all_skills) / max(len(jd_skill_set), 1) * 100, 1)

    return {
        "total_candidates": n,
        "avg_score": avg_score,
        "max_score": max_score,
        "min_score": min_score,
        "score_bands": {
            "excellent_80_plus": excellent,
            "strong_65_79": strong,
            "moderate_45_64": moderate,
            "weak_below_45": weak,
        },
        "top_skills": [{"skill": s, "count": c, "pct": round(c / n * 100)} for s, c in top_skills],
        "rare_required_skills": rare_required[:6],
        "jd_skill_coverage_pct": coverage_pct,
        "hidden_gems_count": sum(1 for c in candidates if c.get("hidden_gem")),
    }


# ─────────────────────────────────────────────
# 3. JD Quality Analyzer
# ─────────────────────────────────────────────

RARE_SKILL_COMBOS = [
    (["kubernetes", "react"], "Kubernetes + React is an unusual DevOps/Frontend combo."),
    (["pytorch", "figma"], "ML + UX design combo is very rare in single candidates."),
    (["grpc", "vue"], "gRPC + Vue.js spans very different engineering specialties."),
]


def analyze_jd_quality(job_description: str, jd_skills: Dict) -> Dict[str, Any]:
    """
    Analyze a job description for quality issues.

    Returns a dict with:
      - skill_count: int
      - issues: list of {severity, message}
      - suggestions: list of str
      - quality_score: int (0–100)
    """
    jd_flat = [s for skills in jd_skills.values() for s in skills]
    skill_count = len(jd_flat)
    word_count  = len(job_description.split())

    issues = []
    suggestions = []

    # Too many required skills
    if skill_count > 10:
        issues.append({
            "severity": "high",
            "message": f"Too many required skills listed ({skill_count}). This significantly narrows the candidate pool.",
        })
        suggestions.append(
            f"Consider splitting into 'Required' and 'Nice-to-have' sections. "
            f"Removing {skill_count - 8} skills could increase matches by up to {(skill_count - 8) * 8}%."
        )

    # JD too short
    if word_count < 100:
        issues.append({"severity": "medium", "message": "Job description is very short. Candidates may not understand role expectations."})
        suggestions.append("Add responsibilities, team context, and expected outcomes (aim for 200+ words).")

    # JD too long / noisy
    if word_count > 800:
        issues.append({"severity": "low", "message": "Job description is very long. Key requirements may be buried."})
        suggestions.append("Consider trimming to 400–600 words with clear section headers.")

    # Rare skill combos
    for combo, msg in RARE_SKILL_COMBOS:
        if all(s in jd_flat for s in combo):
            issues.append({"severity": "medium", "message": f"Rare skill combination detected: {msg}"})

    # No soft skills mentioned
    soft_keywords = ["team", "communication", "collaboration", "agile", "scrum", "leadership"]
    if not any(kw in job_description.lower() for kw in soft_keywords):
        issues.append({"severity": "low", "message": "No team/soft skills mentioned. Top candidates often value culture fit signals."})
        suggestions.append("Add a line about team culture, work style, or collaboration expectations.")

    # No experience level specified
    exp_keywords = ["year", "experience", "senior", "junior", "mid", "entry", "lead"]
    if not any(kw in job_description.lower() for kw in exp_keywords):
        issues.append({"severity": "medium", "message": "Experience level is unclear. This may attract mismatched applicants."})
        suggestions.append("Specify seniority level or years of experience to improve match quality.")

    # Quality score: start at 100, deduct per issue
    deductions = {"high": 20, "medium": 10, "low": 5}
    quality_score = max(0, 100 - sum(deductions[i["severity"]] for i in issues))

    return {
        "skill_count": skill_count,
        "word_count": word_count,
        "issues": issues,
        "suggestions": suggestions,
        "quality_score": quality_score,
    }


# ─────────────────────────────────────────────
# 4. Auto JD Improver
# ─────────────────────────────────────────────

def improve_job_description(job_description: str) -> str:
    """
    Use Claude to rewrite the JD: clearer, unbiased, better structured.
    Falls back to rule-based improvements if no API key.
    """
    if not ANTHROPIC_API_KEY:
        return _rule_based_jd_improve(job_description)

    prompt = f"""You are an expert technical recruiter and HR specialist.

Rewrite the following job description to:
1. Be clearer and more structured (Role Summary, Responsibilities, Requirements, Nice-to-have)
2. Remove any gender-coded or exclusionary language
3. Separate "must-have" from "nice-to-have" skills to improve candidate pool size
4. Be concise but complete (aim for 350-500 words)
5. Add a brief sentence about team culture/collaboration

Original Job Description:
\"\"\"
{job_description}
\"\"\"

Return ONLY the improved job description, no commentary or preamble."""

    try:
        response = requests.post(
            ANTHROPIC_API_URL,
            headers={
                "x-api-key": ANTHROPIC_API_KEY,
                "anthropic-version": "2023-06-01",
                "content-type": "application/json",
            },
            json={
                "model": CLAUDE_MODEL,
                "max_tokens": 800,
                "messages": [{"role": "user", "content": prompt}],
            },
            timeout=20,
        )
        response.raise_for_status()
        return response.json()["content"][0]["text"].strip()
    except Exception as e:
        logger.warning("JD improvement API failed: %s", e)
        return _rule_based_jd_improve(job_description)


def _rule_based_jd_improve(jd: str) -> str:
    """Simple rule-based JD cleanup."""
    # Remove gender-coded words
    replacements = {
        r"\bninja\b": "engineer",
        r"\brock ?star\b": "senior engineer",
        r"\bguru\b": "expert",
        r"\bhe/she\b": "they",
        r"\bhe or she\b": "they",
    }
    improved = jd
    for pattern, replacement in replacements.items():
        improved = re.sub(pattern, replacement, improved, flags=re.IGNORECASE)

    # Add structure hint
    if "responsibilities" not in improved.lower() and "responsibilities" not in improved.lower():
        improved = "## Role Overview\n\n" + improved

    return improved.strip()


# ─────────────────────────────────────────────
# 5. Resume Strength Score
# ─────────────────────────────────────────────

IMPACT_PATTERNS = [
    r"\bimproved\b", r"\breduced\b", r"\bincreased\b", r"\bachieved\b",
    r"\bdelivered\b", r"\bbuilt\b", r"\blaunched\b", r"\boptimized\b",
    r"\bautomated\b", r"\bscaled\b", r"\bdrove\b", r"\bgenerated\b",
    r"\b\d+%\b", r"\b\$[\d,]+\b", r"\b\d+x\b",
]

STRUCTURE_MARKERS = [
    "experience", "education", "skills", "projects",
    "achievements", "certifications", "summary", "objective",
]


def compute_resume_strength(resume_text: str) -> Dict[str, Any]:
    """
    Score a resume on writing quality dimensions independent of job fit.

    Returns:
        {
          "overall": int (0-100),
          "clarity": int,
          "structure": int,
          "impact": int,
          "detail_level": int,
          "notes": [str]
        }
    """
    text_lower = resume_text.lower()
    words      = resume_text.split()
    word_count = len(words)

    notes = []

    # ── Structure score ───────────────────────
    structure_found = sum(1 for m in STRUCTURE_MARKERS if m in text_lower)
    structure_score = min(100, structure_found * 15)
    if structure_found < 3:
        notes.append("Resume lacks clear section headers (Experience, Skills, Education).")

    # ── Impact score ─────────────────────────
    impact_matches = sum(1 for p in IMPACT_PATTERNS if re.search(p, text_lower))
    impact_score   = min(100, impact_matches * 12)
    if impact_matches < 3:
        notes.append("Few impact statements detected. Add metrics: 'Reduced load time by 40%'.")

    # ── Clarity / length ─────────────────────
    if word_count < 200:
        clarity_score = 30
        notes.append("Resume appears very short. Add more detail about experience and projects.")
    elif word_count > 1200:
        clarity_score = 65
        notes.append("Resume is quite long. Consider trimming to 1 page (600–900 words).")
    else:
        clarity_score = 85

    # ── Detail level ─────────────────────────
    bullet_count = resume_text.count("•") + resume_text.count("-") + resume_text.count("·")
    detail_score = min(100, 40 + bullet_count * 5)
    if bullet_count < 5:
        notes.append("Consider using bullet points to list responsibilities and achievements clearly.")

    # ── Overall ──────────────────────────────
    overall = int((structure_score * 0.3) + (impact_score * 0.35) + (clarity_score * 0.2) + (detail_score * 0.15))

    return {
        "overall": overall,
        "clarity": clarity_score,
        "structure": structure_score,
        "impact": impact_score,
        "detail_level": detail_score,
        "notes": notes[:3],  # Top 3 notes
    }


# ─────────────────────────────────────────────
# 6. Risk Flags
# ─────────────────────────────────────────────

def detect_risk_flags(resume_text: str) -> List[str]:
    """
    Detect potential red flags in a resume text.

    Returns a list of human-readable flag strings.
    """
    flags = []
    text_lower = resume_text.lower()

    # Job hopping: many "year" patterns close together
    year_mentions = re.findall(r"\b(20\d{2})\b", resume_text)
    if len(year_mentions) >= 8:
        unique_years = len(set(year_mentions))
        if unique_years >= 6:
            flags.append("⚠️ Frequent job changes — multiple role transitions detected.")

    # Employment gap: years with large gaps between them
    years_as_ints = sorted(set(int(y) for y in year_mentions if 2000 <= int(y) <= 2025))
    for i in range(1, len(years_as_ints)):
        if years_as_ints[i] - years_as_ints[i - 1] > 2:
            flags.append(f"⚠️ Possible employment gap around {years_as_ints[i - 1]}–{years_as_ints[i]}.")
            break  # Report only first gap

    # Overqualification signals
    senior_titles = ["director", "vp ", "vice president", "head of", "principal", "chief"]
    if any(t in text_lower for t in senior_titles):
        flags.append("ℹ️ Senior-level background — verify role expectations match.")

    # Very short resume
    if len(resume_text.split()) < 150:
        flags.append("⚠️ Very brief resume — limited information available for assessment.")

    # No contact info
    if not re.search(r"[\w.+-]+@[\w-]+\.[a-z]{2,}", text_lower):
        flags.append("⚠️ No email address detected in resume.")

    return flags[:4]  # Cap at 4 flags


# ─────────────────────────────────────────────
# 7. Hidden Gems Detection
# ─────────────────────────────────────────────

def detect_hidden_gems(
    candidates: List[Dict],
    jd_skills: Dict,
) -> List[Dict]:
    """
    Flag candidates who score below 70 but have standout unique strengths.
    Adds 'hidden_gem' bool and 'gem_reason' to each candidate.
    """
    jd_flat   = {s for skills in jd_skills.values() for s in skills}
    avg_score = sum(c["score"] for c in candidates) / max(len(candidates), 1)

    for c in candidates:
        c["hidden_gem"]  = False
        c["gem_reason"]  = ""

        if c["score"] >= 70:
            continue  # Already high-performing, not a "hidden" gem

        candidate_skills = {s for skills in c.get("matched_skills", {}).values() for s in skills}

        # Unique skills not in JD but potentially valuable
        unique_skills = candidate_skills - jd_flat
        rare_valuable = [s for s in unique_skills if s in [
            "pytorch", "tensorflow", "rust", "go", "scala", "kotlin",
            "kubernetes", "graphql", "grpc", "redis", "elasticsearch"
        ]]

        resume_strength = c.get("resume_strength", {}).get("overall", 0)
        impact_score    = c.get("resume_strength", {}).get("impact", 0)

        # Mark as hidden gem if:
        if rare_valuable and c["score"] >= 45:
            c["hidden_gem"] = True
            c["gem_reason"] = f"Strong niche skills ({', '.join(rare_valuable[:3])}) not listed in JD but highly valuable."
        elif resume_strength >= 70 and c["score"] >= 40:
            c["hidden_gem"] = True
            c["gem_reason"] = "Well-structured, impact-driven resume suggests strong execution ability."
        elif impact_score >= 60 and c["score"] < avg_score:
            c["hidden_gem"] = True
            c["gem_reason"] = "High impact statements detected — demonstrates results-oriented background."

    return candidates


# ─────────────────────────────────────────────
# 8. Transferable Skills Detection
# ─────────────────────────────────────────────

SKILL_TRANSFER_MAP = {
    "java":         ["kotlin", "scala", "c#"],
    "python":       ["ruby", "perl"],
    "pytorch":      ["tensorflow", "keras", "mxnet"],
    "tensorflow":   ["pytorch", "keras"],
    "react":        ["vue", "angular", "svelte"],
    "vue":          ["react", "angular", "svelte"],
    "angular":      ["react", "vue"],
    "aws":          ["azure", "gcp"],
    "azure":        ["aws", "gcp"],
    "postgresql":   ["mysql", "sqlite"],
    "mysql":        ["postgresql", "sqlite"],
    "docker":       ["podman"],
    "redux":        ["zustand", "mobx", "recoil"],
    "webpack":      ["vite", "parcel", "esbuild"],
    "jest":         ["vitest", "mocha"],
    "node":         ["deno", "bun"],
    "javascript":   ["typescript"],
    "typescript":   ["javascript"],
}


def detect_transferable_skills(
    candidate_skills: Dict,
    jd_skills: Dict,
    missing_skills: Dict,
) -> List[Dict[str, str]]:
    """
    Identify cases where a candidate has a skill that transfers to a missing JD skill.

    Returns a list of {"have": skill, "transfers_to": jd_skill, "confidence": "high/medium"}
    """
    all_candidate = {s for skills in candidate_skills.values() for s in skills}
    all_missing   = {s for skills in missing_skills.values()   for s in skills}

    transfers = []
    for candidate_skill in all_candidate:
        equivalent_skills = SKILL_TRANSFER_MAP.get(candidate_skill, [])
        for equiv in equivalent_skills:
            if equiv in all_missing:
                confidence = "high" if candidate_skill in ["pytorch", "java", "aws", "react"] else "medium"
                transfers.append({
                    "have":         candidate_skill,
                    "transfers_to": equiv,
                    "confidence":   confidence,
                })

    return transfers[:4]  # Top 4 transfers


# ─────────────────────────────────────────────
# 9. Ranking Stability Indicator
# ─────────────────────────────────────────────

def compute_ranking_stability(candidates: List[Dict]) -> List[Dict]:
    """
    Assess how confident we are in each candidate's ranking position.

    Adds 'stability' ("high" | "medium" | "low") and 'stability_note' to each candidate.
    """
    scores = [c["score"] for c in candidates]
    n = len(scores)

    for i, c in enumerate(candidates):
        score = c["score"]

        # Score gap to the next candidate above and below
        gap_above = (scores[i - 1] - score) if i > 0     else 100
        gap_below = (score - scores[i + 1]) if i < n - 1 else 100

        if gap_above >= 10 and gap_below >= 10:
            stability      = "high"
            stability_note = "Clear separation from adjacent candidates."
        elif gap_above >= 5 or gap_below >= 5:
            stability      = "medium"
            stability_note = "Moderate score gap — ranking is likely stable."
        else:
            stability      = "low"
            stability_note = "Very close scores nearby — ranking could shift with more data."

        c["stability"]      = stability
        c["stability_note"] = stability_note

    return candidates


# ─────────────────────────────────────────────
# Master Enrichment Pipeline
# ─────────────────────────────────────────────

def enrich_candidates(
    candidates: List[Dict],
    job_description: str,
    jd_skills: Dict,
) -> Tuple[List[Dict], Dict]:
    """
    Run all enrichment passes on ranked candidates.

    Returns:
        (enriched_candidates, insights_dict)
    """
    # Per-candidate enrichment
    for c in candidates:
        resume_text = c.get("_resume_text", "")

        c["resume_strength"]     = compute_resume_strength(resume_text)
        c["risk_flags"]          = detect_risk_flags(resume_text)
        c["transferable_skills"] = detect_transferable_skills(
            c.get("matched_skills", {}),
            jd_skills,
            c.get("missing_skills", {}),
        )

    # Passes over the full pool
    candidates = cluster_candidates(candidates)
    candidates = detect_hidden_gems(candidates, jd_skills)
    candidates = compute_ranking_stability(candidates)

    # Pool-level insights
    insights = {
        "talent_pool": compute_talent_pool_insights(candidates, jd_skills),
        "jd_quality":  analyze_jd_quality(job_description, jd_skills),
        "clusters":    get_cluster_summary(candidates),
    }

    return candidates, insights
